import numpy as np
import torch
import math
from model.base import *

class Net(BaseNet):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__(n_inputs, n_outputs, n_tasks, args)

        # steps for sharpness
        self.inner_steps = args.inner_batches

        # eta1: update step size of weight perturbation
        self.eta1 = args.eta1

        # eta2: learning rate of lambda(soft weight for basis)
        self.eta2 = args.eta2


    def forward(self, x, t):
        output = self.net.forward(x)

        if self.net.multi_head:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)

        return output

    def observe(self, x, y, t):
        if t != self.current_task:
            self.current_task = t

        for pass_itr in range(self.glances):
            self.iter += 1
            self.zero_grads()

            perm = torch.randperm(x.size(0))
            x = x[perm]
            y = y[perm]

            # get a batch by augmented incoming data with old task data, used for computing tiny-loss
            bx, by, bt = self.get_batch(x, y, t)

            # inner step of sharpness
            fast_weights = None
            inner_sz = math.ceil(len(x) / self.inner_steps)
            meta_losses = torch.zeros(self.inner_steps).float()
            k = 0

            for j in range(0, len(x), inner_sz):
                if j + inner_sz <= len(x):
                    batch_x = x[j: j + inner_sz]
                    batch_y = y[j: j + inner_sz]
                else:
                    batch_x = x[j:]
                    batch_y = y[j:]

                # samples for sharpness/look-ahead are from the current task
                fast_weights = self.update_weight(batch_x, batch_y, t, fast_weights)

                # samples for weight/lambdas update are from the current task and old tasks
                meta_losses[k] = self.meta_loss(bx, by, bt, fast_weights)
                k += 1

            # Taking the gradient step
            with torch.autograd.set_detect_anomaly(True):
                self.zero_grads()
                loss = torch.mean(meta_losses)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)

            if len(self.M_vec) > 0:

                # update the lambdas
                if self.args.method in ['dgpm', 'xdgpm']:
                    torch.nn.utils.clip_grad_norm_(self.lambdas.parameters(), self.args.grad_clip_norm)
                    if self.args.sharpness:
                        self.opt_lamdas.step()
                    else:
                        self.opt_lamdas_step()

                    for idx in range(len(self.lambdas)):
                        self.lambdas[idx] = nn.Parameter(torch.sigmoid(self.args.tmp * self.lambdas[idx]))

                # only use updated lambdas to update weight
                if self.args.method == 'dgpm':
                    self.net.zero_grad()
                    loss = self.meta_loss(bx, by, bt)  # Forward without weight perturbation
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)

                # train on the rest of subspace spanned by GPM
                self.train_restgpm()
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
                self.optimizer.step()

            else:
                self.optimizer.step()

            self.zero_grads()

        # only sample and push to replay buffer once for each task's stream
        # instead of pushing every epoch
        if self.real_epoch == 0:
            self.push_to_mem(x, y, torch.tensor(t))

        return loss


    def update_weight(self, x, y, t, fast_weights):
        """
            Add weight perturbation on the important subspace spanned by GPM for sharpness or look-ahead
            """
        loss = self.take_loss(x, y, t, fast_weights)
        if fast_weights is None:
            fast_weights = self.net.get_params()

        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        graph_required = self.args.second_order
        grads = list(torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required,
                                         allow_unused=True))

        # Get the projection of grads on the subspace spanned by GPM
        if len(self.M_vec) > 0:
            grads = self.grad_projection(grads)

        for i in range(len(grads)):
            if grads[i] is not None:
                grads[i] = torch.clamp(grads[i], min=-self.args.grad_clip_norm, max=self.args.grad_clip_norm)

        if self.args.sharpness:
            fast_weights = list(
                map(lambda p: p[1] + p[0] * self.eta1 if p[0] is not None else p[1], zip(grads, fast_weights)))
        else:
            fast_weights = list(
                map(lambda p: p[1] - p[0] * self.eta1 if p[0] is not None else p[1], zip(grads, fast_weights)))

        return fast_weights


    def grad_projection(self, grads):
        """
            get the projection of grads on the subspace spanned by GPM
            """
        j = 0
        for i in range(len(grads)):
            # only update conv weight and fc weight
            # ignore perturbations with 1 dimension (e.g. BN, bias)
            if grads[i] is None:
                continue
            if grads[i].ndim <= 1:
                continue
            if j < len(self.M_vec):
                if self.args.method in ['dgpm', 'xdgpm']:
                    # lambdas = torch.sigmoid(self.args.tmp * self.lambdas[j]).reshape(-1)
                    lambdas = self.lambdas[j]
                else:
                    lambdas = torch.ones(self.M_vec[j].shape[1])

                if self.cuda:
                    self.M_vec[j] = self.M_vec[j].cuda()
                    lambdas = lambdas.cuda()

                if grads[i].ndim == 4:
                    # rep[i]: n_samples * n_features
                    grad = grads[i].reshape(grads[i].shape[0], -1)
                    grad = torch.mm(torch.mm(torch.mm(grad, self.M_vec[j]), torch.diag(lambdas)), self.M_vec[j].T)
                    grads[i] = grad.reshape(grads[i].shape).clone()
                else:
                    grads[i] = torch.mm(torch.mm(torch.mm(grads[i], self.M_vec[j]), torch.diag(lambdas)), self.M_vec[j].T)

                j += 1

        return grads

    def zero_grads(self):
        self.optimizer.zero_grad()
        self.net.zero_grad()
        if len(self.M_vec) > 0 and self.args.method in ['dgpm', 'xdgpm']:
            self.lambdas.zero_grad()

    def define_lambda_params(self):
        assert len(self.M_vec) > 0

        # Setup learning parameters
        self.lambdas = nn.ParameterList([])
        for i in range(len(self.M_vec)):
            self.lambdas.append(nn.Parameter(self.args.lam_init * torch.ones((self.M_vec[i].shape[1]), requires_grad=True)))

        if self.cuda:
            self.lambdas = self.lambdas.cuda()

        return

    def update_opt_lambda(self, lr=None):
        if lr is None:
            lr = self.eta2
        self.opt_lamdas = torch.optim.SGD(list(self.lambdas.parameters()), lr=lr, momentum=self.args.momentum)

        return

    def opt_lamdas_step(self):
        """
            Performs a single optimization step, but change gradient descent to ascent
            """
        for group in self.opt_lamdas.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.opt_lamdas.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.data = (p.data + group['lr'] * d_p).clone()

        return

