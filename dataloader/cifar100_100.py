# Authorized by Haeyong Kang.

import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle


def get(data_path, args, seed=0, pc_valid=0.10, samples_per_task=-1):

    output_info = []
    size = [3, 32, 32]

    n_tasks = args.n_tasks
    n_outputs = 100
    n_cls = 100 // n_tasks
    if 100 % n_tasks != 0:
        n_cls_last = n_cls + 100 % n_tasks
    else:
        n_cls_last = n_cls

    path = os.path.join(data_path, 'cifar100')

    # Download
    if not os.path.isdir(path):
        os.makedirs(path)

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        # CIFAR100
        dat = {}
        dat['train'] = datasets.CIFAR100(data_path, train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = datasets.CIFAR100(data_path, train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))

        data = {}
        data['name'] = 'cifar100'
        data['train'] = {'x': [], 'y': []}
        data['test'] = {'x': [], 'y': []}

        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            for image, target in loader:
                n = target.cpu().numpy()[0]
                data[s]['x'].append(image)
                data[s]['y'].append(n)

            data[s]['x'] = torch.stack(data[s]['x']).view(-1, size[0], size[1], size[2])
            data[s]['y'] = torch.LongTensor(np.array(data[s]['y'], dtype=int)).view(-1)
            torch.save(data[s]['x'], os.path.join(os.path.expanduser(path), 'data_' + s + '_x.bin'))
            torch.save(data[s]['y'], os.path.join(os.path.expanduser(path), 'data_' + s + '_y.bin'))


    # Load binary files
    dat = {}
    data = {}

    for i in range(n_tasks):
        data[i] = {}
        data[i]['name'] = 'cifar100-' + str(i)
        if i == n_tasks - 1:
            data[i]['ncla'] = 100 - n_cls * i
        else:
            data[i]['ncla'] = n_cls
        data[i]['train'] = {'x': [], 'y': []}
        data[i]['test'] = {'x': [], 'y': []}

    for s in ['train', 'test']:
        dat[s] = {'x': [], 'y': []}
        dat[s]['x'] = torch.load(os.path.join(os.path.expanduser(path), 'data_' + s + '_x.bin'))
        dat[s]['y'] = torch.load(os.path.join(os.path.expanduser(path), 'data_' + s + '_y.bin'))
        assert dat[s]['y'].shape[0] == dat[s]['x'].shape[0]
        for k in range(dat[s]['y'].shape[0]):
            image = dat[s]['x'][k]
            n = dat[s]['y'][k].cpu().numpy()
            nn = n // n_cls
            data[nn][s]['x'].append(image)
            data[nn][s]['y'].append(n)

            #if nn == n_tasks:
            #    nn -= 1
            #    data[nn][s]['x'].append(image)
            #    data[nn][s]['y'].append(n - nn * n_cls)
            #    assert n - nn * n_cls < n_cls_last
            #else:
            #    data[nn][s]['x'].append(image)
            #    data[nn][s]['y'].append(n % n_cls)

        for t in range(n_tasks):
            data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
            data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
            assert data[t]['ncla'] == len(np.unique(data[t][s]['y'].numpy()))


    # Validation
    for t in data.keys():
        r = np.arange(data[t]['train']['x'].size(0))
        r = np.array(shuffle(r, random_state=seed), dtype=int)
        nvalid = int(pc_valid * len(r))
        ivalid = torch.LongTensor(r[:nvalid])
        itrain = torch.LongTensor(r[nvalid:])
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y'] = data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x'] = data[t]['train']['x'][itrain].clone()
        data[t]['train']['y'] = data[t]['train']['y'][itrain].clone()


    # Others
    n = 0
    for t in data.keys():
        output_info.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, output_info, size, n_tasks, n_outputs
