export PYTHONPATH=$HOME/workspaces/continual-lr
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES="$1"

python3 main_wsn_cifar100_100.py     \
		--optim adam                  \
		--dataset cifar100_100        \
		--seed 1                  \
		--gpu $1                      \
		--lr 1e-3                     \
		--lr_min 1e-6                 \
		--lr_patience 6               \
		--lr_factor 2                 \
		--n_epochs 200                \
		--batch_size_train 64         \
		--batch_size_test 256         \
		--model alexnet               \
		--pc_valid 0.05               \
		--sparsity $3                 \
		--soft 0.0 \
		--soft_grad 0.0               \
		--name hard

