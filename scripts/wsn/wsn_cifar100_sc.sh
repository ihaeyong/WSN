export PYTHONPATH=$HOME/workspaces/continual-lr
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES="$1"

python3 main_wsn_cifar_superclass100.py \
		--dataset cifar100_superclass100 \
		--optim adam \
		--seed 0 \
		--gpu $1 \
		--lr 1e-3 \
		--lr_min 1e-5 \
		--lr_patience 6 \
		--lr_factor 2 \
		--pc_valid 0.05 \
		--n_epochs 50 \
		--t_order 0 \
		--batch_size_train 64 \
		--batch_size_test 256 \
		--model lenet \
		--sparsity $3 \
		--name hard 
