export PYTHONPATH=$HOME/workspaces/continual-lr
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES="$1"

python3 main_wsn_five_dataset_resnet18.py \
		--optim adam \
		--seed 1 \
		--gpu $1 \
		--lr 1e-1 \
		--lr_min 1e-5 \
		--lr_patience 6 \
		--lr_factor 2 \
		--pc_valid 0.05 \
		--n_epochs 100 \
		--batch_size_train 64 \
		--batch_size_test 64 \
		--model resnet18 \
		--sparsity $3