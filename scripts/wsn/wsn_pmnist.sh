# Sample Usage: csnb_pmnist.sh [GPU_ID] [SEED_NUM] [SPARSITY]

export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES="$1"

python3 main_wsn_pmnist.py \
			--seed 1 \
			--optim adam \
			--gpu $1 \
			--lr 1e-3 \
			--lr_min 1e-6 \
			--lr_patience 6 \
			--lr_factor 2 \
			--batch_size_train 10 \
			--batch_size_test 10 \
			--n_hidden 100 \
			--pc_valid 0.1 \
			--sparsity $3 \
			--name hard
