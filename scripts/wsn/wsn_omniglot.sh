export PYTHONPATH=$HOME/workspaces/continual-lr
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES="$1"

python3 main_wsn_omniglot.py \
		--optim adam \
		  --seed 1 \
		  --gpu $1 \
		  --lr 1e-3 \
		  --lr_min 1e-6 \
		  --lr_patience 6 \
		  --lr_factor 2 \
		  --n_epochs 100 \
		  --batch_size_train 64 \
		  --batch_size_test 64 \
		  --model large \
		  --pc_valid 0.05 \
		  --sparsity $3