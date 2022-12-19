# Sample Usage: wsn_cifar100_100.sh [GPU_ID] [SEED_NUM] [SPARSITY]
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES="$1"

python3 main_wsn_cifar100_100.py \
    --optim adam \
    --dataset cifar100_100 \
    --seed $2 \
    --gpu $1 \
    --lr 1e-3 \
    --lr_min 1e-6 \
    --lr_patience 6 \
    --lr_factor 2 \
    --n_epochs 200 \
    --batch_size_train 64 \
    --batch_size_test 64 \
    --model alexnet \
    --pc_valid 0.05 \
    --sparsity $3 \
    --encoding huffman
