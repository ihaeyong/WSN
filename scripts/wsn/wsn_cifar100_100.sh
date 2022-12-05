export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES="$1"


python3 main_csnb_cifar100_100.py \
    --optim adam \
    --dataset cifar100_100 \
    --seed 1 \
    --gpu $1 \
    --lr 1e-3 \
    --lr_min 1e-6 \
    --lr_patience 6 \
    --lr_factor 2 \
    --n_epochs 50 \
    --batch_size_train 64 \
    --batch_size_test 64 \
    --model alexnet \
    --pc_valid 0.05 \
    --sparsity 0.5 \
    --encoding huffman
