## [Forget-free Continual Learning with Winning Subnetworks](https://scholar.google.com/citations?view_op=view_citation&hl=ja&user=v_tUj4oAAAAJ&citation_for_view=v_tUj4oAAAAJ:qxL8FJ1GzNcC)-ICML2022 
-----------------------------------------------------------
This is the official implementation of WSN in the paper in Pytorch.


## Dependency
- [PyTorch](https://pytorch.org/) > 1.5

## Dataset
- Permuted MNIST (available current version)
- 5 Datasets (available current version)
- Omniglot Rotation (available current version)
- CIFAR-100 Split (available current version)
- CIFAR-100 Superclass (available current version)
- TinyImageNet (available current version)

## Installation
To execute the codes for running experiments, run the following.
```python
pip install -r requirements.txt
```

## Training
We provide several training examples with this repositories:

- To train WSN on Permuted MNIST on GPU [GPU_ID] with seed number [SEED] and sparsity [SPARSITY], simply run the following
```bash
>> ./scripts/wsn/wsn_pmnist.sh [GPU_ID] [SEED] [SPARSITY]
```

- To train WSN on Cifar100-100 on GPU [GPU_ID] with seed number [SEED] and sparsity [SPARSITY], simply run the following
```bash
>> ./scripts/wsn/wsn_cifar100_100.sh [GPU_ID] [SEED] [SPARSITY]
```

## References 
```
Haeyong Kang, Rusty John Lloyd Mina, Sultan Rizky Hikmawan Madjid, 
Jaehong Yoon, Mark Hasegawa-Johnson, Sung Ju Hwang, Chang D Yoo., 
Forget-free Continual Learning with Winning Subnetworks-ICML2022 
```

