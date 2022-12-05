## Forget-free Continual Learning with Winning Subnetworks-ICML2022 
-----------------------------------------------------------
This is the official implementation of WSN in the paper in Pytorch.


## Dependency
- [PyTorch](https://pytorch.org/) > 1.5

## Dataset
- Permuted MNIST (available current version)
- 5 Datasets 
- Omniglot Rotation
- CIFAR-100 Split
- CIFAR-100 Superclass
- TinyImageNet

## Installation
To execute the codes for running experiments, run the following.
```python
pip install -r requirements.txt
```

## Training
We provide several training examples with this repositories:

- To train WSN on Permuted MNIST on GPU [GPU_ID] with seed number [SEED] and sparsity [SPARSITY], simply run the following
```bash
scripts/wsn/wsn_pmnist.sh [GPU_ID] [SEED] [SPARSITY]
```

## References 
```
Forget-free Continual Learning with Winning Subnetworks-ICML2022 
```
