import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
import h5py
import ipdb

########################################################################################################################

def get(seed=0, fixed_order=False, pc_valid=0, tasknum = 50):
    data = {}
    taskcla = []
    size = [1, 28, 28]
    # tasknum = 25

    if not os.path.isdir('dat/binary_omniglot/'):
        os.makedirs('dat/binary_omniglot')
        
        filename = 'Permuted_Omniglot_task50.pt'
        filepath = os.path.join(os.getcwd(), 'dataloaders/')
        f = torch.load(os.path.join(filepath,filename))

        # ipdb.set_trace()
        ncla_dict = {}
        for i in range(tasknum):
            # print(i)
            data[i] = {}
            data[i]['name'] = 'omniglot-{:d}'.format(i)
            data[i]['ncla'] = (torch.max(f['Y']['train'][i]) + 1).int().item()
            ncla_dict[i] = data[i]['ncla']
            
            for s in ['train', 'test']:
                
                data[i][s] = {'x': [], 'y': []}

                image = f['X']['train'][i]
                target = f['Y']['train'][i]
                
                index_arr = np.arange(len(image))
                np.random.shuffle(index_arr)
            
                data[i][s]['x'] = image[index_arr]
                data[i][s]['y'] = target[index_arr]

            # "Unify" and save
            for s in ['train', 'test']:
#                 data[i][s]['x'] = torch.stack(data[i][s]['x']).view(-1, size[0], size[1], size[2])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
                torch.save(data[i][s]['x'],os.path.join(os.path.expanduser('dat/binary_omniglot'), 'data' + str(i) + s + 'x.bin'))
                torch.save(data[i][s]['y'],os.path.join(os.path.expanduser('dat/binary_omniglot'), 'data' + str(i) + s + 'y.bin'))
        torch.save(ncla_dict, os.path.join(os.path.expanduser('dat/binary_omniglot'), 'ncla_dict.pt'))

    else:
        data_num = 0
        ncla_dict = torch.load(os.path.join(os.path.expanduser('dat/binary_omniglot'), 'ncla_dict.pt'))

        # Load binary files
        for i in range(tasknum):
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[i]['ncla'] = ncla_dict[i]
            data[i]['name'] = 'omniglot-{:d}'.format(i)

            # Load
            for s in ['train', 'test']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser('dat/binary_omniglot'), 'data' + str(i) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser('dat/binary_omniglot'), 'data' + str(i) + s + 'y.bin'))

    # Validation
    data_num = 0
    for t in data.keys():
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'].clone()
        data[t]['valid']['y'] = data[t]['train']['y'].clone()
        # print(data[t]['train']['x'].shape[0])
        data_num += data[t]['train']['x'].shape[0]
    # print(data_num)

    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t//5, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size

########################################################################################################################
get(seed=0)