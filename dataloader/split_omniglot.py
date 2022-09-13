import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
import h5py


########################################################################################################################

def get(seed=0, fixed_order=False, pc_valid=0, tasknum = 50):
    data = {}
    taskcla = []
    size = [1, 28, 28]
    tasknum = 50

    if not os.path.isdir('dat/binary_omniglot_split/'):
        os.makedirs('dat/binary_omniglot_split')
        
        filename = 'Permuted_Omniglot_task50.pt'
        filepath = 'dataloaders'
        f = torch.load(os.path.join(filepath,filename))
        ncla_dict = {}
        for i in range(tasknum):
            data[i] = {}
            data[i]['name'] = 'omniglot-{:d}'.format(i)
            data[i]['ncla'] = (torch.max(f['Y']['train'][i]) + 1).int().item()
            ncla_dict[i] = data[i]['ncla']
                
            data[i]['train'] = {'x': [], 'y': []}
            data[i]['test'] = {'x': [], 'y': []}
            data[i]['valid'] = {'x': [], 'y': []}

            image = f['X']['train'][i]
            target = f['Y']['train'][i]

            index_arr = np.arange(len(image))
            np.random.shuffle(index_arr)
            train_ratio = (len(image)//10)*8
            valid_ratio = (len(image)//10)*1
            test_ratio = (len(image)//10)*1
            
            train_idx = index_arr[:train_ratio]
            valid_idx = index_arr[train_ratio:train_ratio+valid_ratio]
            test_idx = index_arr[train_ratio+valid_ratio:]

            data[i]['train']['x'] = image[train_idx]
            data[i]['train']['y'] = target[train_idx]
            data[i]['valid']['x'] = image[valid_idx]
            data[i]['valid']['y'] = target[valid_idx]
            data[i]['test']['x'] = image[test_idx]
            data[i]['test']['y'] = target[test_idx]
            

            # "Unify" and save
            for s in ['train', 'test', 'valid']:
#                 data[i][s]['x'] = torch.stack(data[i][s]['x']).view(-1, size[0], size[1], size[2])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
                torch.save(data[i][s]['x'],os.path.join(os.path.expanduser('dat/binary_omniglot_split'), 'data' + str(i) + s + 'x.bin'))
                torch.save(data[i][s]['y'],os.path.join(os.path.expanduser('dat/binary_omniglot_split'), 'data' + str(i) + s + 'y.bin'))
        torch.save(ncla_dict, os.path.join(os.path.expanduser('dat/binary_omniglot_split'), 'ncla_dict.pt'))

    else:
        
        ncla_dict = torch.load(os.path.join(os.path.expanduser('dat/binary_omniglot_split'), 'ncla_dict.pt'))
        # Load binary files
        # ids=list(np.arange(tasknum))
        ids=list(shuffle(np.arange(50),random_state=seed))
        print('Task order =',ids)
        for i in range(tasknum):
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test','valid'])
            data[i]['ncla'] = ncla_dict[ids[i]]
            data[i]['name'] = 'omniglot-{:d}'.format(i)

            # Load
            for s in ['train', 'test', 'valid']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser('dat/binary_omniglot_split'), 
                                                          'data' + str(ids[i]) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser('dat/binary_omniglot_split'), 
                                                          'data' + str(ids[i]) + s + 'y.bin'))


    # Others
    n = 0
    data_num = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
        print('Task %d: %d classes'%(t+1,data[t]['ncla']))
#         print(data[t]['train']['x'].shape[0])
        data_num += data[t]['train']['x'].shape[0]
    print(data_num)
        
    data['ncla'] = n
    print(n)
    return data, taskcla, size

########################################################################################################################

