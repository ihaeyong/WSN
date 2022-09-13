import torch
from PIL import Image
import os, sys
import  numpy as np
import torch
from torchvision import transforms, datasets
from sklearn.utils import shuffle
import ipdb
from copy import deepcopy

# Base code adapted from https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglotNShot.py

class Omniglot(torch.utils.data.Dataset):
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        self.all_items = find_classes(os.path.join(self.root, self.processed_folder))
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):
        filename = self.all_items[index][0]
        img = str.join('/', [self.all_items[index][2], filename])

        target = self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.all_items)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "images_evaluation")) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, "images_background"))

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            file_processed = os.path.join(self.root, self.processed_folder)
            print("== Unzip from " + file_path + " to " + file_processed)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(file_processed)
            zip_ref.close()
        print("Download finished.")


def find_classes(root_dir):
    retour = []
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            if (f.endswith("png")):
                r = root.split('/')
                lr = len(r)
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    print("== Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    print("== Found %d classes" % len(idx))
    return idx



omniglot_dir = './data/'
omniglot_rotation_dir = './data/binary_omniglot_rotation'
root = omniglot_dir
########################################################################################################################
def get(seed=0, fixed_order=False, pc_valid=0):
    data = {}
    taskcla = []
    size = [1, 32, 32]

    ntasks = 100
    imgsz = 32

    if not os.path.isdir(omniglot_rotation_dir):
        os.makedirs(omniglot_rotation_dir)
        # Pre-load
        # Omniglot
        train_dataset = Omniglot(root, download=True,
                                transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                            lambda x: x.resize((imgsz, imgsz)),
                                                            lambda x: np.reshape(x, (imgsz, imgsz, 1)),
                                                            lambda x: np.transpose(x, [2, 0, 1]),
                                                            lambda x: x/255.])
                                )

        temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
        labels = []
        for (img, label) in train_dataset:
            labels.append(label)
            if label in temp.keys():
                temp[label].append(img)
            else:
                temp[label] = [img]

        labels = np.array(labels)
        labels = labels.reshape(-1, 20)[:1200]
        labels = np.concatenate((labels, labels, labels, labels), axis=1)
        labels = labels.reshape(100, 12, -1)

        images = []
        for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
            images.append(np.array(imgs))

        # as different class may have different number of imgs
        images = np.array(images)  # [[20 imgs],..., 1623 classes in total]
        images = images[:1200]
        # each character contains 20 imgs
        print('data shape:', images.shape)  # [1623, 20, 1, 32, 32]
        del temp  # Free memory

        # Rotate Images
        rot90 = np.rot90(images.transpose(3,4,0,1,2)).transpose(2,3,4,0,1)
        rot180 = np.rot90(rot90.transpose(3,4,0,1,2)).transpose(2,3,4,0,1)
        rot270 = np.rot90(rot180.transpose(3,4,0,1,2)).transpose(2,3,4,0,1)

        # Concatenate all images
        images = np.concatenate((images, rot90, rot180, rot270), axis=1)

        # Shuffle images per label
        for i in range(1200):
            images[i] = shuffle(images[i])

        # Slice to Train and Test
        images_train, images_test = images[:,:60,:,:,:], images[:, 60:, :, :, :]
        labels_train, labels_test = labels[:,:,:60], labels[:,:,60:]

        # Partition
        for i in range(100):
            task_images_train = images_train[i*12:(i+1)*12].reshape(-1, 1, 32, 32)
            task_labels_train = labels_train[i].reshape(-1) - (i * 12)
            task_images_test = images_test[i*12:(i+1)*12].reshape(-1, 1, 32, 32)
            task_labels_test = labels_test[i].reshape(-1) - (i * 12)

            task_images_train, task_labels_train = shuffle(task_images_train, task_labels_train)
            task_images_test, task_labels_test = shuffle(task_images_test, task_labels_test)

            data[i] = {}
            data[i]['name'] = 'omniglot-' + str(i+1)
            data[i]['ncla'] = 12
            data[i]['train'] = {}
            data[i]['valid'] = {}
            data[i]['test'] = {}

            data[i]['train']['x'] = deepcopy(task_images_train)
            data[i]['train']['y'] = deepcopy(task_labels_train)
            data[i]['valid']['x'] = deepcopy(task_images_train)
            data[i]['valid']['y'] = deepcopy(task_labels_train)
            data[i]['test']['x'] = deepcopy(task_images_test)
            data[i]['test']['y'] = deepcopy(task_labels_test)

        # Unify and Save
        for t in data.keys():
            for s in ['train','test','valid']:
                data[t][s]['x'] = torch.FloatTensor(data[t][s]['x']) #torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
                data[t][s]['y'] = torch.LongTensor(data[t][s]['y']).view(-1)
                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser(omniglot_rotation_dir),'data'+str(t)+s+'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser(omniglot_rotation_dir),'data'+str(t)+s+'y.bin'))

    # Load Binary Files
    data = {}
    ids = list(np.arange(100))
    print("Task order =", ids)
    for i in range(100):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test','valid']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser(omniglot_rotation_dir),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser(omniglot_rotation_dir),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'omniglot-'+str(ids[i])


    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size

# data, taskcla, size = get()

# ipdb.set_trace()