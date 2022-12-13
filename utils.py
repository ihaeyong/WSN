# Authorized by Haeyong Kang.
# from _typeshed import OpenBinaryModeReading
# from ast import dump
# from genericpath import exists
from collections import defaultdict, namedtuple
import os
import struct
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from heapq import heappush, heappop, heapify

import pickle, json

Node = namedtuple('Node', 'freq value left right')
Node.__lt__ = lambda x, y: x.freq < y.freq

def save_pickle(save_file, f_dict, s_type='pickle'):

    if s_type == 'pickle':
        with open(save_file, 'wb') as fw:
            pickle.dump(f_dict, fw)
    else:
        with open(save_file, 'w') as f:
            json.dump(f_dict, f)

def load_pickle(save_file, s_type='pickle'):

    if s_type == 'pickle':
        with open(save_file, 'rb') as fr:
            f_dict = pickle.load(fr)
    else:
        with open(save_file) as data_file:
            f_file = json.load(data_file)

        f_dict = dict()
        for i, l in enumerate(f_file):
            f_dict[i] = f_file[str(i)]

    return f_dict

def safe_load(file_name, cuda=False):

    try:
        if cuda:
            result = np.load(file_name, allow_pickle=True).item()
        else:
            result = np.load(file_name)
        print("sucessfully to load", file_name)
    except:
        print("failed to load", file_name)
        import ipdb; ipdb.set_trace()
        return

    return result


def safe_save(save_path, data):

    # Make sure that the folders exists
    hierarchy = save_path.split("/")
    for i in range(1, len(hierarchy)):
        folder = "/".join(hierarchy[:i])

        if not os.path.exists(folder):
            os.mkdir(folder)

    np.save(save_path, data)

    print("Saved {}".format(save_path))

def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if ('conv' in name or 'fc' in name) and not 'w_m' in name:
            tensor = p.data.cpu().numpy()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params
            print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')

def huffman_encode_model(model, directory='encodings/'):
    os.makedirs(directory, exist_ok=True)
    original_total = 0
    compressed_total = 0
    print(f"{'Layer':<15} | {'original':>10} {'compressed':>10} {'improvement':>11} {'percent':>7}")
    print('-'*70)
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight = param.data.cpu().numpy()
            shape = weight.shape
            form = 'csr' if shape[0] < shape[1] else 'csc'
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)

            # Encode
            t0, d0 = huffman_encode(mat.data, name+f'_{form}_data', directory)
            t1, d1 = huffman_encode(mat.indices, name+f'_{form}_indices', directory)
            t2, d2 = huffman_encode(calc_index_diff(mat.indptr), name+f'_{form}_indptr', directory)

            # Print statistics
            original = mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
            compressed = t0 + d0 + t1 + d1 + t2 + d2

            print(f"{name:<15} | {original:10} {compressed:10} {original / compressed:>10.2f}x {100 * compressed / original:>6.2f}%")
        else: # bias
            # Note that we do not huffman encode bias
            bias = param.data.cpu().numpy()
            bias.dump(f'{directory}/{name}')

            # Print statistics
            original = bias.nbytes
            compressed = original

            print(f"{name:<15} | {original:10} {compressed:10} {original / compressed:>10.2f}x {100 * compressed / original:>6.2f}%")
        original_total += original
        compressed_total += compressed

    print('-'*70)
    print(f"{'total':15} | {original_total:>10} {compressed_total:>10} {original_total / compressed_total:>10.2f}x {100 * compressed_total / original_total:>6.2f}%")

def huffman_encode(arr, prefix, save_dir='./'):
    """
    Encodes numpy array 'arr' and saves it to 'save_dir'
    The names of binary files are prefixed with 'prefix'
    returns the number of bytes for the tree and the data after the compression
    """
    # Infer dtype
    dtype = str(arr.dtype)

    # Calculate frequency in arr
    freq_map = defaultdict(int)
    convert_map = {'float32':float, 'int32':int}
    for value in np.nditer(arr):
        value = convert_map[dtype](value)
        freq_map[value] += 1

    # Make heap
    heap = [Node(frequency, value, None, None) for value, frequency in freq_map.items()]
    heapify(heap)

    # Merge nodes
    while(len(heap) > 1):
        node1 = heappop(heap)
        node2 = heappop(heap)
        merged = Node(node1.freq + node2.freq, None, node1, node2)
        heappush(heap, merged)

    # Generate code value mapping
    value2code = {}

    def generate_code(node, code):
        if node is None: return
        if node.value is not None:
            value2code[node.value] = code
            return
        generate_code(node.left, code + '0')
        generate_code(node.right, code + '1')

    root = heappop(heap)
    generate_code(root, '')

    # Path to save location
    directory = Path(save_dir)

    # Dump data
    data_encoding = ''.join(value2code[convert_map[dtype](value)] for value in np.nditer(arr))
    datasize = dump(data_encoding, directory/f'{prefix}.bin')

    # Dump codebook (huffmann tree)
    codebook_encoding = encode_huffman_tree(root, dtype)
    treesize = dump(codebook_encoding, directory/f'{prefix}_codebook.bin')

    return treesize, datasize

# Encode / decode huffman tree
def encode_huffman_tree(root, dtype):
    """
    Encodes a huffman tree to strig of '0's and '1's
    """
    converter = {'float32':float2bitstr, 'int32':int2bitstr}
    code_list = []
    def encode_node(node):
        if node.value is not None: # leaf node
            code_list.append('1')
            lst = list(converter[dtype](node.value))
            code_list.append(lst)
        else:
            code_list.append('0')
            encode_node(node.left)
            encode_node(node.right)
        encode_node(root)
        return ''.join(code_list)

# Helper functions for converting between bit string and (float or int)
def float2bitstr(f):
    four_bytes = struct.pack('>f', f) # bytes
    return ''.join (f'{byte:08b}' for byte in four_bytes) # string of '0's and '1's

def bitstr2float(bitstr):
    byte_arr = bytearray(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>f', byte_arr)[0]

def int2bitstr(integer):
    four_bytes = struct.pack('>I', integer) # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes) # string of '0's and '1's

def bitstr2int(bitstr):
    byte_arr = bytearray(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>I', byte_arr)[0]

# Functions for calculating / reconstructing index diff
def calc_index_diff(indptr): return indptr[1:] - indptr[:-1]

# https://stackoverflow.com/a/43357954/6365092
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
