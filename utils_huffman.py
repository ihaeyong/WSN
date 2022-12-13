# Authorized by Haeyong Kang.

import sys
import numpy as np
from utils import safe_save
from copy import deepcopy
import torch

import time
# https://towardsdatascience.com/huffman-encoding-python-implementation-8448c3654328
# https://github.com/YCAyca/Data-Structures-and-Algorithms-with-Python/blob/main/Huffman_Encoding/huffman.py

DEBUG = False
# A Huffman Tree Node
class Node:
    def __init__(self, prob, symbol, left=None, right=None):
        # probability of symbol
        self.prob = prob

        # symbol 
        self.symbol = symbol

        # left node
        self.left = left

        # right node
        self.right = right

        # tree direction (0/1)
        self.code = ''

""" A helper function to print the codes of symbols by traveling Huffman Tree"""
codes = dict()

def Calculate_Codes(node, val=''):
    # huffman code for current node
    newVal = val + str(node.code)

    if(node.left):
        Calculate_Codes(node.left, newVal)
    if(node.right):
        Calculate_Codes(node.right, newVal)

    if(not node.left and not node.right):
        codes[node.symbol] = newVal

    return codes

""" A helper function to calculate the probabilities of symbols in given data"""
def Calculate_Probability(data):
    symbols = dict()
    for element in data:
        if symbols.get(element) == None:
            symbols[element] = 1
        else:
            symbols[element] += 1
    return symbols

""" A helper function to obtain the encoded output"""
def Output_Encoded(data, coding):
    encoding_output = []
    for c in data:
      #  print(coding[c], end = '')
        encoding_output.append(coding[c])

    string = ''.join([str(item) for item in encoding_output])
    return string

""" A helper function to calculate the space difference between compressed and non compressed data"""
def Total_Gain(data, coding):
    before_compression = len(data) * 2 # total bit space to stor the data before compression
    after_compression = 0
    symbols = coding.keys()
    for symbol in symbols:
        count = data.count(symbol)
        after_compression += count * len(coding[symbol]) #calculate how many bit is required for that symbol in total
    print("==Space usage before compression (in bits):", before_compression) if DEBUG else None
    print("==Space usage after compression (in bits):",  after_compression) if DEBUG else None
    return after_compression / before_compression

def Huffman_Encoding(data):
    symbol_with_probs = Calculate_Probability(data)
    symbols = symbol_with_probs.keys()
    probabilities = symbol_with_probs.values()
    print("==symbols: ", symbols) if DEBUG else None
    print("==probabilities: ", probabilities) if DEBUG else None

    nodes = []
    # converting symbols and probabilities into huffman tree nodes
    for symbol in symbols:
        nodes.append(Node(symbol_with_probs.get(symbol), symbol))

    while len(nodes) > 1:
        # sort all the nodes in ascending order based on their probability
        nodes = sorted(nodes, key=lambda x: x.prob)
        # for node in nodes:
        #      print(node.symbol, node.prob)

        # pick 2 smallest nodes
        right = nodes[0]
        left = nodes[1]

        left.code = 0
        right.code = 1

        # combine the 2 smallest nodes to create new node
        newNode = Node(left.prob+right.prob, left.symbol+right.symbol, left, right)

        nodes.remove(left)
        nodes.remove(right)
        nodes.append(newNode)

    huffman_encoding = Calculate_Codes(nodes[0])
    print("symbols with codes", huffman_encoding) if DEBUG else None
    gain = Total_Gain(data, huffman_encoding)
    encoded_output = Output_Encoded(data,huffman_encoding)
    return encoded_output, nodes[0], gain


def Huffman_Decoding(encoded_data, huffman_tree):
    tree_head = huffman_tree
    decoded_output = []
    for x in encoded_data:
        if x == '1':
            huffman_tree = huffman_tree.right
        elif x == '0':
            huffman_tree = huffman_tree.left
        try:
            if huffman_tree.left.symbol == None and huffman_tree.right.symbol == None:
                pass
        except AttributeError:
            decoded_output.append(huffman_tree.symbol)
            huffman_tree = tree_head

    string = ''.join([str(item) for item in decoded_output])
    return string

def comp_decomp_mask_huffman(per_task_masks, task_id, device):

    per_task_mask = deepcopy(per_task_masks[task_id])
    bit_mask_ratio = []
    bit_mask_sparsity = []
    # Compression algorithm is based on google encoded polyline format.
    for key in per_task_masks[task_id].keys():
        if 'weight' in key:

            weight = per_task_mask[key].cpu().numpy().astype(np.bool)
            weight_size = per_task_mask[key].cpu().numpy().shape
            original_size = sum(sys.getsizeof(i) for i in weight.reshape(-1))

            weight_str = ''.join(str(x) for x in weight.reshape(-1).astype(np.uint8))

            encoding, tree, gain = Huffman_Encoding(weight_str)
            compressed_size = sys.getsizeof(encoding) + sys.getsizeof(tree)
            compression_ratio = ((original_size - compressed_size) * 100.0) / original_size

            decoding = Huffman_Decoding(encoding,tree)
            decoding_str = ' '.join(x for x in decoding)
            decoding_w = np.array(decoding_str.split()).astype(np.uint8).reshape(weight.shape)
            assert (weight == decoding_w).all()

            per_task_mask[key] = torch.Tensor(decoding_w).to(device)
            bit_mask_ratio.append(compression_ratio)
            bit_mask_sparsity.append(compressed_size / original_size)


    print("task_id:{}, comp_ratio:{}, bitmap_sparsity:{}".format(task_id, np.mean(bit_mask_ratio), np.mean(bit_mask_sparsity)))
    return per_task_mask, np.mean(bit_mask_sparsity)


def dec2bin_mask(int_masks, bits=10):
    if bits > 1 and False:
        import ipdb; ipdb.set_trace()

    mask = 2 ** torch.arange(bits - 1, -1, -1).to(int_masks.device).long()
    dec = int_masks.unsqueeze(-1).bitwise_and(mask).ne(0).long()
    if len(int_masks.size()) > 3:
        dec = dec.permute(4, 0, 1, 2, 3)
    else:
        dec = dec.permute(2, 0, 1)
    return dec.long()


def bin2dec_mask(key, per_task_mask, int_masks, bit=None, per_bits=None):
    if bit > 0 and False:
        import ipdb; ipdb.set_trace()
    mask = 2 ** torch.arange(per_bits - 1, -1, -1).to(int_masks.device).long()
    int_masks = deepcopy(per_task_mask)[key].long() * mask[-(bit+1)]
    
    return int_masks.long()

def accum_int_mask(bit, task_id, per_task_masks, int_masks=None, per_bits=None):
    
    # Accumulate prime masks
    if int_masks is None:
        int_masks = deepcopy(per_task_masks[task_id])
        for key in int_masks.keys():
            if "last" in key:
                if key in curr_head_keys:
                    continue
            if 'weight' in key:
                int_masks[key] = bin2dec_mask(
                    key=key,
                    per_task_mask=per_task_masks[task_id],
                    int_masks=int_masks[key],
                    bit=bit, per_bits=per_bits)
    else:
        for key in int_masks.keys():
            if "last" in key:
                if key in curr_head_keys:
                    continue
            if 'weight' in key:
                int_masks[key] += bin2dec_mask(
                    key=key,
                    per_task_mask=per_task_masks[task_id],
                    int_masks=int_masks[key],
                    bit=bit, per_bits= per_bits)

    return int_masks

def Huffman(per_task_masks, per_bits):

    task_sparsity = {}
    num_tasks = len(per_task_masks.keys())
    max_tasks = num_tasks + 1
    # [0 2 4 6]....
    task_id_list = [task_id for task_id in per_task_masks.keys() if task_id % per_bits == 0 and task_id < max_tasks]
    for task_id in task_id_list:
        int_masks = None
        for bit, tid in enumerate(range(task_id, task_id + per_bits)):
            if (tid+1) > num_tasks:
                break
            print("task_id:{} -> tid:{}, bit:{}".format(task_id, tid, bit))
            int_masks = accum_int_mask(bit, tid, per_task_masks, int_masks, per_bits)

        task_sparsity[tid] = {} 
        total_org_size = 0
        total_com_size = 0
        # huffman encoding / decoding
        tstart=time.time()
        for key in int_masks.keys():
            if 'weight' in key:
                weight = int_masks[key].cpu().numpy()
                if weight.max() > 255:
                    import ipdb; ipdb.set_trace()
                weight_size = int_masks[key].cpu().numpy().shape
                weight_str = ''.join(chr(x+33) for x in weight.reshape(-1))

                original_size = sys.getsizeof(weight.astype(np.bool)) * per_bits
                encoding, tree, gain = Huffman_Encoding(weight_str)

                compressed_size = sys.getsizeof(encoding)+sys.getsizeof(tree)+sys.getsizeof(weight.shape)

                compression_ratio = ((original_size - compressed_size) * 100.0) / original_size

                decoding = Huffman_Decoding(encoding,tree)
                decoding_str = ' '.join(str(ord(x)-33) for x in decoding)
                decoding_w = np.array(decoding_str.split()).astype(np.uint8).reshape(weight.shape)
                total_org_size += original_size
                total_com_size += compressed_size

                assert (weight == decoding_w).all()
                print(task_id, key, "comp_ratio:{}".format(compression_ratio))

        elapsed_time = (time.time() - tstart) * 1000
        print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
        task_sparsity[tid]['sparsity'] = total_com_size/total_org_size
        task_sparsity[tid]['time'] = elapsed_time
        print(tid, task_sparsity[tid])
    return task_sparsity


if __name__ == '__main__' :
    """ First Test """
    if False:
        #data = "AAAAAAABCCCCCCDDEEEEE"
        data = "011100"
        #data = "0111030010191012040261"
        print(data, len(data) * 8)
        encoding, tree, gain = Huffman_Encoding(data)
        print("Encoded output", encoding)
        print("Decoded Output", Huffman_Decoding(encoding,tree))

        origin_size = sys.getsizeof(data)
        compre_size = sys.getsizeof(encoding) + sys.getsizeof(tree)
        print("Encoded gain:", gain, "sys gain:", compre_size/origin_size)

    elif False:
        binary = np.zeros((3,3,3)).astype(np.bool)
        data = "304076401"
        encoding, tree, gain = Huffman_Encoding(data)
        print("Encoded output", encoding)
        print("Decoded Output", Huffman_Decoding(encoding,tree))

        origin_size = sys.getsizeof(binary)
        compre_size = sys.getsizeof(encoding) + sys.getsizeof(tree)
        print("Encoded gain:", gain, "sys gain:", compre_size/origin_size, "origin_size:", origin_size, "compre_size:", compre_size)

    elif True:
        per_task_masks = np.load('./results2/csnb_tiny_data/csnb_tiny_dataset_resnet18_SEED_4_LR_0.001_SPARSITY_0.5.pertask.npy', allow_pickle=True).item()
        
        bit_mask_ratio = []
        bit_mask_sparsity = []

        total_org_size = 0
        total_com_size = 0
        #per_bits = 2 # sparsity: 0.78 for 40 tasks
        #per_bits = 3 # sparsity: 0.55 for 40 tasks
        #per_bits = 4 # sparsity: 0.48 for 40 tasks
        #per_bits = 5 # sparsity: 0.40 for 40 tasks
        #per_bits = 6 # sparsity: 0.35 for 40 tasks
        
        per_bits = 7 # sparsity: 0.32 for 40 tasks
        #per_bits = 7 # sparsity: 0.43  for 7 tasks
        #per_bits = 7 # sparsity: 0.39  for 14 tasks
        #per_bits = 7 # sparsity: 0.36  for 21 tasks
        #per_bits = 7 # sparsity: 0.35  for 28 tasks
        #per_bits = 7 # sparsity: 0.34  for 35 tasks

        num_tasks = len(per_task_masks.keys())
        max_tasks = num_tasks + 1
        # [0 2 4 6]....
        task_id_list = [task_id for task_id in per_task_masks.keys() if task_id % per_bits == 0and task_id < max_tasks]
        for task_id in task_id_list:
            int_masks = None
            for bit, tid in enumerate(range(task_id, task_id + per_bits)):
                if (tid+1) > num_tasks:
                    break
                print("task_id:{} -> tid:{}, bit:{}".format(task_id, tid, bit))
                int_masks = accum_int_mask(bit, tid, per_task_masks, int_masks, per_bits)

            # rle encoding and decoding
            for key in int_masks.keys():

                if 'weight' in key:
                    weight = int_masks[key].cpu().numpy()
                    if weight.max() > 255:
                        import ipdb; ipdb.set_trace()
                    weight_size = int_masks[key].cpu().numpy().shape
                    weight_str = ''.join(chr(x+33) for x in weight.reshape(-1))

                    original_size = sys.getsizeof(weight.astype(np.bool)) * per_bits
                    encoding, tree, gain = Huffman_Encoding(weight_str)

                    compressed_size = sys.getsizeof(encoding) + sys.getsizeof(tree) + sys.getsizeof(weight.shape)

                    compression_ratio = ((original_size - compressed_size) * 100.0) / original_size

                    decoding = Huffman_Decoding(encoding,tree)
                    decoding_str = ' '.join(str(ord(x)-33) for x in decoding)
                    decoding_w = np.array(decoding_str.split()).astype(np.uint8).reshape(weight.shape)

                    total_org_size += original_size
                    total_com_size += compression_ratio

                    bit_mask_ratio.append(compression_ratio)
                    bit_mask_sparsity.append(compressed_size / original_size)

                    print(task_id, key, gain, "{}".format(compression_ratio))
                    assert (weight == decoding_w).all()

        print("int_mask_avg_ratio:{}, sparsity:{}, avg_sparcity:{}".format(
            np.mean(bit_mask_ratio), np.mean(bit_mask_sparsity), total_com_size/total_org_size))

    elif False:
        int_masks = np.load('./results2/csnb_tiny_data/csnb_tiny_dataset_resnet18_SEED_4_LR_0.001_SPARSITY_0.5.intmask.npy', allow_pickle=True).item()

        # rle encoding and decoding
        for key in int_masks.keys():

            if 'weight' in key:
                import ipdb; ipdb.set_trace()
                weight = int_masks[key].cpu().numpy().astype(np.uint64)
                weight_size = int_masks[key].cpu().numpy().shape
                weight_str = ' '.join(str(x) for x in weight.reshape(-1))
                
                if False:
                    original_size = sum(sys.getsizeof(i) for i in weight.astype(np.uint64).reshape(-1))
                else:
                    original_size = sys.getsizeof(weight)
                encoding, tree, gain = Huffman_Encoding(weight_str)

                compressed_size = sys.getsizeof(encoding) + sys.getsizeof(tree) + sys.getsizeof(weight.shape)

                compression_ratio = ((original_size - compressed_size) * 100.0) / original_size

                decoding = Huffman_Decoding(encoding,tree)
                decoding_str = ' '.join(x for x in decoding)
                decoding_w = np.array(decoding_str.split()).astype(np.uint64).reshape(weight.shape)

                total_org_size += original_size
                total_com_size += compression_ratio

                bit_mask_ratio.append(compression_ratio)
                bit_mask_sparsity.append(compressed_size / original_size)

                print(task_id, key, gain, "{}".format(compression_ratio))
                assert (weight == decoding_w).all()

        print("int_mask_avg_ratio:{}, sparsity:{}, avg_sparcity:{}".format(
            np.mean(bit_mask_ratio), np.mean(bit_mask_sparsity), total_com_size/total_org_size))
        

    else:
        per_task_masks = np.load('./results2/csnb_tiny_data/csnb_tiny_dataset_resnet18_SEED_4_LR_0.001_SPARSITY_0.5.pertask.npy', allow_pickle=True).item()
        
        bit_mask_ratio = []
        bit_mask_sparsity = []

        total_org_size = 0
        total_com_size = 0
        for task_id in per_task_masks.keys():
            print("task_id:{}".format(task_id))
            #if target_id == task_id:
            per_task_mask = per_task_masks[task_id]

            # rle encoding and decoding
            for key in per_task_masks[task_id].keys():

                if 'weight' in key:
                    weight = per_task_mask[key].cpu().numpy().astype(np.uint8)
                    weight_size = per_task_mask[key].cpu().numpy().shape
                    weight_str = ''.join(str(x) for x in weight.reshape(-1))
                    #import ipdb; ipdb.set_trace()
                    if False:
                        original_size = sum(sys.getsizeof(i) for i in weight.astype(np.bool).reshape(-1))
                    else:
                        import ipdb; ipdb.set_trace()
                        original_size = sys.getsizeof(weight.astype(np.bool).reshape(-1))
                    encoding, tree, gain = Huffman_Encoding(weight_str)

                    compressed_size = sys.getsizeof(encoding) + sys.getsizeof(tree) + sys.getsizeof(weight.shape)

                    compression_ratio = ((original_size - compressed_size) * 100.0) / original_size

                    decoding = Huffman_Decoding(encoding,tree)
                    decoding_str = ' '.join(x for x in decoding)
                    decoding_w = np.array(decoding_str.split()).astype(np.uint8).reshape(weight.shape)

                    total_org_size += original_size
                    total_com_size += compression_ratio

                    bit_mask_ratio.append(compression_ratio)
                    bit_mask_sparsity.append(compressed_size / original_size)

                    print(task_id, key, gain, "{}".format(compression_ratio))
                    assert (weight == decoding_w).all()

        #safe_save('results2/bit_masks', bit_masks)
        #safe_save('results2/com_masks', com_masks)
        print("int_mask_avg_ratio:{}, sparsity:{}, avg_sparcity:{}".format(
            np.mean(bit_mask_ratio), np.mean(bit_mask_sparsity), total_com_size/total_org_size))



