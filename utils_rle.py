# Modified by Haeyong Kang.
import numpy as np
import torch

import sys
import numpy as np
from copy import deepcopy
from utils import safe_save

import time

PRECISION_LOWER_LIMIT = 0
PRECISION_UPPER_LIMIT = 10
# Compression algorithm is based on google encoded polyline format. 
# https://github.com/amit1rrr/numcompress/blob/master/numcompress/numcompress.py
# Used for N-Dimensional array. Separates dimension and the series.
# Should have ASCII value between (0, 63) to avoid overlapping with regular compress output.
SEPARATOR = ','
def compress(series, precision=3):
    last_num = 0
    result = ''

    if not isinstance(series, list):
        raise ValueError('Input to compress should be of type list.')

    if not isinstance(precision, int):
        raise ValueError('Precision parameter needs to be a number.')

    if precision < PRECISION_LOWER_LIMIT or precision > PRECISION_UPPER_LIMIT:
        raise ValueError('Precision must be between 0 to 10 decimal places.')

    is_numerical_series = all(isinstance(item, int) or isinstance(item, float) for item in series)

    if not is_numerical_series:
        raise ValueError('All input list items should either be of type int or float.')

    if not series:
        return result

    # Store precision value at the beginning of the compressed text
    result += chr(precision + 63)

    for num in series:
        # 1. Take the initial signed value: -179.9832104
        diff = num - last_num
        # 2. Take the decimal value and multiply it by 1e5, rounding the result: -17998321
        diff = int(round(diff * (10 ** precision)))
        # 3. Convert the decimal value to binary.
        # Note that a negative value must be calculated
        # using its two's complement by inverting the binary value and adding one to the result:
        # 00000001 00010010 10100001 11110001
        # 11111110 11101101 01011110 00001110
        # 11111110 11101101 01011110 00001111

        # 4. Left-shift the binary value one bit: 11111101 11011010 10111100 00011110
        # 5. If the original decimal value is negative, invert this encoding:
        # 00000010 00100101 01000011 11100001
        diff = ~(diff << 1) if diff < 0 else diff << 1

        while diff >= 0x20:
            # 8. OR each value with 0x20 if another bit chunk follows:
            # 100001 111111 110000 101010 100010 000001
            result += (chr((0x20 | (diff & 0x1f)) + 63))

            # 6. Break the binary value out into 5-bit chunks (starting from the right hand side):
            # 00001 00010 01010 10000 11111 00001
            # 7. Place the 5-bit chunks into reverse order: 00001 11111 10000 01010 00010 00001
            diff >>= 5
        # 9. Convert each value to decimal: 33 63 48 42 34 1
        # 10. Add 63 to each value:
        # Convert each value to its ASCII equivalent: `~oia@

        result += (chr(diff + 63))
        last_num = num

    return result


def decompress(text):
    result = []
    index = last_num = 0

    if not isinstance(text, str):
        raise ValueError('Input to decompress should be of type str.')

    if not text:
        return result

    # decode precision value
    precision = ord(text[index]) - 63
    index += 1

    if precision < PRECISION_LOWER_LIMIT or precision > PRECISION_UPPER_LIMIT:
        raise ValueError('Invalid string sent to decompress. Please check the string for accuracy.')

    while index < len(text):
        index, diff = decompress_number(text, index)
        last_num += diff
        result.append(last_num)

    result = [round(item * (10 ** (-precision)), precision) for item in result]
    return result


def decompress_number(text, index):
    result = 1
    shift = 0

    while True:
        # ord fun : chr -> ord('a') = 97
        b = ord(text[index]) - 63 - 1
        index += 1
        result += b << shift
        shift += 5

        if b < 0x1f:
            break

    return index, (~result >> 1) if (result & 1) != 0 else (result >> 1)


def compress_ndarray(series, precision=3):
    shape = "*".join(map(str, series.shape))
    series_compressed = compress(series.flatten().tolist(), precision)
    return '{}{}{}'.format(shape, SEPARATOR, series_compressed )


def decompress_ndarray(text):
    shape_str, series_text = text.split(SEPARATOR)
    shape = tuple(int(dimension) for dimension in shape_str.split('*'))
    series = decompress(series_text)
    return np.array(series).reshape(*shape)



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


def comp_decomp_mask(per_task_masks, task_id, device):

    per_task_mask = deepcopy(per_task_masks[task_id])
    bit_mask_ratio = []
    bit_mask_sparsity = []
    # Compression algorithm is based on google encoded polyline format.
    for key in per_task_masks[task_id].keys():
        if 'weight' in key:

            weight = per_task_mask[key].cpu().numpy().astype(np.bool)
            original_size = sum(sys.getsizeof(i) for i in weight.reshape(-1))
            com_series = compress_ndarray(weight)
            compressed_size = sys.getsizeof(com_series)

            decom_series = decompress_ndarray(com_series)
            assert (decom_series == weight).all()
            comp_ratio = ((original_size - compressed_size) * 100.0) / original_size
            per_task_mask[key] = torch.Tensor(decom_series).to(device)
            bit_mask_ratio.append(comp_ratio)
            bit_mask_sparsity.append(compressed_size / original_size)

    print("task_id:{}, comp_ratio:{}, bitmap_sparsity:{}".format(task_id, np.mean(bit_mask_ratio), np.mean(bit_mask_sparsity)))
    return per_task_mask, np.mean(bit_mask_sparsity)



def Polyline(per_task_masks, per_bits):

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
        # polyline encoding / decoding
        tstart=time.time()
        for key in int_masks.keys():
            if 'weight' in key:
                weight = int_masks[key].cpu().numpy()
                if weight.max() > 255:
                    import ipdb; ipdb.set_trace()
                    
                original_size = sys.getsizeof(weight.astype(np.bool)) * per_bits
                com_series = compress_ndarray(weight)
                compressed_size = sys.getsizeof(com_series)

                compression_ratio = ((original_size - compressed_size) * 100.0) / original_size

                decom_series = decompress_ndarray(com_series)
                assert (decom_series == weight).all()
                    
                total_org_size += original_size
                total_com_size += compressed_size
                
                print(task_id, key, "{}".format(compression_ratio))
                assert (weight == decom_series).all()

        elapsed_time = (time.time() - tstart) * 1000
        print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
        task_sparsity[tid]['sparsity'] = total_com_size/total_org_size
        task_sparsity[tid]['time'] = elapsed_time
        print(tid, task_sparsity[tid])
    return task_sparsity


if __name__ == '__main__':


    per_task_masks = np.load('./results2/csnb_tiny_data/csnb_tiny_dataset_resnet18_SEED_4_LR_0.001_SPARSITY_0.5.pertask.npy', allow_pickle=True).item()
        
    bit_mask_ratio = []
    bit_mask_sparsity = []

    total_org_size = 0
    total_com_size = 0
    #per_bits = 2 # sparsity: 0.96 for 40 tasks
    #per_bits = 3 # sparsity: 0.65 for 40 tasks
    #per_bits = 4 # sparsity: 0.50 for 40 tasks
    #per_bits = 5 # sparsity: 0.49 for 40 tasks
    #per_bits = 6 # sparsity: 0.40 for 40 tasks
    
    per_bits = 7 # sparsity: 0.36 for 40 tasks
    #per_bits = 7 # sparsity: 0.39 for 7 tasks
    #per_bits = 7 # sparsity: 0.38 for 14 tasks
    #per_bits = 7 # sparsity: 0.37 for 21 tasks
    #per_bits = 7 # sparsity: 0.37 for 28 tasks
    #per_bits = 7 # sparsity: 0.36 for 35 tasks

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
                original_size = sys.getsizeof(weight.astype(np.bool)) * per_bits
                com_series = compress_ndarray(weight)
                compressed_size = sys.getsizeof(com_series)

                compression_ratio = ((original_size - compressed_size) * 100.0) / original_size

                decom_series = decompress_ndarray(com_series)
                assert (decom_series == weight).all()
                    
                total_org_size += original_size
                total_com_size += compression_ratio
                
                bit_mask_ratio.append(compression_ratio)
                bit_mask_sparsity.append(compressed_size / original_size)
                
                print(task_id, key, "{}".format(compression_ratio))
                assert (weight == decom_series).all()

    print("int_mask_avg_ratio:{}, sparsity:{}, avg_sparcity:{}".format(
        np.mean(bit_mask_ratio), np.mean(bit_mask_sparsity), total_com_size/total_org_size))
                

