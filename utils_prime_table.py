# Authorized by Haeyong Kang.
# To get prime mask tables (prime subnetworks)

import os
import numpy as np
import math

from multiprocessing import Process, Pool
import multiprocessing as mp
from itertools import combinations, permutations, repeat

from utils import *

# Whether it is prime number or not
def is_prime(n):
    for i in range(2, int(math.sqrt(n))+1):
        if n % i == 0:
            return False
    return True

# Get primes and product of primes
def get_primes(num_primes, log_flag=False, log_scaler=1.0, prime_scaler=1.0):
    primes = []
    num = 2
    prod_prime = 0
    while len(primes) < num_primes:
        if is_prime(num):
            if log_flag:
                primes.append(np.log2(num * prime_scaler) * log_scaler)
                prod_prime += np.log2(num * prime_scaler) * log_scaler
            elif False:
                primes.append(np.log(num * prime_scaler) * log_scaler)
                prod_prime += np.log(num * prime_scaler) * log_scaler
            else:
                primes.append(num)
                prod_prime *= num
        num+=1
    return primes, prod_prime

# Get mode table of product of primes
def get_mod_prods_forward(task_id=0,primes=None,remainder_dict=None,num_tasks=None,debug=False):
    items = [i for i in range(0, task_id+1)]
    mod_prod_list = [0.0]
    prev_mod_prod_list = []
    for tid in range(task_id):
        prev_mod_prod_list=list(set(prev_mod_prod_list) | set(remainder_dict[tid]))

    for select in range(0,len(items)):
        comb = list(combinations(items, select))
        if debug:
            print("task_id:{},item:{},comb:{},select:{}".format(task_id+1,items,comb,select))
        prod_tables = []
        for com in comb:
            prod = 0.0
            for idx in com:
                if task_id != idx:
                    prod += primes[idx]

            print("=== comp:{}, prod:{}".format(com, prod)) if debug else None
            if prod not in mod_prod_list:
                if prod not in prev_mod_prod_list:
                    mod_prod_list.append(prod)

        print() if debug else None

    print(mod_prod_list) if debug else None
    print(prev_mod_prod_list) if debug else None
    print() if debug else None

    return mod_prod_list


# Get mode table of product of primes
def get_mod_prods(task_id=0, primes=None, remainder_dict=None, num_tasks=None, debug=False):
    items = [i for i in range(0, num_tasks)]
    mod_prod_list = []

    for select in range(0,num_tasks):
        comb = list(combinations(items, select))
        if debug:
            print("task_id:{},item:{},comb:{},select:{}".format(task_id+1,items,comb,select))
        prod_tables = []
        for com in comb:
            prod = 0.0
            for idx in com:
                if task_id != idx:
                    prod += primes[idx]

            print("=== comp:{}, prod:{}".format(com, prod)) if debug else None
            if prod not in mod_prod_list:
                mod_prod_list.append(prod)

        print() if debug else None

    print(mod_prod_list) if debug else None
    print() if debug else None

    return mod_prod_list

# Get remainder dictionary
def get_remainder(primes, log_scaler):
    remainder_dict = {}
    # decoding masks
    for task_id, prime in enumerate(primes):
        mod_prod_list = get_mod_prods(task_id=task_id, primes=primes,
                                      remainder_dict=remainder_dict,
                                      num_tasks=len(primes),debug=True)
        remainder_dict[task_id] = mod_prod_list

    num_tasks = len(primes)
    save_pickle('./data/num_task{}_scale{}.pickle'.format(num_tasks,log_scaler),
                remainder_dict)
    print('done')


# Multi-processing for Prime Mod Tables
def func(task_id, sparse, num_tasks, log_flag, log_scaler):

    num_sparse_tasks = num_tasks*sparse
    primes, prod_prime = get_primes(num_primes=num_sparse_tasks,
                                    log_flag=log_flag,
                                    log_scaler=log_scaler)

    if sparse > 1:
        sparse_primes=[]
        for i, prm in enumerate(primes):
            if i % sparse != 0:
                sparse_primes.append(prm)

        print(sparse_primes)
    else:
        sparse_primes = primes

    mod_prod_list = get_mod_prods(task_id=task_id, primes=sparse_primes,
                                  remainder_dict=None,
                                  num_tasks=num_tasks,debug=True)

    return(task_id, mod_prod_list)

if __name__ == '__main__':

    save_flag = True
    log_flag = True
    log_scaler = 1e0
    num_tasks = 20
    sparse = 1

    if save_flag:
        pool = mp.Pool()
        task_ids = range(num_tasks)
        results = dict(pool.starmap(func,
                                    zip(task_ids,
                                        repeat(sparse),
                                        repeat(num_tasks),
                                        repeat(log_flag),
                                        repeat(log_scaler))))

        save_pickle('./data/num_task{}_sparse{}_scale{}.json'.format(
            num_tasks, sparse, log_scaler), results, 'json')


        num_sparse_tasks = num_tasks*sparse
        primes, prod_prime = get_primes(num_primes=num_sparse_tasks,
                                        log_flag=log_flag,
                                        log_scaler=log_scaler)

        if sparse > 1:
            sparse_primes=[]
            for i, prm in enumerate(primes):
                if i % sparse != 0:
                    sparse_primes.append(prm)

            print(sparse_primes)
        else:
            sparse_primes = primes

        print('primes:{}'.format(sparse_primes))

        for task_id in range(num_tasks):
            diff_list = []
            prev = 0
            for mod in results[task_id]:
                diff= np.abs(prev - mod)
                prev = mod
                if diff > 0:
                    diff_list.append(diff)
                

            print("task_id:{}, min:{}, max:{}, mean:{}".format(
                task_id,
                np.min(diff_list),
                np.max(diff_list),
                np.mean(diff_list)
            ))

    else:
        prime_mod_dict = load_pickle('num_task{}_sparse{}_scale{}.json'.format(
            num_tasks, sparse, log_scaler), 'json')
        
    print('done')
