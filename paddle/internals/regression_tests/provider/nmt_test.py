# Copyright (c) 2016 Baidu, Inc. All Rights Reserved

import os
import sys

from paddle.trainer.PyDataProviderWrapper import *

UNK_IDX = 2
START = "<s>"
END = "<e>"

@init_hook_wrapper
def hook(obj, src_dict, trg_dict, file_list, **kwargs):
    # job_mode = 1: training mode
    # job_mode = 0: generating mode
    obj.job_mode = trg_dict is not None
    obj.src_dict = src_dict
    obj.logger.info("src dict len : %d" % (len(obj.src_dict)))
    obj.sample_count = 0

    if obj.job_mode:
        obj.trg_dict = trg_dict 
        obj.slots = [IndexSlot(len(obj.src_dict)), 
                     IndexSlot(len(obj.trg_dict)),
                     IndexSlot(len(obj.trg_dict))]
        obj.logger.info("trg dict len : %d" % (len(obj.trg_dict)))
    else:
        obj.slots = [IndexSlot(len(obj.src_dict)),
                     IndexSlot(len(open(file_list[0], "r").readlines()))]

def _get_ids(s, dictionary):
    words = s.strip().split()
    return [dictionary[START]] + \
           [dictionary.get(w,UNK_IDX) for w in words] +\
           [dictionary[END]]

@provider(use_seq=True, init_hook=hook, pool_size=PoolSize(5000))
# PoolSize(5000) means read at most 5000 samples to memory
def process(obj, file_name):
    with open(file_name, 'r') as fdata:
        for line_count, line in enumerate(fdata):
            line_split = line.strip().split('\t')
            if obj.job_mode and len(line_split) != 2:
                continue
            src_seq = line_split[0] # one source sequence
            trg_seq = line_split[1] # one target sequence
            src_ids = _get_ids(src_seq, obj.src_dict)

            if obj.job_mode:
                trg_words = trg_seq.split()
                trg_ids = [obj.trg_dict.get(w, UNK_IDX) for w in trg_words]
                trg_ids_next = trg_ids + [obj.trg_dict[END]]
                trg_ids = [obj.trg_dict[START]] + trg_ids
                yield src_ids, trg_ids, trg_ids_next
            else:
                yield src_ids, [line_count]

@provider(use_seq=True, init_hook=hook)
def process2(obj, file_name):
    with open(file_name, "r") as fdata:
        for line in fdata:
            line_split = line.strip().split("\t")
            if obj.job_mode and len(line_split) != 2:
                continue
            obj.sample_count += 1
            src_seq = line_split[0] # multiple source sequence
            trg_seq = line_split[1] # one target sequence
            src_ids_list = [_get_ids(seq, obj.src_dict) \
                for seq in src_seq.split("\x01")]

            if obj.job_mode:
                trg_words = trg_seq.split()
                trg_ids = [obj.trg_dict.get(w, UNK_IDX) for w in trg_words]
                trg_ids_next = trg_ids + [obj.trg_dict[END]]
                trg_ids = [obj.trg_dict[START]] + trg_ids
                yield src_ids_list, trg_ids, trg_ids_next
            else:
                yield src_ids_list, [obj.sample_count]
