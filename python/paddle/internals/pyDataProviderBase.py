#!/usr/bin/python
# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


'''
Description: used for PyDataProvider, user SHOULD
derive from PyDataProviderBase and implement the
functions below. The result will be organised
in binary format.
'''

class PyDataProviderBase:
    def __init__(self, *file_list, **kwargs):
        '''
            Description: Init with a list of data file
            file_list is the name list of input files.
            kwargs["load_data_args"] is the value of 'load_data_args' which can be set in config.
        '''
        pass

    def shuffle(self):
        '''
            Description: set shuffle operation
        '''
        pass

    def reset(self):
        '''
          Description: execute some reset operation
          in each pass.
        '''
        pass

    def getHeader(self):
        '''
            Description: Get header info
            Return format:
                slot_num  // Total count of all slots
                use_sequence_flag // 1 if use sequence, 0 if not use sequence
                [(slot_type_1, slot_dim_1),  // A list of all slot type and dim
                 ...,
                 (slot_type_N, slot_dim_N)
                ]
            Note: The type of all header data all int32_t
        '''
        pass

    def getNextBatch(self, batch_size):
        '''
            Description: Get a batch of samples
            Return format:
                batch_size // specify the return value of getNextBatch
                // The flowing data was organise by slot, and the order of
                // slots was the same as in header. Different slot type has
                // diffent structure of data.
                // Now there are 4 kinds of data type.
                // 1 Dense type
                sample_num   // sample num in dense type slot
                value0 value1 ... valueN  // dense type slot values
                // 2 Sparse type with weight value
                sample_num
                index0 index1 ... indexN // the offset of sample in values.
                                        // Use CSR format to store sparse value.
                length value0 value1 ... valueN
                length weight0  weight1 ... weightN
                // 3 Sparse type without weight value type
                sample_num
                index0 index1 ... indexN
                length value0 value1 ... valueN
                // 4 Index type
                sample_num
                value0 value1 ... valueN

                // If you use sequence, you should specify  sequences offsets in
                // each slot.
                slot1_sequence_num sequence_pos0  ... sequence_posN
                ...
                slotN_sequence_num sequence_pos0  ... sequence_posN
        '''
        pass


