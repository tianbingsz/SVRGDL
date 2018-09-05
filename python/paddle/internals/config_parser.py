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

# internals' config_parser.py
from paddle.trainer.config_parser import *
from paddle.proto.DataConfig_pb2 import *
 
@config_func
def ImgData(
            files,
            channels,
            img_size,
            crop_size,
            meta_file,
            module='trainer.pyload',
            data_func='getNextBatch',
            meta_func='getMeta'):
    data_config = DataConfig()

    data_config.type = "image"
    data_config.files = files
    data_config.img_config.meta_file = meta_file
    data_config.feat_dim = \
            (img_size - 2 * crop_size) ** 2 * channels
    data_config.img_config.channels = channels
    data_config.img_config.img_size = img_size
    data_config.img_config.crop_size = crop_size
    data_config.img_config.module = module
    data_config.img_config.data_func = data_func
    data_config.img_config.meta_func = meta_func
    return data_config
  
@config_func
def SparseData(
        type,
        files,
        slot_dims,
        float_slot_dims=[],
        read_ltr_format_input=False,
        **xargs):

    data_config = DataBase(**xargs)
    data_config.type = type
    data_config.files = files
    data_config.slot_dims.extend(slot_dims)
    data_config.float_slot_dims.extend(float_slot_dims)
    data_config.read_ltr_format_input = read_ltr_format_input
    return data_config
  
@config_func
def LtrData(
        type,
        files,
        slot_dims,
        max_num_pair_per_query,
        pair_filter_phi_threshold,
        float_slot_dims=[],
        file_group_queue_capacity=None,
        load_file_count=None,
        load_thread_num=None,
        **xargs):

    data_config = SparseData(type, files, slot_dims, float_slot_dims, **xargs)
    data_config.ltr_conf.max_num_pair_per_query = max_num_pair_per_query
    data_config.ltr_conf.pair_filter_phi_threshold = pair_filter_phi_threshold
    if file_group_queue_capacity is not None:
        data_config.file_group_conf.queue_capacity = file_group_queue_capacity
    if load_file_count is not None:
        data_config.file_group_conf.load_file_count = load_file_count
    if load_thread_num is not None:
        data_config.file_group_conf.load_thread_num = load_thread_num
    return data_config

# Define the data used for training the neural network
@config_func
def SpeechData(
        files=None,
        feat_dim=None,
        slot_dims=None,
        context_len=None,
        buffer_capacity=None,
        train_sample_num=None,
        file_load_num=None,
        **xargs):
    data_config = DataBase(**xargs)
    data_config.type = 'speech'
    data_config.files = files
    data_config.feat_dim = feat_dim
    if context_len is not None:
        data_config.context_len = context_len
    data_config.buffer_capacity = buffer_capacity
    data_config.train_sample_num = train_sample_num
    data_config.file_load_num = file_load_num
    return data_config
  
@config_layer('fullcontext')
class FullContextLayer(LayerBase):
    def __init__(
            self,
            name,
            size,
            inputs,
            context_start,
            context_length,
            bias=True,
            **xargs):
        super(FullContextLayer, self).__init__(name, 'fullcontext', size, inputs=inputs, **xargs)
        config_assert(len(inputs) == 1, 'FullContextLayer must have 1 input')
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            self.config.full_context_config.input_dim=input_layer.size
            self.config.full_context_config.context_length=context_length
            begin_pad = max(0, -context_start)
            end_pad = max(0, context_start + context_length - 1)
            self.config.full_context_config.begin_pad = begin_pad
            self.config.full_context_config.end_pad = end_pad
            psize = (input_layer.size + begin_pad + end_pad) * self.config.size * context_length
            dims = [input_layer.size + begin_pad + end_pad, self.config.size * context_length]
            self.create_input_parameter(input_index, psize, dims)
        self.create_bias_parameter(bias, self.config.size)
        
'''
DataTrimLayer: A layer to trim input data.

Example:
    Layer(type = "data_trim", name = "data", size = 200)
    or DataTrimLayer(name = "data", size = 200)

Note:
    (1) size must less than or equal slot dim.
    (2) Only supports VECTOR_SPARSE_NON_VALUE and VECTOR_SPARSE_VALUE
        SlotType.
    (3) The trimed data only retain [0,size) dimensions and
        ignore [size, slot_dim) dimensions.
'''
@config_layer('data_trim')
class DataTrimLayer(LayerBase):
    layer_type = 'data_trim'
    def __init__(
            self,
            name,
            size,
            device=None):
        super(DataTrimLayer, self).__init__(name, self.layer_type, size, inputs=[], device=device)
        
@config_layer('perturbation_layer')
class PerturbationLayer(LayerBase):
    def __init__(self,
                 name,
                 inputs,
                 perturb_conf,
                 device=None):
        super(PerturbationLayer, self).__init__(name, 'perturbation_layer', 0, inputs=inputs, device=device)
        config_assert(len(self.inputs) == 1,
                     'PerturbationLayer must have one and only one input')
        self.config.perturb_conf = perturb_conf

@config_func
def PerturbationConfig(target_size,
                       sampling_rate=None,
                       padding_value=None,
                       scale=None,
                       rotation=None):
    perturb_conf = PerturbationConfig()
    perturb_conf.target_size = target_size
    perturb_conf.sampling_rate = sampling_rate
    perturb_conf.padding_value = padding_value
    perturb_conf.scale = scale
    perturb_conf.rotation = rotation
    return perturb_conf

def get_config_funcs(trainer_config):
    global g_config
    g_config = trainer_config
    return dict(ImgData=ImgData, SparseData=SparseData,
                LtrData=LtrData, SpeechData=SpeechData)
