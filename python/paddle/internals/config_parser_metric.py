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

from paddle.trainer.config_parser import *
from paddle.proto.DataConfig_pb2 import *

@config_func
def MetricCon(
        buffer_capacity=None,
        train_batch_num=None,
        sample_num_per_class=None,
        template_num=None,
        batch_size=None):
    metric_conf = MetricDataConf()
    metric_conf.train_batch_num = train_batch_num
    metric_conf.buffer_capacity = buffer_capacity
    metric_conf.sample_num_per_class = sample_num_per_class
    metric_conf.template_num = template_num
    metric_conf.batch_size = batch_size
    return metric_conf

# Define a function
@config_func
def parse_metric_conf(metric_para, metric_conf):
    metric_conf.gamma = metric_para.gamma
    metric_conf.threshold = metric_para.threshold
    metric_conf.max_pos_ref_num = metric_para.max_pos_ref_num

@config_func
def MetricData(data_config, metric_conf):
    config = DataConfig()
    config.CopyFrom(data_config)
    config.metric_conf.CopyFrom(metric_conf)
    config.underline_type = data_config.type
    config.type = "metric"
    return config


@config_func
def ExternalConfig(layers, inputLayerNames, outputLayerNames):
    for name in layers:
        g_config.model_config.external_config.layer_names.append(name)
    for name in inputLayerNames:
        g_config.model_config.external_config.input_layer_names.append(name)
    for name in outputLayerNames:
        g_config.model_config.external_config.output_layer_names.append(name)

@config_class
class Metric(Cfg):
    def __init__(self,
                 gamma,
                 threshold,
                 max_pos_ref_num):
        self.add_keys(locals())

@config_layer('metric_learning_cost')
class MetricCostLayer(LayerBase):
    def __init__(self,
                 name,
                 inputs,
                 metric_conf,
                 device=None):
        super(MetricCostLayer, self).__init__(name, 'metric_learning_cost',
                                              0, inputs=inputs,
                                              device=device)
        parse_metric_conf(metric_conf, self.config.metric_conf)
        self.set_layer_size(self.get_input_layer(0).size)
        config_assert(len(self.inputs) == 2,
                         'MetricCostLayer must have two input')

def get_config_funcs(trainer_config):
    global g_config
    g_config = trainer_config
    return dict(MetricData=MetricData)
