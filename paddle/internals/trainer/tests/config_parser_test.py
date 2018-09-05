# Copyright (c) 2016 Baidu, Inc. All Rights Reserved

from paddle.trainer.config_parser import parse_config_and_serialize

if __name__ == '__main__':
    parse_config_and_serialize(
       'internals/trainer/tests/sample_trainer_config_image.conf',
       'trainer_id=1,local=1,extension_module_name=paddle.internals.config_parser')
