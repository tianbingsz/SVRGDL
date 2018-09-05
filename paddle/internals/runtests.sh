#!/bin/bash
# Copyright (c) 2016 Baidu, Inc. All Rights Reserved

set -x
cd "$(dirname "$0")"/../

utils/tests/test_StringUtils &&
gserver/tests/test_ProtoDataProvider &&
if [ -e gserver/tests/test_PyDataProvider ]
then
    PYTHONPATH=../python/:. gserver/tests/test_PyDataProvider
fi &&
#metric learning test
# if [ -e metric_learning/test/test_ExternalMachine ]; then
#     METRIC_CONFIG_ARG=extension_module_name=metric_learning.config_parser_metric
#     PYTHONPATH=. metric_learning/test/test_ExternalMachine --config_args=$METRIC_CONFIG_ARG &&
#     PYTHONPATH=. metric_learning/test/test_MetricDataProvider --config_args=$METRIC_CONFIG_ARG &&
#     sh ./.set_port.sh -n 3 -p data_server_port dserver/test/test_DataServer --config_args=$METRIC_CONFIG_ARG
#     #PYTHONPATH=. metric_learning/test/test_MetricGrad --config_args=$METRIC_CONFIG_ARG --external=1
# else
#     echo "metric learning module not found"
# fi &&
sh ./.set_port.sh -p port pserver/test/test_ProtoServer &&
sh ./.set_port.sh -n 4 -p port pserver/test/test_ParameterServer2 &&
#compile with PADDLE_ENABLE_RDMA=YES
#pserver/test/test_ProtoServer --rdma_tcp="rdma" --server_addr="10.92.95.103" &&
#pserver/test/test_ParameterServer2 --rdma_tcp="rdma" --pservers="10.92.95.103" --ports_num=1 --server_addr="10.92.95.103" --server_cpu=0 &&
utils/tests/test_CommandLineParser &&
parameter/tests/test_common &&
math/tests/test_matrix &&
math/tests/test_matrixCompare &&
math/tests/test_sparseMatrixCompare &&
utils/tests/test_Logging &&
math/tests/test_ExecViaCpu &&
math/tests/test_SIMDFunctions &&
math/tests/test_batchTranspose &&
math/tests/test_perturbation &&
math/tests/test_CpuGpuVector &&
utils/tests/test_Thread &&
internals/gserver/tests/test_FullContextLayer &&
PYTHONPATH=../python/:. python trainer/tests/config_parser_test.py &&
PYTHONPATH=../python/:. python internals/trainer/tests/config_parser_test.py &&
PYTHONPATH=../python/:. trainer/tests/test_Prediction &&
PYTHONPATH=../python/:. trainer/tests/test_Compare &&
PYTHONPATH=../python/:. trainer/tests/test_Trainer &&
PYTHONPATH=../python/:. sh ./.set_port.sh -p port trainer/tests/test_TrainerOnePass &&
PYTHONPATH=../python/:. gserver/tests/test_LayerGrad &&
PYTHONPATH=../python/:. ./trainer/tests/test_CompareTwoNets --config_file_a=trainer/tests/sample_trainer_config_qb_rnn.conf --config_file_b=trainer/tests/sample_trainer_config_rnn.conf --need_high_accuracy=1 &&
PYTHONPATH=../python/:. trainer/tests/test_CompareTwoOpts --config_file_a=trainer/tests/sample_trainer_config_opt_a.conf --config_file_b=trainer/tests/sample_trainer_config_opt_b.conf --num_passes=1 --need_high_accuracy=1 &&
PYTHONPATH=../python/:. sh ./.set_port.sh -n 6 -p port trainer/tests/test_CompareSparse &&
PYTHONPATH=../python/:. gserver/tests/test_Evaluator &&
PYTHONPATH=../python/:. gserver/tests/test_RecurrentGradientMachine &&
PYTHONPATH=../python/:. trainer/tests/test_recurrent_machine_generation &&
PYTHONPATH=../python/:. trainer/tests/test_PyDataProviderWrapper &&
PYTHONPATH=../python/:. internals/gserver/tests/test_SelectiveFCLayer &&
bash ./api/test/run_tests.sh
