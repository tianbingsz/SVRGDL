#!/bin/sh
# Copyright (c) 2016 Baidu, Inc. All Rights Reserved

set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# regression test for NMT training and generation task
python regression_tests/run_nmt_regression_tests.py

# regression test for rnn chunking task
python regression_tests/run_rnn_regression_tests.py

# regression test for MNIST image classification
python regression_tests/run_mnist_regression_tests.py
