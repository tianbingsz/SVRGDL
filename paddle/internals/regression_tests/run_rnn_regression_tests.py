# Copyright (c) 2016 Baidu, Inc. All Rights Reserved

from paddle_version_checker import with_gpu, with_pydataprovider
from paddle_runner import PaddleRunner
from paddle_log_parser import PaddleLogParser
from util import download_file_list, parse_json
import unittest
import json
import time
import os


class test_rnn_chunking(unittest.TestCase):
    """
    Run regression test of rnn chunking.
    Check the final training and testing error are within reasonable range.
    """

    @classmethod
    def setUpClass(cls):
        """
        Down chunking data
        """
        out_path = "./regression_tests/data/chunking/"
        file_list = ["train_files.txt", "test_files.txt"] + \
                    ["train_proto.bin", "test_proto.bin"]
        baseurl = "http://m1-idl-gpu2-bak31.m1.baidu.com:8088/" \
                  "paddle_regression_test/chunking/"

        download_file_list(baseurl, file_list, out_path)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.log_dir = "./regression_tests/logs/chunking_rnn/"
        self.runner = PaddleRunner()
        with open("./regression_tests/default.json", 'r') as f:
            parse_json(self.runner, json.load(f))

    def tearDown(self):
        pass

    @with_pydataprovider()
    def test_cpu_dense(self):
        """
        test chunking with cpu and use dense updater
        """
        self.runner.FLAGS_config("./regression_tests/conf/chunking_rnn.conf")
        self.runner.FLAGS_use_gpu(False)
        self.runner.FLAGS_trainer_count(12)
        self.runner.ENV_logdir(self.log_dir + "cpu/dense/")
        self.log_path = self.log_dir + "cpu/dense/paddle_trainer.INFO"
        print self.runner
        p = self.runner.exec_()
        p.wait()
        parser = PaddleLogParser(self.log_path)
        print 'last train error: ' + str(parser.last_train_error())
        print 'last test error: ' + str(parser.last_test_error())
        self.assertLess(parser.last_train_error(), 0.02)
        self.assertLess(parser.last_test_error(), 0.05)

    @with_pydataprovider()
    def test_cpu_sparse(self):
        """
        test chunking with cup and use sparse updater 
        """
        self.runner.FLAGS_config(
            "./regression_tests/conf/chunking_rnn.sparse.conf")
        self.runner.FLAGS_use_gpu(False)
        self.runner.FLAGS_trainer_count(12)
        self.runner.ENV_logdir(self.log_dir + "cpu/sparse/")
        self.log_path = self.log_dir + "cpu/sparse/paddle_trainer.INFO"
        print self.runner
        p = self.runner.exec_()
        p.wait()
        parser = PaddleLogParser(self.log_path)
        print 'last train error: ' + str(parser.last_train_error())
        print 'last test error: ' + str(parser.last_test_error())
        self.assertLess(parser.last_train_error(), 0.02)
        self.assertLess(parser.last_test_error(), 0.05)


if __name__ == '__main__':
    unittest.main()
