# Copyright (c) 2016 Baidu, Inc. All Rights Reserved

from paddle_version_checker import with_gpu, with_pydataprovider
from paddle_runner import PaddleRunner
from paddle_log_parser import PaddleLogParser
from util import download_file_list, parse_json
import unittest
import json
import time

class test_MNIST(unittest.TestCase):
    """
    Run regression test of MNIST image classification.
    Check the final training and testing error are within reasonable range.
    """
    @classmethod
    def setUpClass(cls):
        """
        Down mnist data
        """
        out_path = "./regression_tests/data/mnist/"
        file_list = ["train.lst", "test.lst", "mnist.meta"] + \
                    ["train_batch_%03d" % i for i in range(12)] + \
                    ["test_batch_%03d" % i for i in range(2)]
        baseurl = "http://m1-idl-gpu2-bak31.m1.baidu.com:8088/" \
                  "paddle_regression_test/mnist/"

        download_file_list(baseurl, file_list, out_path)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.conf_path = "./regression_tests/conf/mnist.conf"
        self.log_dir = "./regression_tests/logs/mnist/"
        self.runner = PaddleRunner()
        self.runner.FLAGS_config(self.conf_path)
        with open("./regression_tests/default.json", 'r') as f:
            parse_json(self.runner, json.load(f))
        
    def tearDown(self):
        pass
   
    @with_gpu()
    @with_pydataprovider()
    def test_gpu(self):
        self.runner.FLAGS_use_gpu(True)
        self.runner.FLAGS_trainer_count(2)
        self.runner.ENV_logdir(self.log_dir + "gpu/")
        self.log_path = self.log_dir + "gpu/paddle_trainer.INFO"
        print self.runner
        p = self.runner.exec_()
        p.wait()
        time.sleep(10)
        parser = PaddleLogParser(self.log_path)
        self.assertLess(parser.last_train_error(), 0.02)
        self.assertLess(parser.last_test_error(), 0.03)
 

    @with_pydataprovider()
    def test_cpu(self):
        self.runner.FLAGS_use_gpu(False)
        self.runner.FLAGS_trainer_count(2)
        self.runner.ENV_logdir(self.log_dir + "cpu/")
        self.log_path = self.log_dir + "cpu/paddle_trainer.INFO"
        print self.runner
        p = self.runner.exec_()
        p.wait()
        parser = PaddleLogParser(self.log_path)
        self.assertLess(parser.last_train_error(), 0.02)
        self.assertLess(parser.last_test_error(), 0.03)
 
if __name__ == '__main__':
    unittest.main()
