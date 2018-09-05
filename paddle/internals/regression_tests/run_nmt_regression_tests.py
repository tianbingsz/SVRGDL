# Copyright (c) 2016 Baidu, Inc. All Rights Reserved

from paddle_version_checker import with_gpu, with_pydataprovider
from paddle_runner import PaddleRunner
from paddle_log_parser import PaddleLogParser, PaddleBeamSearchParser
from util import download_file_list, parse_json
import unittest
import json
import time

class test_NMT(unittest.TestCase):
    """
    Run regression test of training a sequence generation model that generates
    one target sequence from multiple source sequences.
    Check the final training and test error are within reasonable range.
    """
    @classmethod
    def setUpClass(cls):
        """
        Down nmt data
        """
        out_path = "./regression_tests/data/nmt/"
        file_list = ["train.lst", "test.lst", "src.dict"] + \
                    ["train_data", "test_data", "trg.dict"]
        baseurl = "http://m1-idl-gpu2-bak31.m1.baidu.com:8088/" \
                  "paddle_regression_test/nmt/"

        download_file_list(baseurl, file_list, out_path)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.conf_path = "./regression_tests/conf/nmt.conf"
        self.log_dir = "./regression_tests/logs/nmt/"
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
        self.runner.FLAGS_log_period(2)
        self.runner.ENV_logdir(self.log_dir + "gpu/")
        self.log_path = self.log_dir + "gpu/paddle_trainer.INFO"
        print self.runner
        p = self.runner.exec_()
        p.wait()
        time.sleep(10)
        parser = PaddleLogParser(self.log_path)
        self.assertLess(parser.last_train_error(), 0.45)
        self.assertLess(parser.last_test_error(), 0.55)


    @with_pydataprovider()
    def test_cpu(self):
        self.runner.FLAGS_use_gpu(False)
        self.runner.FLAGS_trainer_count(6)
        self.runner.FLAGS_log_period(2)
        self.runner.ENV_logdir(self.log_dir + "cpu/")
        self.log_path = self.log_dir + "cpu/paddle_trainer.INFO"
        print self.runner
        p = self.runner.exec_()
        p.wait()
        parser = PaddleLogParser(self.log_path)
        self.assertLess(parser.last_train_error(), 0.45)
        self.assertLess(parser.last_test_error(), 0.55)

class test_NMT_gen(unittest.TestCase):
    """
    Run regression test of neural machine translation (NMT) from French to English.
    Check the final generating English sequences are within reasonable range.
    """
    @classmethod
    def setUpClass(cls):
        """
        Down nmt data
        """
        out_path = "./regression_tests/data/nmt_gen/"
        file_list = ["gen.lst", "gen_data", "fr.dict", "en.dict"] + \
                    ["model.tar.gz", "cpu_result", "gpu_result"]
        baseurl = "http://m1-idl-gpu2-bak31.m1.baidu.com:8088/" \
                  "paddle_regression_test/nmt_gen/"

        download_file_list(baseurl, file_list, out_path)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.conf_path = "./regression_tests/conf/nmt_gen.conf"
        self.log_dir = "./regression_tests/logs/nmt_gen/"
        self.out_path = "./regression_tests/data/nmt_gen/"
        self.config_args = "generating=1,gen_trans_file=" + self.log_dir
        self.runner = PaddleRunner()
        self.runner.FLAGS_config(self.conf_path)
        self.runner.FLAGS_job("test")
        self.runner.FLAGS_save_dir("./regression_tests/data/nmt_gen/model")
        self.runner.FLAGS_trainer_count(1)
        with open("./regression_tests/default.json", 'r') as f:
            parse_json(self.runner, json.load(f))
        self.runner.FLAGS_test_pass(12)
        self.runner.FLAGS_num_passes(13)

    def tearDown(self):
        pass

    def compare_beam_search_result(self, result_path1, result_path2):
        """
        Compare the beam search results. The benchmark of the same results are:
        1. generating sequence must be the same.
        2. error of score should within 0.001.
        """
        score1, seqHash1 = PaddleBeamSearchParser(result_path1).find_info()
        score2, seqHash2 = PaddleBeamSearchParser(result_path2).find_info()
        self.assertEquals(seqHash1, seqHash2)
        self.assertEquals(len(score1), len(score2))
        for i in range(1, len(score1)):
            self.assertLess(abs(score1[i] - score2[i]), 0.001)
            
    @with_gpu()
    @with_pydataprovider()
    def test_gpu(self):
        self.runner.FLAGS_use_gpu(True)
        self.runner.FLAGS_config_args(self.config_args + "gpu/gen.txt")
        self.runner.ENV_logdir(self.log_dir + "gpu/")
        self.log_path = self.log_dir + "gpu/paddle_trainer.INFO"
        print self.runner
        p = self.runner.exec_()
        p.wait()
        time.sleep(10)
        self.compare_beam_search_result(self.log_dir + "gpu/gen.txt",
                                        self.out_path + "gpu_result")

    @with_pydataprovider()
    def test_cpu(self):
        self.runner.FLAGS_use_gpu(False)
        self.runner.FLAGS_config_args(self.config_args + "cpu/gen.txt")
        self.runner.ENV_logdir(self.log_dir + "cpu/")
        self.log_path = self.log_dir + "cpu/paddle_trainer.INFO"
        print self.runner
        p = self.runner.exec_()
        p.wait()
        self.compare_beam_search_result(self.log_dir + "cpu/gen.txt",
                                        self.out_path + "cpu_result")

if __name__ == '__main__':
    unittest.main()
