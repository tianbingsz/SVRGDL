/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>

#include "paddle/utils/Locks.h"
#include "paddle/utils/ThreadLocal.h"
#include "paddle/utils/TypeDefs.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/Vector.h"
#include "DataConfig.pb.h"
#include "paddle/gserver/dataproviders/DataProvider.h"

#include "cvm_reader_interface.h"

namespace paddle {

class FcrCvmDataProvider : public DataProvider {
protected:
  uint64_t sampleDim_;  // overall feature dimension
  bool withLineid_;  // whether input data contains lineid info (extra 8 bytes
                     // at last)
  uint64_t bufferCapacity_;        // the number of samples of downloader buffer
  uint64_t streamNum_;             // number of stream
  std::string training_dataPath_;  // feature for training data path
  std::string
      training_dataCompressOpt_;    // compress method of training data files
  bool isTrainingDataNeedShuffle_;  // whether need to shuffle filelist of
                                    // training data every epoch
  uint64_t downloadRetryNum_;       // retry number of data downloader
  uint64_t downloadRetrySleepUs_;   // sleep time between retries, in us
  uint64_t downloadTraceInterval_;  // speed monitor interval of downloader, in
                                    // second
  uint64_t downloadThreadNum_;      // thread number of asyn downloader
  uint64_t downloadOnceCount_;      // # of instance of one download block
  std::string comlogConfPath_;      // path of comlog conf file
  std::string comlogConfFile_;      // filename of comlog conf file
  std::string mean_var_filename_;   // filename of feature normalization
  cvm::Reader::extend_arg_t::MISSING_MODE
      missing_mode_;  // missing value strategy of feature value

  ThreadLocal<std::vector<MatrixPtr>> dataBatch_;
  ThreadLocal<IVectorPtr> labelBatch_;

  std::vector<std::vector<uint64_t>> streamDef_;

  cvm::ReaderEx* reader_;
  cvm::FileReader* downloadFileReader_;
  cvm::PreparedFileQueue* fileQueue_;

  cvm::SHOW_CLK_TYPE* show_;
  cvm::SHOW_CLK_TYPE* clk_;
  cvm::FEATURE_TYPE** fea_;

  cvm::FEATURE_TYPE* feaTransMean_;
  cvm::FEATURE_TYPE* feaTransStdvar_;

  std::vector<std::string> fileList_;

  RWLock lock_;

public:
  FcrCvmDataProvider(const DataConfig& config, bool useGpu);
  ~FcrCvmDataProvider();
  static const uint64_t MAX_MINI_BATCH = 1024000;
  static const uint64_t MAX_LINE_BUFF_SIZE = 10240;
  static const char MEAN_VAR_FILE_DELIMITER = ' ';
  static const uint64_t LINEID_SIZE_IN_BYTES = sizeof(uint64_t);

  void shuffle();

  virtual void reset();

  virtual int64_t getSize();

  virtual int64_t getNextBatchInternal(int64_t size, DataBatch* batch);

private:
  int load_mean_var(const char* mean_var_file, const uint32_t slot_num);
  int initCvmReader();
  int destroyCvmReader();
};

}  // namespace paddle
