/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include "paddle/utils/Stat.h"
#include "paddle/utils/Util.h"
#include "paddle/parameter/Argument.h"
#include <algorithm>
#include <stdlib.h>
#include "paddle/utils/Logging.h"

#include "comlog/comlog.h"

#include "paddle/gserver/dataproviders/DataProvider.h"
#include "FcrCvmDataProvider.h"
#include "cvm_reader_interface.h"

namespace paddle {

using namespace std;

REGISTER_DATA_PROVIDER(fcr_cvm, FcrCvmDataProvider);

FcrCvmDataProvider::FcrCvmDataProvider(const DataConfig& config, bool useGpu)
    : DataProvider(config, useGpu) {
  /* initialize the size of a sample, and the buffer */
  sampleDim_ = config_.fcr_cvm_conf().feat_dim();
  if (config_.fcr_cvm_conf().has_with_lineid()) {
    withLineid_ = config_.fcr_cvm_conf().with_lineid();
  } else {
    withLineid_ = false;
  }
  bufferCapacity_ = config_.buffer_capacity();
  streamNum_ = config_.fcr_cvm_conf().stream_def_size();
  training_dataPath_ = config_.fcr_cvm_conf().training_data_path();
  training_dataCompressOpt_ =
      config_.fcr_cvm_conf().training_data_compress_opt();
  if (config_.fcr_cvm_conf().has_is_training_data_need_shuffle()) {
    isTrainingDataNeedShuffle_ =
        config_.fcr_cvm_conf().is_training_data_need_shuffle();
  } else {
    isTrainingDataNeedShuffle_ = true;
  }
  downloadRetryNum_ = config_.fcr_cvm_conf().download_retry_num();
  downloadRetrySleepUs_ = config_.fcr_cvm_conf().download_retry_sleep_us();
  downloadTraceInterval_ = config_.fcr_cvm_conf().download_trace_interval();
  downloadThreadNum_ = config_.fcr_cvm_conf().download_thread_num();
  downloadOnceCount_ = config_.fcr_cvm_conf().download_once_count();
  comlogConfPath_ = config_.fcr_cvm_conf().comlog_conf_path();
  comlogConfFile_ = config_.fcr_cvm_conf().comlog_conf_file();
  mean_var_filename_ = config_.fcr_cvm_conf().mean_var_filename();

  string missing_mode_string = config_.fcr_cvm_conf().missing_value_stra();
  missing_mode_ = cvm::Reader::extend_arg_t::MISSING_AS_UNKNOWN;
  if (missing_mode_string == "MISSING_AS_MEAN") {
    missing_mode_ = cvm::Reader::extend_arg_t::MISSING_AS_MEAN;
  } else if (missing_mode_string == "MISSING_AS_ZERO") {
    missing_mode_ = cvm::Reader::extend_arg_t::MISSING_AS_ZERO;
  } else {
    LOG(FATAL) << "unsupport missing_value_stra [" << missing_mode_string
               << "]";
    exit(1);
  }

  CHECK(downloadThreadNum_ <= UINT32_MAX);

  if (com_loadlog(comlogConfPath_.c_str(), comlogConfFile_.c_str()) != 0) {
    LOG(FATAL) << "init comlog failed, conf_path[" << comlogConfPath_
               << "], conf_file[" << comlogConfFile_ << "].\n";
    exit(1);
  }

  streamDef_.clear();
  streamDef_.resize(streamNum_);
  for (uint64_t i = 0; i < streamNum_; i++) {
    streamDef_[i].clear();
    const paddle::FcrCvmSlotList& slot_list =
        config_.fcr_cvm_conf().stream_def(i);
    uint64_t slotSize = slot_list.slot_size();
    for (uint64_t j = 0; j < slotSize; j++) {
      CHECK(slot_list.slot(j) < sampleDim_);
      streamDef_[i].push_back(slot_list.slot(j));
    }
  }

  show_ = new (std::nothrow) cvm::SHOW_CLK_TYPE[MAX_MINI_BATCH];
  clk_ = new (std::nothrow) cvm::SHOW_CLK_TYPE[MAX_MINI_BATCH];
  CHECK(NULL != show_ && NULL != clk_) << "new mem for show/clk failed";

  fea_ = new (std::nothrow) cvm::FEATURE_TYPE*[streamNum_];
  CHECK(NULL != fea_) << "new mem for fea failed";
  for (uint64_t i = 0; i < streamNum_; ++i) {
    fea_[i] = new (std::nothrow)
        cvm::FEATURE_TYPE[streamDef_[i].size() * MAX_MINI_BATCH];
    CHECK(NULL != fea_[i]) << "new mem for fea[" << i << "] failed";
  }

  feaTransMean_ = new (std::nothrow) cvm::FEATURE_TYPE[sampleDim_];
  feaTransStdvar_ = new (std::nothrow) cvm::FEATURE_TYPE[sampleDim_];
  CHECK(NULL != feaTransMean_ && NULL != feaTransStdvar_)
      << "new mem for feaTrans failed";
  if (0 != load_mean_var(mean_var_filename_.c_str(), sampleDim_)) {
    LOG(FATAL) << "load mean_var failed.\n";
    exit(1);
  }

  // init cvm-reader
  cvm::FileReader* list_file_reader = NULL;
  if (training_dataCompressOpt_ == "none") {
    list_file_reader = new (std::nothrow) cvm::BinaryFileReader();
  } else if (training_dataCompressOpt_ == "gz") {
    list_file_reader = new (std::nothrow) cvm::GzFileReader();
  } else {
    LOG(FATAL) << "unsupported training_dataCompressOpt ["
               << training_dataCompressOpt_ << "].\n";
    exit(1);
  }
  if (NULL == list_file_reader) {
    LOG(FATAL) << "new mem for list_file_reader failed\n";
    exit(1);
  }
  list_file_reader->set_retry_max_times(downloadRetryNum_);
  list_file_reader->set_retry_sleep_us(downloadRetrySleepUs_);
  list_file_reader->set_trace_interval(downloadTraceInterval_);
  // get filelist
  std::vector<std::string> overall_file_list;
  if (cvm::Reader::muti_list_files(training_dataPath_.c_str(),
                                   overall_file_list, list_file_reader) != 0) {
    LOG(FATAL) << "get training_dataPath file list fail [" << training_dataPath_
               << "]\n";
    exit(1);
  }
  // file list split for multi-machine training
  int cur_rank = -1;
  char* cur_rank_str = getenv("OMPI_COMM_WORLD_RANK");
  if (NULL != cur_rank_str && cur_rank_str[0] != '\0') {
    cur_rank = atoi(cur_rank_str);
  }
  int tot_rank = 0;
  char* tot_rank_str = getenv("OMPI_COMM_WORLD_SIZE");
  if (NULL != tot_rank_str && tot_rank_str[0] != '\0') {
    tot_rank = atoi(tot_rank_str);
  }
  fileList_.clear();
  if (tot_rank > 0) {
    CHECK(cur_rank >= 0 && cur_rank < tot_rank)
        << "invalid OMPI_COMM_WORLD_RANK [" << cur_rank_str << "] and "
        << "OMPI_COMM_WORLD_SIZE [" << tot_rank_str << "]\n";
    LOG(INFO) << "Multi-machine mode, split training filelist, [ID \% "
              << tot_rank << " = " << cur_rank << "] for this rank\n";
    size_t file_num = overall_file_list.size();
    for (size_t i = cur_rank; i < file_num; i += tot_rank) {
      fileList_.push_back(overall_file_list[i]);
    }
  } else {
    fileList_ = overall_file_list;
  }

  delete list_file_reader;
  list_file_reader = NULL;

  downloadFileReader_ = NULL;
  fileQueue_ = NULL;
  reader_ = NULL;
}

FcrCvmDataProvider::~FcrCvmDataProvider() {
  destroyCvmReader();
  delete[] show_;
  show_ = NULL;
  delete[] clk_;
  clk_ = NULL;
  if (NULL != fea_) {
    for (uint64_t i = 0; i < streamNum_; ++i) {
      delete[] fea_[i];
      fea_[i] = NULL;
    }
  }
  delete[] fea_;
  fea_ = NULL;
  delete[] feaTransMean_;
  feaTransMean_ = NULL;
  delete[] feaTransStdvar_;
  feaTransStdvar_ = NULL;

  if (com_closelog(6000) != 0) {
    LOG(FATAL) << "close comlog failed\n";
    exit(1);
  }
}

void FcrCvmDataProvider::shuffle() {
  // do nothing
  return;
}

int64_t FcrCvmDataProvider::getNextBatchInternal(int64_t size,
                                                 DataBatch* batch) {
  CHECK(batch != NULL);
  CHECK(size > 0 && size <= (int64_t)MAX_MINI_BATCH);
  CHECK(reader_ != NULL);
  batch->clear();

  std::lock_guard<RWLock> guard(lock_);

  // read data
  uint64_t actual_size = 0;
  {
    REGISTER_TIMER("fcrCvmReadData");
    if (reader_->read(show_, clk_, fea_, NULL, size, actual_size) != 0) {
      LOG(FATAL) << "reader_->read failed\n";
      return 0;
    }
  }
  if (actual_size == 0) {
    LOG(WARNING) << "reader_->read, read nothing\n";
    return 0;
  }

  /* sample feature */
  if ((*dataBatch_).empty()) {
    (*dataBatch_).resize(streamNum_);
  }
  for (uint64_t i = 0; i < streamNum_; i++) {
    CpuMatrix cpuData(fea_[i], actual_size, streamDef_[i].size(), false);
    MatrixPtr& data = (*dataBatch_)[i];
    Matrix::resizeOrCreate(data, actual_size, streamDef_[i].size(), false,
                           useGpu_);
    data->copyFrom(cpuData);
    batch->appendData(data);
  }

  /* samples label */
  CpuIVector cpuLabel(actual_size, clk_);
  IVectorPtr& label = *labelBatch_;
  if (NULL == label) {
    label = IVector::create(actual_size, useGpu_);
  } else {
    label->resize(actual_size);
  }
  label->copyFrom(cpuLabel);
  batch->appendLabel(label);

  batch->setSize(actual_size);

  return actual_size;
}

void FcrCvmDataProvider::reset() {
  if (destroyCvmReader() != 0) {
    LOG(FATAL) << "destroy cvm_reader object failed while reset.";
    exit(1);
  }
  if (initCvmReader() != 0) {
    LOG(FATAL) << "init cvm_reader object failed while reset.";
    exit(1);
  }
  DataProvider::reset();
}

int FcrCvmDataProvider::initCvmReader() {
  CHECK(fileQueue_ == NULL);
  CHECK(downloadFileReader_ == NULL);
  CHECK(reader_ == NULL);

  fileQueue_ = new (std::nothrow) cvm::PreparedFileQueue;
  CHECK(NULL != fileQueue_) << "new mem for fileQueue failed";
  if (training_dataCompressOpt_ == "none") {
    downloadFileReader_ = new (std::nothrow) cvm::BinaryFileReader();
  } else if (training_dataCompressOpt_ == "gz") {
    downloadFileReader_ = new (std::nothrow) cvm::GzFileReader();
  } else {
    LOG(FATAL) << "unsupported training_dataCompressOpt ["
               << training_dataCompressOpt_ << "].\n";
    return -1;
  }
  CHECK(NULL != downloadFileReader_) << "new mem for downloadFileReader failed";
  downloadFileReader_->set_retry_max_times(downloadRetryNum_);
  downloadFileReader_->set_retry_sleep_us(downloadRetrySleepUs_);
  downloadFileReader_->set_trace_interval(downloadTraceInterval_);

  // shuffle file list
  if (isTrainingDataNeedShuffle_) {
    LOG(INFO) << "Shuffling training data file list ...\n";
    std::srand(unsigned(std::time(0)));
    std::random_shuffle(fileList_.begin(), fileList_.end());
  }
  LOG(INFO) << "Training data file list:\n";
  for (uint64_t i = 0; i < std::min(fileList_.size(), (size_t)5); i++) {
    LOG(INFO) << "  " << fileList_[i].c_str() << "\n";
  }
  if (fileList_.size() > (size_t)5) {
    LOG(INFO) << "  "
              << "......\n";
  }
  LOG(INFO) << "Training data file list count: " << fileList_.size() << "\n";

  if (fileQueue_->init(fileList_) != 0) {
    LOG(FATAL) << "init file_queue failed\n";
    return -1;
  }

  reader_ = new (std::nothrow) cvm::ReaderEx();
  CHECK(reader_ != NULL);
  if (reader_->init(
          downloadThreadNum_,
          cvm::Reader::normal_arg_t(
              sampleDim_, (withLineid_ ? LINEID_SIZE_IN_BYTES : 0),
              bufferCapacity_ / downloadThreadNum_, downloadOnceCount_),
          fileQueue_, downloadFileReader_,
          cvm::Reader::extend_arg_t(missing_mode_, feaTransMean_,
                                    feaTransStdvar_, &streamDef_),
          downloadTraceInterval_,
          cvm::ReaderEx::buffer_arg_t(static_cast<uint32_t>(downloadThreadNum_),
                                      MAX_MINI_BATCH)) != 0) {
    LOG(FATAL) << "reader_->init failed\n";
    return -1;
  }
  downloadFileReader_->trace_reset();

  return 0;
}

int FcrCvmDataProvider::destroyCvmReader() {
  if (reader_ != NULL) {
    if (reader_->destroy() != 0) {
      LOG(FATAL) << "reader_->destroy failed\n";
      return -1;
    }
    delete reader_;
    reader_ = NULL;
  }
  if (NULL != fileQueue_) {
    if (fileQueue_->destroy() != 0) {
      LOG(FATAL) << "fileQueue_.destroy failed\n";
      return -1;
    }
    delete fileQueue_;
    fileQueue_ = NULL;
  }

  delete downloadFileReader_;
  downloadFileReader_ = NULL;

  return 0;
}

int64_t FcrCvmDataProvider::getSize() {
  LOG(FATAL) << "Currently, not implemented";
  return 0;
}

int FcrCvmDataProvider::load_mean_var(const char* mean_var_file,
                                      const uint32_t slot_num) {
  if (NULL == mean_var_file || slot_num <= 0) {
    LOG(FATAL) << "invalid input parameter.\n";
    return -1;
  }

  if (NULL == feaTransMean_ || NULL == feaTransStdvar_) {
    LOG(FATAL)
        << "feaTransMean_ & feaTransStdvar_ have not been initialized yet.\n";
    return -1;
  }

  FILE* fp = fopen(mean_var_file, "r");
  if (NULL == fp) {
    LOG(FATAL) << "open mean_var_file[" << mean_var_file << "] failed.\n";
    return -1;
  }

  char line_buff[MAX_LINE_BUFF_SIZE];
  uint64_t cur_slot = 0;
  while (!feof(fp) && cur_slot < slot_num) {
    if (NULL == fgets(line_buff, MAX_LINE_BUFF_SIZE, fp)) {
      continue;
    }
    // is blank line
    if (0 == line_buff[0]) {
      continue;
    }
    char* delimiter_ptr = strchr(line_buff, MEAN_VAR_FILE_DELIMITER);
    if (NULL == delimiter_ptr) {
      LOG(FATAL) << "bad format of mean_var_file, content=[" << line_buff
                 << "].\n";
      break;
    }
    *delimiter_ptr = '\0';
    feaTransMean_[cur_slot] = static_cast<cvm::FEATURE_TYPE>(atof(line_buff));
    delimiter_ptr++;
    feaTransStdvar_[cur_slot] =
        static_cast<cvm::FEATURE_TYPE>(atof(delimiter_ptr));
    cur_slot++;
  }

  if (cur_slot != slot_num) {
    LOG(FATAL) << "error occured while reading mean_var_file or line number ["
               << cur_slot << "] mismatch with slot_num [" << slot_num
               << "].\n";
    fclose(fp);
    return -1;
  }

  fclose(fp);

  return 0;
}

}  // namespace paddle
