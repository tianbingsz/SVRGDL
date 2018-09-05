/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include "paddle/utils/Util.h"
#include <unistd.h>
#include "paddle/utils/Logging.h"
#include <algorithm>

#include "CepstrumExtract.h"
#include "paddle/gserver/dataproviders/DataProvider.h"

namespace paddle {


class SpeechDataProvider : public SimpleDataProviderBase {
protected:
  CepstrumExtractPtr cepstrum_;

public:
  SpeechDataProvider(const DataConfig& config, bool useGpu);
  ~SpeechDataProvider();
  virtual void reset();

protected:
  virtual int64_t fillBufferImp(real* data, int* label, int* info,
                                int64_t size);
};

REGISTER_DATA_PROVIDER(speech, SpeechDataProvider);

SpeechDataProvider::SpeechDataProvider(const DataConfig& config, bool useGpu)
    : SimpleDataProviderBase(config, useGpu, /* withInfo= */ false) {
  CHECK_EQ(config.type(), "speech");

  cepstrum_ = CepstrumExtractPtr(new CepstrumExtract(
      config_.feat_dim(), config_.context_len(), config_.train_sample_num()));
  cepstrum_->setDataFileList(config_.files().c_str(), config_.file_load_num());
}

SpeechDataProvider::~SpeechDataProvider() {}

int64_t SpeechDataProvider::fillBufferImp(real* data, int* label, int* info,
                                          int64_t size) {
  (void)info;
  int64_t n = 0;

  while (n < size) {
    /* if the data field is empty, read the data from next file-block */
    if (cepstrum_->eof()) {
      if (!cepstrum_->readData()) {
        break;
      }
      cepstrum_->randomize();
    }
    cepstrum_->write2Buf(data, label);
    data += sampleDim_;
    label++;
    n++;
  }

  return n;
}

void SpeechDataProvider::reset() {
  this->cepstrum_->reset();
  SimpleDataProviderBase::reset();
}

}  // namespace paddle

