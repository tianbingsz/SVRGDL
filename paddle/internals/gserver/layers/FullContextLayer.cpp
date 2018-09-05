/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include "FullContextLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "paddle/math/SIMDFunctions.h"

namespace paddle {

REGISTER_LAYER(fullcontext, FullContextLayer);

bool FullContextLayer::init(const LayerMap& layerMap,
                            const ParameterMap& parameterMap) {
  CHECK(!useGpu_) << "FullContextLayer does not support gpu";
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  /* initialize the weightList */
  CHECK(inputLayers_.size() == parameters_.size());
  // FullContextLayer should have exactly 1 input.
  CHECK_EQ(1U, inputLayers_.size());

  // size of the parameters
  size_t height =
      (inputLayers_[0]->getSize() + config_.full_context_config().begin_pad() +
       config_.full_context_config().end_pad());
  size_t width = getSize() * config_.full_context_config().context_length();

  // create a new weight
  CHECK_EQ(parameters_[0]->getSize(), width * height);
  Weight* w = new Weight(height, width, parameters_[0]);

  // append the new weight to the list
  weights_.emplace_back(w);

  /* initialize biases_ */
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
  }

  // init member MatrixPtrs
  allResultMatrix_ = Matrix::create(1, 1);
  resultMaxMatrix_ = Matrix::create(nullptr, 1, getSize(),
                                    /*trans=*/false,
                                    /*useGpu*/ false);

  // We don't need sequenceStartPositions because each sample of output_ is
  // for the cost of one sequence.
  setNeedSequenceInfo(false);

  return true;
}

void FullContextLayer::prefetch() {
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    auto* sparseParam =
        dynamic_cast<SparsePrefetchRowCpuMatrix*>(weights_[i]->getW().get());
    if (sparseParam) {
      MatrixPtr input = getInputValue(i);
      sparseParam->addRows(input);
    }
  }
}

void FullContextLayer::forward(PassType passType) {
  Layer::forward(passType);

  size_t dim = getSize();

  const Argument& input = getInput(0);
  int64_t newBatchSize = input.getNumSequences();
  auto startPositions =
      input.sequenceStartPositions->getVector(false);
  const int* starts = startPositions->getData();
  size_t numSequences = startPositions->getSize() - 1;
  // check
  CHECK_EQ(numSequences, (size_t)newBatchSize);
  CHECK_EQ(starts[numSequences], input.getBatchSize());
  CHECK(input.ids);

  // reset output: resize to "num of sequences", not "batch size".
  resetOutput(newBatchSize, dim);

  real* biastable = nullptr;
  if (biases_) {
    biastable = biases_->getW()->getData();  // copy bias from here
  }

  MatrixPtr mat = weights_[0]->getW();
  real* table = mat->getData();

  MatrixPtr outV = getOutputValue();
  real* outptr = outV->getData();
  int* ids = input.ids->getData();

  for (size_t seqid = 0; seqid < numSequences; seqid++) {
    // for each input sequence
    int length = starts[seqid + 1] - starts[seqid];
    allResultMatrix_->resize(length, dim);
    allResultMatrix_->zeroMem();
    real* allResultData = allResultMatrix_->getData();

    for (int windowid = 0; windowid < length; windowid++) {
      for (int pos = 0; pos < config_.full_context_config().context_length();
           ++pos) {
        // what is current id?
        int index = windowid + pos - config_.full_context_config().begin_pad();
        int id = -1;
        if (index < 0) {
          id = config_.full_context_config().input_dim() + index +
               config_.full_context_config().begin_pad();
        } else if (index >= length) {
          id = config_.full_context_config().input_dim() + index +
               config_.full_context_config().begin_pad() - length;
        } else {
          id = ids[index + starts[seqid]];
        }
        if (id == -1) {
          continue;
        }
        // get real values according to id
        real* src = table +
                    dim * config_.full_context_config().context_length() * id +
                    pos * dim;
        // get summation
        simd::addTo(allResultData + windowid * dim, src, dim);
      }
    }

    // calc max using colMax
    resultMaxMatrix_->setData(outptr + seqid * dim);
    allResultMatrix_->colMax(*resultMaxMatrix_);

    // add bias
    if (biases_) {
      simd::addTo(outptr + seqid * dim, biastable, dim);
    }
  }
  // activate
  forwardActivation();
}

void FullContextLayer::backward(const UpdateCallback& callback) {
  LOG(FATAL) << "not implemented";
}

}  // namespace paddle
