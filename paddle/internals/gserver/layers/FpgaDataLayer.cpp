/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

#ifdef PADDLE_USE_FPGA
#include "FpgaLayer.h"
#include "paddle/gserver/layers/DataLayer.h"

namespace paddle {

class FpgaDataLayer : public DataLayer, public IFpgaLayer {
public:
  explicit FpgaDataLayer(const LayerConfig& config): DataLayer(config) {}

  bool fpgaForward(PassType passType) {
    if (this->fpgaValue) {
      fpgaValue = std::make_shared<fpga::FpgaDsp24bitDataMatrix>(0);
    }
    size_t height = data_.value->getHeight();
    size_t width = data_.value->getWidth();
    if (fpgaValue->resize(height, width) != 0) {
      return false;
    }

    if (fpgaValue->copy_from_in_row((float *)(data_.value->getData()),
                                          height, width) != 0) {
      return false;
    }
    forward(passType);
    return true;
  }

  bool loadFpgaData(bool isOutputLayer) {
    return true;
  }

private:
  FpgaMatrixPtr getFpgaInput(size_t idx) {
    auto l = dynamic_cast<IFpgaLayer*>(this->inputLayers_[idx].get());
    CHECK(l != nullptr);
    return l->fpgaValue;
  }
};

REGISTER_LAYER(fpga_data, FpgaDataLayer);
}  // namespace paddle

#endif
