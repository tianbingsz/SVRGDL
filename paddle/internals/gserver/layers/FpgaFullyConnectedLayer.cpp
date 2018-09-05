/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

#ifdef PADDLE_USE_FPGA
#include "FpgaLayer.h"
#include "paddle/gserver/layers/FullyConnectedLayer.h"
#include "paddle/utils/Stat.h"

namespace paddle {

static inline int paddleToFpgaAct(std::string actType) {
  if ("stanh" == actType) {
    return fpga::ACTIVATE_TYPE_STANH;
  } else if ("relu" == actType) {
    return fpga::ACTIVATE_TYPE_RELU;
  } else if ("tanh" == actType) {
    return fpga::ACTIVATE_TYPE_TANH;
  } else if ("softmax" == actType) {
    return fpga::ACTIVATE_TYPE_SOFTMAX;
  } else if ("linear" == actType) {
    return fpga::ACTIVATE_TYPE_LINEAR;
  }

  return fpga::ACTIVATE_TYPE_NONE;
}

class FpgaFullyConnectedLayer : public FullyConnectedLayer,
    public IFpgaLayer {
public:
  explicit FpgaFullyConnectedLayer(const LayerConfig& config):
    FullyConnectedLayer(config), activationFpgaSupport_(false) {}

  bool fpgaForward(PassType passType) {
    Layer::forward(passType);

    /* malloc memory for the output_ if necessary */
    FpgaMatrixPtr input = this->getFpgaInput(0);
    int batchSize = input->get_height();
    int size = getSize();

    {
      REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
      resizeOutput(batchSize, size);
    }

    FpgaMatrixPtr output = this->fpgaValue;
    int ret = output->resize(batchSize, size);
    if (0 != ret) {
      return false;
    }
    output->zero_mem();

    /*
     * Fpga forward:
     * Limitation:
     *   1. Fpga only support C = A * B with FpgaMatrix.mul(), C = C' + C is done
     *      by CPU. Thus, C = A * B + C is implemented by FpgaMatrix.mul_ofloat()
     *      which contains mul() and add()
     *   2. Fpga does not support all activation funcs (such as softmax)
     *   3. Fpga has different data layout than Cpu Matrix, so we need to use
     *      copy_from_in_row and copy_to_in_row to transfer data, also we provide
     *      optimized act funcs to do activation(relu, linear, sigmoid ...)
     *
     * Fpga forward implementation has Two Major Cases:
     *   1. Layer has only one input and Fpga support the activation func:
     *     In this case there is no need to do C = C' + C
     *     Two sub cases:
     *     A. hidden layers: FpgaMatrix.mul() to do matrix mul and act
     *     B. last layer: FpgaMatrix.mul() and copy data to Paddle CpuDataMatrix
     *
     *   2. Layer has more than one input or Fpga doesn't support activation func:
     *     In this case data should be copy to CPU to do C = C' + C and activation
     *     Two sub cases:
     *     A. fpga-dnnlib does not provide optimized activation functions:
     *        FpgaMatrix->copy_to transfer data to Paddle CpuDataMatrix
     *        forwardActivation() to do activation
     *        I. hidden layers: FpgaMatrix->copy_from transfers data to Fpga
     *        II.last Layer: do nothing
     *
     *     B. fpga-dnnlib provides the optimized-cpu-activation functions:
     *        FpgaMatrix->activate do activation
     *        I. hidden layers: FpgaMatrix->reload data to fpga device
     *        II.last layer: FpgaMatrix->copy_to transfers data to CpuDataMatrix
     */
    size_t inputCnt = inputLayers_.size();
    MatrixPtr outV = getOutputValue();
    if ((1 == inputCnt) && (true == activationFpgaSupport_)) {
      auto input = getInput(0);
      FpgaMatrixPtr inputMatrix = this->getFpgaInput(0);
      CHECK(fpgaValue)
          << "The input of fpga 'fc' layer must be fpga matrix";
      REGISTER_TIMER_INFO("FwMulTimer", getName().c_str());
      ret = output->mul(inputMatrix.get(), fpgaWeights_[0].get());
      if (0 != ret) {
        return false;
      }

      if (isOutputLayer_) {
        outV->zeroMem();
        real *cpuData = outV->getData();
        ret = output->copy_to_in_row((float *)cpuData, batchSize, size);
        if (0 != ret) {
          return false;
        }
      }
    } else {
      for (size_t i = 0; i != inputCnt; ++i) {
        FpgaMatrixPtr inputMatrix = this->getFpgaInput(i);
        CHECK(inputMatrix)
            << "The input of fpga 'fc' layer must be fpga matrix";
        REGISTER_TIMER_INFO("FwMulTimer", getName().c_str());
        // ofloat perform C = A * B + C
        ret = output->mul_ofloat(inputMatrix.get(), fpgaWeights_[i].get());
        if (0 != ret) {
          return false;
        }
      }

      // C' = activate(C), the activation is done by CPU in FpgaMatrix
      REGISTER_TIMER_INFO("FwCpuAtvTimer", getName().c_str());
      bool actSucc = output->activate(fpgaWeights_[0].get());

      // fpga-dnnlib DOESNOT provide optmized act func
      if (false == actSucc) {
        outV->zeroMem();
        real *cpuData = outV->getData();
        ret = output->copy_to_in_row((float *)cpuData, batchSize, size);
        if (0 != ret) {
          return false;
        }

        /* activation */ {
          REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
          forwardActivation();
        }
        if (!isOutputLayer_) {
          ret = output->copy_from_in_row((float *)cpuData, batchSize, size);
          if (0 != ret) {
            return false;
          }
        }
      } else {
        // fpga-dnnlib DOES provide optmized act func
        // actSucc == true
        if (!isOutputLayer_) {
          ret = output->reload_data();
          if (0 != ret) {
            return false;
          }
        } else {  // isOutputLayer_ == true
          outV->zeroMem();
          real *cpuData = outV->getData();
          ret = output->copy_to_in_row((float *)cpuData, batchSize, size);
          if (0 != ret) {
            return false;
          }
        }
      }  // end actSucc == true
    }

    return true;
  }

  bool loadFpgaData(bool isOutputLayer) {
    for (size_t i = 0; i < inputLayers_.size(); i++) {
      size_t height = inputLayers_[i]->getSize();
      size_t width = getSize();

      // 1. Get weight data
      MatrixPtr weightMatrix = getWeight(i).getW();
      float *weightSrc = weightMatrix->getData();
      std::string paddleActType = config_.active_type();

      // 2. get activate type
      int actType = paddleToFpgaAct(paddleActType);
      activationFpgaSupport_ = fpga::activation_hardware_support(actType);

      // 3. Build fpgaWeight
      FpgaWeightMatrixPtr fpgaWeight =
          std::make_shared<fpga::FpgaDsp24bitWeightMatrix>(0, actType);

      if (fpgaWeight->resize(height, width) != 0) {
        return false;
      }
      if (fpgaWeight->copy_from_in_row(weightSrc, height, width) != 0) {
        return false;
      }
      if (fpgaWeight->load_zero_bias(width) != 0) {
        return false;
      }

      fpgaWeights_.push_back(fpgaWeight);
    }

    if (biasParameter_.get() != NULL) {
      // get bias data
      MatrixPtr biasMatrix = biases_->getW();
      float *biasSrc = biasMatrix->getData();

      /* init layer biases into first weight matrix */
      FpgaWeightMatrixPtr fpgaWeight = fpgaWeights_[0];
      if (fpgaWeight->load_bias(biasSrc, getSize()) != 0) {
        return false;
      }
    }

    /* fpga output data will all added to data Matrix[0] */
    fpgaValue = std::make_shared<fpga::FpgaDsp24bitDataMatrix>(0);
    isOutputLayer_ = isOutputLayer;
    return true;
  }

  FpgaMatrixPtr getFpgaInput(size_t idx) {
    auto l = dynamic_cast<IFpgaLayer*>(this->inputLayers_[idx].get());
    CHECK(l != nullptr);
    return l->fpgaValue;
  }

protected:
  std::vector<FpgaWeightMatrixPtr> fpgaWeights_;
  bool activationFpgaSupport_;
  /// is layer at output or not.
  bool isOutputLayer_;
};
REGISTER_LAYER(fpga_fc, FpgaFullyConnectedLayer);
}  // namespace paddle

#endif
