/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

#pragma once
#ifdef PADDLE_USE_FPGA
#include "paddle/gserver/gradientmachines/NeuralNetwork.h"
#include "paddle/internals/gserver/layers/FpgaLayer.h"
namespace paddle {

class FpgaNeuralNetwork : public NeuralNetwork {
public:
  FpgaNeuralNetwork(const std::string& name, NeuralNetwork* rootNetwork):
    NeuralNetwork(name, rootNetwork), useFpga_(false) {
    LOG(INFO) << "New Fpga Neural Network";
  }

  void loadFpgaData();
  void forward(const std::vector<Argument>& inArgs,
               std::vector<Argument>* outArgs,
               PassType passType);

  void fpgaForward(PassType passType);

  void init(const ModelConfig& config, ParamInitCallback callback,
            const std::vector<ParameterType>& parameterTypes, bool useGpu);

protected:
  void onLoadParameter() {
    this->loadFpgaData();
  }

private:
  bool useFpga_;
};


NeuralNetwork* newCustomNeuralNetwork(
    const std::string& name, NeuralNetwork* network);
}  // namespace paddle
#endif
