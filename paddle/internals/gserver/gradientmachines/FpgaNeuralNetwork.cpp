/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

#ifdef PADDLE_USE_FPGA
#include "FpgaNeuralNetwork.h"
#include "paddle/utils/CommandLineParser.h"
#include "paddle/internals/fpga_output/include/dnn-api/fpga.h"
#include "paddle/utils/CustomStackTrace.h"
#include "paddle/utils/Stat.h"
#include "paddle/utils/StringUtil.h"
#include <set>

P_DEFINE_bool(use_fpga, false, "use fpga to do DNN");
namespace paddle {

NeuralNetwork* newCustomNeuralNetwork(const std::string& name,
                                      NeuralNetwork* network) {
  return new FpgaNeuralNetwork(name, network);
}

void FpgaNeuralNetwork::loadFpgaData() {
  int devId = 0;
  // If --use_fpga is not set, don't check fpga device, use CPU to calculate
  // directly,
  // so the fpga device will not be opened at all
  useFpga_ = FLAGS_use_fpga && openAndCheckFpga(devId);
  if (!useFpga_) {
    fpga::log_init_usecpu();
    return;
  }

  for (auto& layer : layers_) {
    auto l = dynamic_cast<IFpgaLayer*>(layer.get());
    useFpga_ = l->loadFpgaData(std::find(outputLayers_.begin(),
                                         outputLayers_.end(), layer) !=
                                         outputLayers_.end());
    if (!useFpga_) {
      LOG(WARNING) << "Fpga data preparation failed, will use CPU instead";
      break;
    }
  }

  if (useFpga_) {
    fpga::log_init_usefpga();
  } else {
    fpga::log_init_usecpu();
  }
}

void FpgaNeuralNetwork::forward(const std::vector<Argument>& inArgs,
                                std::vector<Argument>* outArgs,
                                PassType passType) {
  CHECK_EQ(inArgs.size(), dataLayers_.size());
  outArgs->resize(outputLayers_.size());
  for (size_t i = 0; i != dataLayers_.size(); ++i) {
    dataLayers_[i]->setData(inArgs[i]);
  }

  if (FLAGS_use_fpga) {
    fpgaForward(passType);
  } else  {
    for (auto& layer : layers_) {
      REGISTER_TIMER_INFO("ForwardTimer", layer->getName().c_str());
      gLayerStackTrace.push(layer->getName());
      layer->forward(passType);
    }
    fpga::log_cpu_cal();
  }

  outArgs->clear();
  outArgs->reserve(outputLayers_.size());
  for (auto& layer : outputLayers_) {
    outArgs->push_back(layer->getOutput());
  }
  if (passType == PASS_TEST) {
    gLayerStackTrace.clear();
  }
}

void FpgaNeuralNetwork::fpgaForward(PassType passType) {
  bool fpgaRet = false;
  if (useFpga_) {
    for (auto& layer : layers_) {
      REGISTER_TIMER_INFO("ForwardTimer", layer->getName().c_str());
      auto l = dynamic_cast<IFpgaLayer*>(layer.get());
      fpgaRet = l != nullptr && l->fpgaForward(passType);
      if (!fpgaRet) {
        LOG(WARNING) << "Fpga forward operation failed, will fallback to CPU";
        useFpga_ = false;  // use cpu forever when fpga failed
        break;
      }
    }

    /*
     * Whole DNN forward OK
     */
    if (fpgaRet) {
      fpga::log_fpga_cal();
    }
  }

  /*
   * if fpgaRet == false:
   *    1. no fpgaexist
   *    2. fpga calculation failed
   * fallback to cpu calculation
   */
  if (!fpgaRet) {
    for (auto& layer : layers_) {
      REGISTER_TIMER_INFO("ForwardTimer", layer->getName().c_str());
      layer->forward(passType);
    }
    fpga::log_cpu_cal();
  }
}

void FpgaNeuralNetwork::init(const ModelConfig& config,
                             ParamInitCallback callback,
                             const std::vector<ParameterType>& parameterTypes,
                             bool useGpu) {
  ModelConfig realConfig = config;

  std::set<std::string> fpgaLayers;
  std::string prefix = "fpga_";

  Layer::registrar_.forEachType([&fpgaLayers, &prefix](
                                const std::string& name) {
    if (str::startsWith(name, prefix)) {
      fpgaLayers.insert(name.substr(prefix.size()));
    }
  });

  for (paddle::LayerConfig & l : *realConfig.mutable_layers()) {
    if (fpgaLayers.find(l.type()) != fpgaLayers.end()) {
      l.set_type(prefix + l.type());
    }
  }

  NeuralNetwork::init(realConfig, callback, parameterTypes, useGpu);
}

}  // namespace paddle
#endif
