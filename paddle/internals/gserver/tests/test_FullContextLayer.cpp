/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include <gtest/gtest.h>
#include "paddle/gserver/layers/Layer.h"
#include "paddle/gserver/layers/DataLayer.h"
#include "paddle/internals/gserver/layers/FullContextLayer.h"
#include "paddle/gserver/tests/TestUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

struct Parameters {
  ParameterPtr embedding;
  ParameterPtr context;
  ParameterPtr fullMatrix;
  ParameterPtr fullContext;
  ParameterPtr fullMatrixBias;
};

struct Layers {
  LayerPtr dataLayer;
  LayerPtr fullContextLayer;
  LayerPtr tablePLayer;
  LayerPtr contextPLayer;
  LayerPtr fullMatrixPLayer;
  LayerPtr maxLayer;
};

struct Configs {
  int wordDim;
  int dictSize;
  int contextStart;
  int contextLength;
  int beginPad;
  int endPad;
  int totalPad;
  int matrixDim;
  int batchSize;
  bool hasBias;
  string activation;
};

/**
 * a: 1 x wordDim
 * b: wordDim x matrixDim
 * result: 1 x matrixDim
 */
void simpleMul(const Configs& config, const real* a, const real* b,
               real* result) {
  for (int dest = 0; dest < config.matrixDim; dest++) {
    double sum = 0.0;
    for (int src = 0; src < config.wordDim; src++) {
      sum += (double)a[src] * (double)b[src * config.matrixDim + dest];
    }
    result[dest] = sum;
  }
}

void realCalc(Parameters parameters, Layers layers) {
  LayerMap layerMap;
  ParameterMap parameterMap;
  layerMap["data"] = layers.dataLayer;
  parameterMap["fullcontext"] = parameters.fullContext;
  parameterMap["fullmatrix.bias"] = parameters.fullMatrixBias;
  parameterMap["table"] = parameters.embedding;
  parameterMap["context"] = parameters.context;
  parameterMap["fullmatrix"] = parameters.fullMatrix;
  layerMap["fullcontext"] = layers.fullContextLayer;
  layers.fullContextLayer->init(layerMap, parameterMap);
  layerMap["table"] = layers.tablePLayer;
  layers.tablePLayer->init(layerMap, parameterMap);
  layerMap["context"] = layers.contextPLayer;
  layers.contextPLayer->init(layerMap, parameterMap);
  layerMap["fullmatrix"] = layers.fullMatrixPLayer;
  layers.fullMatrixPLayer->init(layerMap, parameterMap);
  layerMap["max"] = layers.maxLayer;
  layers.maxLayer->init(layerMap, parameterMap);
  // real calc
  layers.fullContextLayer->forward(PASS_GC);
  Argument output = layers.fullContextLayer->getOutput();
  // real calc
  layers.tablePLayer->forward(PASS_GC);
  layers.contextPLayer->forward(PASS_GC);
  layers.fullMatrixPLayer->forward(PASS_GC);
  layers.maxLayer->forward(PASS_GC);
  Argument maxLayerOutput = layers.maxLayer->getOutput();
  // check eq
  EXPECT_EQ(maxLayerOutput.value->getHeight(), output.value->getHeight());
  EXPECT_EQ(maxLayerOutput.value->getWidth(), output.value->getWidth());
  for (size_t i = 0; i < maxLayerOutput.value->getHeight(); i++) {
    for (size_t j = 0; j < maxLayerOutput.value->getWidth(); j++) {
      EXPECT_TRUE(approximatelyEqual(maxLayerOutput.value->getElement(i, j),
                                     output.value->getElement(i, j), 1.0e-4));
    }
  }
}

Parameters createParametersByConfig(const Configs& config, int loadParameters) {
  Parameters paras;
  ParameterConfig embeddingConf, contextConf, fullMatrixConf, fullContextConf,
      fullMatrixBiasConf;
  fullMatrixBiasConf.set_size(config.matrixDim);
  fullMatrixBiasConf.add_dims(1);
  fullMatrixBiasConf.add_dims(config.matrixDim);
  fullMatrixBiasConf.set_initial_smart(true);
  // set size
  embeddingConf.set_size(config.dictSize * config.wordDim);
  embeddingConf.add_dims(config.dictSize);
  embeddingConf.add_dims(config.wordDim);
  embeddingConf.set_initial_smart(true);
  //
  contextConf.set_size(config.totalPad * config.wordDim);
  contextConf.add_dims(config.totalPad);
  contextConf.add_dims(config.wordDim);
  contextConf.set_initial_smart(true);
  //
  fullMatrixConf.set_size(config.contextLength * config.wordDim *
                          config.matrixDim);
  fullMatrixConf.add_dims(config.contextLength * config.wordDim);
  fullMatrixConf.add_dims(config.matrixDim);
  fullMatrixConf.set_initial_smart(true);
  //
  fullContextConf.set_size((config.dictSize + config.totalPad) *
                           config.matrixDim * config.contextLength);
  fullContextConf.add_dims(config.dictSize + config.totalPad);
  fullContextConf.add_dims(config.matrixDim * config.contextLength);
  fullContextConf.set_initial_smart(true);
  // done
  // create Parameters
  paras.embedding = std::make_shared<Parameter>(embeddingConf, FLAGS_use_gpu);
  paras.context = std::make_shared<Parameter>(contextConf, FLAGS_use_gpu);
  paras.fullMatrix = std::make_shared<Parameter>(fullMatrixConf, FLAGS_use_gpu);
  paras.fullContext =
      std::make_shared<Parameter>(fullContextConf, FLAGS_use_gpu);
  paras.fullMatrixBias =
      std::make_shared<Parameter>(fullMatrixBiasConf, FLAGS_use_gpu);
  paras.embedding->initialize();
  paras.context->initialize();
  paras.fullMatrix->initialize();
  paras.fullMatrixBias->initialize();
  paras.fullContext->initialize();
  if (loadParameters == 1) {
    paras.embedding->load("embedding.w0");
    paras.context->load("context.w");
    paras.fullMatrix->load("fullmatrix.w0");
  } else {
    paras.embedding->randomize();
    paras.context->randomize();
    paras.fullMatrix->randomize();
    paras.fullMatrixBias->randomize();
  }
  for (int i = 0; i < config.dictSize + config.totalPad; i++) {
    for (int j = 0; j < config.contextLength; j++) {
      if (i < config.dictSize) {
        simpleMul(config, paras.embedding->getBuf(PARAMETER_VALUE)->getData() +
                              config.wordDim * i,
                  paras.fullMatrix->getBuf(PARAMETER_VALUE)->getData() +
                      config.matrixDim * j * config.wordDim,
                  paras.fullContext->getBuf(PARAMETER_VALUE)->getData() +
                      i * config.matrixDim * config.contextLength +
                      config.matrixDim * j);
      } else {
        simpleMul(config, paras.context->getBuf(PARAMETER_VALUE)->getData() +
                              config.wordDim * (i - config.dictSize),
                  paras.fullMatrix->getBuf(PARAMETER_VALUE)->getData() +
                      config.matrixDim * j * config.wordDim,
                  paras.fullContext->getBuf(PARAMETER_VALUE)->getData() +
                      i * config.matrixDim * config.contextLength +
                      config.matrixDim * j);
      }
    }
  }
  return paras;
}

Layers createLayersByConfig(const Configs& config) {
  Layers layers;
  // data layer, as input for full context
  LayerConfig dataConfig;
  dataConfig.set_name("data");
  dataConfig.set_type("data");
  dataConfig.set_size(config.dictSize);
  layers.dataLayer = LayerPtr(new DataLayer(dataConfig));
  Argument data;
  // generate data
  data.ids = VectorT<int>::create(config.batchSize, false /*useGpu*/);
  data.ids->rand(config.dictSize);
  generateSequenceStartPositions(config.batchSize,
                                 data.sequenceStartPositions);
  DataLayerPtr ddataLayer =
      std::dynamic_pointer_cast<DataLayer>(layers.dataLayer);
  ddataLayer->setData(data);
  ddataLayer->forward(PASS_GC);
  // layer config
  LayerConfig layerConfig;
  layerConfig.set_active_type(config.activation);
  layerConfig.set_type("fullcontext");
  layerConfig.set_size(config.matrixDim);
  layerConfig.set_name("fullcontext");
  layerConfig.mutable_full_context_config()->set_input_dim(config.dictSize);
  layerConfig.mutable_full_context_config()->set_context_length(
      config.contextLength);
  layerConfig.mutable_full_context_config()->set_begin_pad(config.beginPad);
  layerConfig.mutable_full_context_config()->set_end_pad(config.endPad);
  // bias
  if (config.hasBias) {
    layerConfig.set_bias_parameter_name("fullmatrix.bias");
  }
  layerConfig.add_inputs();
  LayerInputConfig& input = *(layerConfig.mutable_inputs(0));
  input.set_input_layer_name("data");
  input.set_input_parameter_name("fullcontext");
  layers.fullContextLayer = Layer::create(layerConfig);
  // normal network:
  // table projection
  ProjectionConfig tablePConf;
  tablePConf.set_type("table");
  tablePConf.set_input_size(config.dictSize);
  tablePConf.set_output_size(config.wordDim);
  // layer
  LayerConfig tableLConf;
  tableLConf.set_name("table");
  tableLConf.set_type("mixed");
  tableLConf.set_size(config.wordDim);
  tableLConf.add_inputs();
  LayerInputConfig& tableLInput = *(tableLConf.mutable_inputs(0));
  *tableLInput.mutable_proj_conf() = tablePConf;
  tableLInput.set_input_layer_name("data");
  tableLInput.set_input_parameter_name("table");
  layers.tablePLayer = Layer::create(tableLConf);
  // context projection
  ProjectionConfig contextPConf;
  contextPConf.set_type("context");
  contextPConf.set_input_size(config.wordDim);
  contextPConf.set_context_start(config.contextStart);
  contextPConf.set_context_length(config.contextLength);
  contextPConf.set_trainable_padding(true);
  contextPConf.set_output_size(config.wordDim * config.contextLength);
  // layer
  LayerConfig contextLConf;
  contextLConf.set_name("context");
  contextLConf.set_type("mixed");
  contextLConf.set_size(config.wordDim * config.contextLength);
  contextLConf.add_inputs();
  LayerInputConfig& contextLInput = *(contextLConf.mutable_inputs(0));
  *contextLInput.mutable_proj_conf() = contextPConf;
  contextLInput.set_input_layer_name("table");
  contextLInput.set_input_parameter_name("context");
  layers.contextPLayer = Layer::create(contextLConf);
  // full matrix projection
  ProjectionConfig fullMatrixPConf;
  fullMatrixPConf.set_type("fc");
  fullMatrixPConf.set_input_size(config.wordDim * config.contextLength);
  fullMatrixPConf.set_output_size(config.matrixDim);
  // layer
  LayerConfig fullMatrixLConf;
  fullMatrixLConf.set_name("fullmatrix");
  fullMatrixLConf.set_type("mixed");
  fullMatrixLConf.set_active_type(config.activation);
  fullMatrixLConf.set_size(config.matrixDim);
  fullMatrixLConf.add_inputs();
  LayerInputConfig& fullMatrixLInput = *(fullMatrixLConf.mutable_inputs(0));
  *fullMatrixLInput.mutable_proj_conf() = fullMatrixPConf;
  fullMatrixLInput.set_input_layer_name("context");
  fullMatrixLInput.set_input_parameter_name("fullmatrix");
  // bias
  if (config.hasBias) {
    fullMatrixLConf.set_bias_parameter_name("fullmatrix.bias");
  }
  layers.fullMatrixPLayer = Layer::create(fullMatrixLConf);
  // max layer
  LayerConfig maxLConf;
  maxLConf.set_type("max");
  maxLConf.set_size(config.matrixDim);
  maxLConf.add_inputs();
  LayerInputConfig& maxLInput = *(maxLConf.mutable_inputs(0));
  maxLInput.set_input_layer_name("fullmatrix");
  layers.maxLayer = Layer::create(maxLConf);
  return layers;
}

void testUsingConfigs(const Configs& configs) {
  Parameters parameters = createParametersByConfig(configs, 0);
  Layers layers = createLayersByConfig(configs);
  realCalc(parameters, layers);
}

TEST(fullflow, fullflow) {
  Configs configs;
  configs.wordDim = 128;
  configs.dictSize = 1000;
  configs.contextStart = -1;
  configs.contextLength = 3;
  configs.beginPad = 1;
  configs.endPad = 1;
  configs.totalPad = 2;
  configs.matrixDim = 96;
  configs.batchSize = 100;
  for (auto hasBias : {true, false}) {
    for (auto activation : {"sigmoid", "tanh"}) {
      configs.hasBias = hasBias;
      configs.activation = activation;
      testUsingConfigs(configs);
    }
  }
}

void mergeParameters() {
  Configs configs;
  configs.wordDim = 64;
  configs.dictSize = 1451594;
  configs.contextStart = -1;
  configs.contextLength = 3;
  configs.beginPad = 1;
  configs.endPad = 1;
  configs.totalPad = 2;
  configs.matrixDim = 128;
  Parameters paras = createParametersByConfig(configs, 1);
  paras.fullContext->save("finaloutput");
}

int main(int argc, char** argv) {
  FLAGS_use_gpu = false;
  if (argc == 1) {
    srand(time(NULL));
    initMain(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  } else {
    mergeParameters();
    return 0;
  }
}
