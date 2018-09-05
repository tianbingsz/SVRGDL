/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#pragma once

#include "paddle/gserver/layers/Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/ThreadLocal.h"

namespace paddle {

/*
 * FullContextLayer: combine the following 4 layers into one:
 * (a) MixedLayer + TabelProjection,
 * (b) MixedLayer + ContextProjection,
 * (c) MixedLayer + FullMatrixProjection,
 * (d) MaxLayer.
 * USAGE:
 * (1) train a neural network with layers like:
 * [python config starts here]
 * Layer(
 *   name = slot_names,
 *   type = "data",
 *   size = word_dim,
 * )
 * Layer(
 *   name = slot_names + "_embedding_",
 *   type = "mixed",
 *   size = wordvec_dim,
 *   bias = False,
 *   inputs = TableProjection(slot_names, parameter_name = "embedding.w0"),
 * )
 * Layer(
 *   name = slot_names + "_context_",
 *   type = "mixed",
 *   size = context_dim,
 *   bias = False,
 *   inputs = ContextProjection(
 *     slot_names + "_embedding_",
 *     context_start = context_start,
 *     context_length = context_length,
 *     trainable_padding = True, parameter_name = "context.w")
 * )
 * Layer(
 *   name = slot_names + "_matrix_",
 *   type = "mixed",
 *   bias = Bias(parameter_name = "fullmatrix.bias"),
 *   size = matrix_dim,
 *   active_type = "tanh",
 *   inputs = [
 *     FullMatrixProjection(slot_names + "_context_", parameter_name =
 *"fullmatrix.w0"),
 *   ]
 * )
 * Layer(
 *   name = slot_names + "_max_",
 *   type = "max",
 *   inputs = [
 *     slot_names + "_matrix_",
 *   ],
 * )
 * [python config ends here]
 * (2) "MERGE" the parameters into ONE parameter for FullContextLayer, and then
 *modify the config as follows:
 * [python config starts here]
 * Layer(
 *   name = slot_names,
 *   type = "data",
 *   size = word_dim,
 * )
 * Layer(
 *   name = slot_names + "_fullcontext_",
 *   type = "fullcontext",
 *   active_type = "tanh",
 *   size = matrix_dim,
 *   context_start = context_start,
 *   context_length = context_length,
 *   bias = Bias(parameter_name = "fullmatrix.bias"),
 *   inputs = [
 *     Input(slot_names, parameter_name = "finaloutput"),
 *   ]
 * )
 * [python config ends here]
 * (3) how to "MERGE" parameters into ONE?
 * use gserver/tests/test_FullContextLayer.cpp:
 * (3.1) edit function mergeParameters() to fit YOUR OWN DIMS.
 * (3.2) compile the test_FullContextLayer binary file.
 * (3.3) prepare the 3 input parameters (embedding.w0, context.w and
 *fullmatrix.w0).
 * (3.4) run "./test_FullContextLayer 1". you will see the "merged" parameter in
 *"finaloutput".
 *
 * ATTENTION: This layer is designed for fast prediction (forward), NOT FOR
 *TRAINING.
 */

class FullContextLayer : public Layer {
protected:
  WeightList weights_;
  std::unique_ptr<Weight> biases_;
  MatrixPtr allResultMatrix_;
  MatrixPtr resultMaxMatrix_;

public:
  explicit FullContextLayer(const LayerConfig& config) : Layer(config) {}

  ~FullContextLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  Weight& getWeight(int idx) { return *weights_[idx]; }

  void prefetch();
  void forward(PassType passType);
  void backward(const UpdateCallback& callback = nullptr);
};

}  // namespace paddle
