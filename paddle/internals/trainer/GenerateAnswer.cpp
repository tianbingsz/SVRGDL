/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

/*
 * Generate subsequent sequence from the initial sequence.
 * In the config, TestData will specify the data which contain
 * the initial sequences. Each sequence from test data will be
 * used to generate a seuqnce.
 *
 * Example:

PYTHONPATH=~/code/idl/paddle/core/output/pylib     \
~/code/idl/paddle/core/output/bin/paddle_gen_answer \
--config ~/code/idl/dl/language/qa/conf/qa_lstm.conf \
--num_sequences_per_sample=1 \
--dict ~/data/generated/webquestions/webquestions.dict \
--eos_id=1635 \
--answer_context_id=1 \
--result_file=test.txt \
--model_path ~/models/qa_lstm/pass-00299 \
--config_args=generating=1 \
--use_gpu=0

 */

#include "paddle/utils/PythonUtil.h"

#include <fenv.h>
#include <fstream>

#include "paddle/utils/Flags.h"
#include "paddle/utils/Util.h"

#include "paddle/gserver/gradientmachines/GradientMachine.h"
#include "paddle/gserver/dataproviders/DataProvider.h"

#include "paddle/trainer/Trainer.h"

P_DECLARE_string(config);
P_DECLARE_int32(log_period);
P_DEFINE_int32(num_sequences_per_sample, 1,
               "Generate so many sequences per input sample");
P_DEFINE_int32(max_length, 100, "Max sequence length");
P_DEFINE_string(dict, "", "dictionary for mapping id to word");
P_DEFINE_int32(answer_context_id, 1,
               "Context id for answer from util.DataSetting");
P_DEFINE_int32(eos_id, 0,
               "End-of-qeustion id from util.DataSetting.answer_end_symbol");
P_DEFINE_string(result_file, "", "Result file name");
P_DEFINE_string(model_path, "", "Directory for the model");

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

int main(int argc, char** argv) {
  initMain(argc, argv);
  initPython(argc, argv);

  auto config = TrainerConfigHelper::createFromFlagConfig();

  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);

  std::vector<std::string> dict;
  if (!FLAGS_dict.empty()) {
    loadFileList(FLAGS_dict, dict);
  }

  std::ofstream outf(FLAGS_result_file);
  PCHECK(outf);

  std::unique_ptr<GradientMachine> gradMachine(GradientMachine::create(
      *config, GradientMachine::kTesting));

  std::unique_ptr<DataProvider> dataProvider;
  CHECK(config->hasTestDataConfig());
  dataProvider.reset(DataProvider::create(config->getTestDataConfig()));
  dataProvider->setSkipShuffle();

  gradMachine->loadParameters(FLAGS_model_path);

  DataBatch dataBatch;
  vector<Argument> outArgs;
  vector<Argument> inArgs;
  Argument feedback;
  feedback.ids = IVector::create(/* size= */ 1, FLAGS_use_gpu);
  Argument context;
  context.ids = IVector::create(/* size= */ 1, FLAGS_use_gpu);
  context.ids->setElement(0, FLAGS_answer_context_id);

  feedback.sequenceStartPositions =
      ICpuGpuVector::create(/* size= */ 2, /* useGpu= */ false);
  feedback.sequenceStartPositions->getMutableData(false)[0] = 0;
  feedback.sequenceStartPositions->getMutableData(false)[1] = 1;

  int64_t batchSize;
  int64_t sampleId = 0;
  int id;
  const int kContextSlot = 0;
  const int kWordIdSlot = 1;
  const int kNextWordIdSlot = 2;
  while ((batchSize = dataProvider->getNextBatch(/* size= */ 1, &dataBatch)) >
         0) {
    inArgs = dataBatch.getStreams();
    CHECK_EQ(3UL, inArgs.size());
    for (int i = 0; i < FLAGS_num_sequences_per_sample; ++i) {
      outf << sampleId << ": ";
      gradMachine->resetState();
      gradMachine->forward(inArgs, &outArgs, PASS_TEST);
      inArgs[kWordIdSlot] = feedback;
      inArgs[kContextSlot] = context;
      for (auto& arg : inArgs) {
        arg.sequenceStartPositions = feedback.sequenceStartPositions;
      }
      // copy the last token of question to word id slot
      feedback.ids->copyFrom(
          inArgs[kNextWordIdSlot].ids->getData() + batchSize - 1, 1);
      int length = 0;
      do {
        gradMachine->forward(inArgs, &outArgs, PASS_TEST);
        CHECK_EQ(1UL, outArgs.size()) << "Only one output is supported";
        CHECK(outArgs[0].ids) << "Only id output is supported";
        CHECK_EQ(1UL, outArgs[0].ids->getSize()) << "Wrong output length";
        feedback.ids->copyFrom(*outArgs[0].ids);
        id = outArgs[0].ids->getElement(0);
        if (!dict.empty()) {
          outf << dict[id];
        } else {
          outf << id;
        }
        if (id != FLAGS_eos_id) {
          outf << " ";
        }
        ++length;
      } while (id != FLAGS_eos_id && length < FLAGS_max_length);
      outf << endl;
    }
    ++sampleId;
    if (sampleId % FLAGS_log_period == 0) {
      LOG(INFO) << "Processed " << sampleId << " samples";
    }
    if (!dataProvider) break;
  }
}
