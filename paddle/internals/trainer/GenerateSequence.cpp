/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

/*
 * generate sequence for a given language model.
 *
 * Generate sequence given a fixed context.
 * The context is specified by TestData in the config. Each
 * data sample from test data will be used to generate one
 * sequence.
 *
 */

#include "paddle/utils/PythonUtil.h"

#include <fenv.h>
#include <fstream>
#include <math.h>

#include "paddle/utils/Flags.h"
#include "paddle/utils/Util.h"

#include "paddle/gserver/gradientmachines/GradientMachine.h"
#include "paddle/gserver/dataproviders/DataProvider.h"

#include "paddle/trainer/Trainer.h"

P_DECLARE_string(config);
P_DECLARE_int32(log_period);
P_DECLARE_int32(beam_size);

P_DEFINE_int32(num_sequences_per_sample, 1,
               "Generate so many sequences per input sample");
P_DEFINE_int32(max_length, 100, "Max sequence length");
P_DEFINE_string(dict, "", "dictionary for mapping id to word");
P_DEFINE_int32(bos_id, 0, "Begin-of-sequence id.");
P_DEFINE_int32(eos_id, 0, "End-of-sequence id.");
P_DEFINE_string(result_file, "", "Result file name");
P_DEFINE_string(model_path, "", "Directory for the model");

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

// used to represent partial sequence
struct Path {
  std::vector<int> ids;
  real logProb;
  MachineState machineState;

  Path() { logProb = 0; }

  Path(std::vector<int>& ids, real logProb, MachineState& machineState)
      : ids(ids), logProb(logProb), machineState(machineState) {}

  bool operator<(const Path& other) const { return (logProb > other.logProb); }
};

// Return top k (k == beam_size) optimal paths using beam search. The last
// element of inArgs is the Argument of feedback. gradMachine has MaxIdLayer
// as output and outArgs thus stores top k labels and their probabilities per
// position
void findNBest(const std::unique_ptr<GradientMachine>& gradMachine,
               vector<Argument>& inArgs, vector<Path>& finalPaths) {
  vector<Path> paths;
  Path emptyPath;
  paths.push_back(emptyPath);
  finalPaths.clear();
  gradMachine->resetState();
  Argument feedback = inArgs.back();
  feedback.ids->setElement(0, FLAGS_bos_id);

  real minFinalPathLogProb = 0;
  size_t beam = 0;
  int id;
  vector<Argument> outArgs;
  while (true) {  // iterate over each generated word
    vector<Path> newPaths;
    MachineState machineState;
    for (size_t j = 0; j < paths.size(); j++) {
      Path& path = paths[j];
      if (path.machineState.size() > 0) {
        gradMachine->setState(path.machineState);
        feedback.ids->setElement(0, path.ids.back());
      }
      gradMachine->forward(inArgs, &outArgs, PASS_TEST);
      gradMachine->getState(machineState);
      beam = outArgs[0].ids->getSize();
      for (size_t k = 0; k < beam; k++) {
        id = outArgs[0].ids->getElement(k);
        real prob = outArgs[0].in->getElement(0, k);
        vector<int> nids(path.ids);
        nids.push_back(id);
        real newLogProb = path.logProb + log(prob);
        Path newPath(nids, newLogProb, machineState);
        if (id == FLAGS_eos_id || (int)nids.size() >= FLAGS_max_length) {
          finalPaths.push_back(newPath);
          if (minFinalPathLogProb > newPath.logProb) {
            minFinalPathLogProb = newPath.logProb;
          }
        } else {
          newPaths.push_back(newPath);
        }
      }
    }

    if (newPaths.size() == 0) {
      break;
    }
    std::nth_element(newPaths.begin(),
                     newPaths.begin() + min(beam, newPaths.size()),
                     newPaths.end());
    if (newPaths.size() > beam) {
      newPaths.resize(beam);
    }
    // pathA < pathB means pathA.logProb > pathB.logProb
    real maxPathLogProb =
        std::min_element(newPaths.begin(), newPaths.end())->logProb;
    if (finalPaths.size() >= beam && minFinalPathLogProb >= maxPathLogProb) {
      break;
    }
    paths = newPaths;
  }  // end while

  std::partial_sort(finalPaths.begin(),
                    finalPaths.begin() + min(beam, finalPaths.size()),
                    finalPaths.end());
  if (finalPaths.size() > beam) {
    finalPaths.resize(beam);
  }
}

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
  if (config->hasTestDataConfig()) {
    dataProvider.reset(DataProvider::create(config->getTestDataConfig()));
    dataProvider->setSkipShuffle();
  }

  gradMachine->loadParameters(FLAGS_model_path);

  DataBatch dataBatch;
  vector<Argument> inArgs;
  Argument feedback;
  feedback.ids = IVector::create(/* size= */ 1, FLAGS_use_gpu);

  feedback.sequenceStartPositions =
      ICpuGpuVector::create(/* size= */ 2, /* useGpu= */ false);
  feedback.sequenceStartPositions->setElement(0, 0, false);
  feedback.sequenceStartPositions->setElement(1, 1, false);

  int64_t batchSize;
  int64_t sampleId = 0;
  while (!dataProvider ||
         (batchSize = dataProvider->getNextBatch(/* size= */ 1, &dataBatch)) >
             0) {
    inArgs = dataBatch.getStreams();
    for (auto& arg : inArgs) {
      arg.sequenceStartPositions = feedback.sequenceStartPositions;
    }
    inArgs.push_back(feedback);

    for (int i = 0; i < FLAGS_num_sequences_per_sample; ++i) {
      vector<Path> finalPaths;
      findNBest(gradMachine, inArgs, finalPaths);

      for (size_t j = 0; j < finalPaths.size(); j++) {
        Path& p = finalPaths[j];
        outf << sampleId << "-" << j << ": ";
        for (size_t k = 0; k < p.ids.size(); k++) {
          int id = p.ids[k];
          if (!dict.empty()) {
            outf << dict[id];
          } else {
            outf << id;
          }
          outf << " ";
        }
        outf << p.logProb << endl;
      }
    }  // end for
    ++sampleId;
    if (sampleId % FLAGS_log_period == 0) {
      LOG(INFO) << "Processed " << sampleId << " samples";
    }

    if (!dataProvider) break;
  }  // end while
}
