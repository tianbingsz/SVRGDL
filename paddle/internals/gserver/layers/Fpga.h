/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

#pragma once

#include <memory>

#ifdef PADDLE_USE_FPGA

#include "fpga_basic.h"
#include "fpga_24bit_matrix.h"

namespace fpga = baidu::fpga::dnnlib;
namespace paddle {

inline bool openAndCheckFpga(int dev_id) {
  return fpga::open_and_check_dsp_device(dev_id);
}

typedef std::shared_ptr<fpga::FpgaDsp24bitMatrix> FpgaMatrixPtr;
typedef std::shared_ptr<fpga::FpgaDsp24bitWeightMatrix> FpgaWeightMatrixPtr;
typedef std::shared_ptr<fpga::FpgaDsp24bitDataMatrix> FpgaDataMatrixPtr;
}
#else
namespace paddle {
typedef std::shared_ptr<void> FpgaMatrixPtr;
typedef std::shared_ptr<void> FpgaWeightMatrixPtr;
typedef std::shared_ptr<void> fpgaDataMatrixPtr;
}
#endif
