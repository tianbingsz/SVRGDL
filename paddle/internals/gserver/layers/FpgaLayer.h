/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

#pragma once
#include "paddle/utils/GlobalConstants.h"
#include "Fpga.h"
namespace paddle {

class IFpgaLayer {
public:
    IFpgaLayer(): fpgaValue(nullptr) {}
    virtual ~IFpgaLayer() {}

    /**
     * Fpga Forward propagation.
     * The default implemention of fpgaForward is do nothing and return false.
     * All inherited implemention should call Layer::foward().
     */
    virtual bool fpgaForward(PassType passType) { return false; }

    /**
     * loadFpgaData.
     * The default implemention of loadFpgaData is do nothing and return true.
     * All inherited implemention should init FpgaMatrixes and return true.
     */
    virtual bool loadFpgaData(bool isOutputLayer) { return false; }


    FpgaMatrixPtr fpgaValue;  // fpga matrix.
};

}  // namespace paddle
