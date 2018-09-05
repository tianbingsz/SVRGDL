#!/bin/bash
# Copyright (c) 2016 Baidu, Inc. All Rights Reserved

cd `dirname $0`
SRC_DIR=$PWD/../../../../proto/
SRC_INTERNAL_DIR=$PWD/../../../internals/proto
TRG_DIR=$PWD/../../../proto

if [ ! -d ${TRG_DIR} ]; then
  mkdir -p ${TRG_DIR}
fi

for filename in ${SRC_DIR}/*.m4
do
  filename=`basename $filename`  # remove path
  filename=${filename%.*}        # remove .m4
  m4 -Dreal=${PADDLE_REAL_TYPE} -I ${SRC_INTERNAL_DIR} ${SRC_DIR}/$filename.m4 > ${TRG_DIR}/$filename
done
