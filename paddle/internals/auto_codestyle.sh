#!/bin/bash
# Copyright (c) 2016 Baidu, Inc. All Rights Reserved

cd "$(dirname "$0")"/../

function print_usage(){
  echo "Paddle code style check and format tools"
  echo "Usage:"
  echo "    sh run_cpplint.sh check"
  echo "    sh run_cpplint.sh format"
  echo "Options:"
  echo "    check means check all source file format. return 0 if no error."
  echo "    format means format all source file by clang-format."
  echo ""
  exit 1
}

if [  $# -eq 0 ]; then
  print_usage
elif [ $1 == "check" ]; then
  MODE=0
elif [ $1 == "format" ]; then
  MODE=1
else
  print_usage
fi

if [ $MODE -eq 0 ]; then
echo "Checking source files code style..."
else
echo "Formatting source files..."
fi


# CONFIGS
SOURCES=`find . -name '*.cpp' \
       -o -name '*.h' \
       -o -name '*.cu' \
       -o -name '*.cuh'`
DROP_PATTERNS=("site-packages" 
    "ImportanceSampler"
    "internals/cblas/include"
    "internals/mkl/include"
    "conv/conv_util.cu"
    "conv/filter_acts.cu"
    "conv/img_acts.cu"
    "conv/mytime"
    "conv/nvmatrix"
    "conv/weight_acts"
    "conv/cudaconv2"
    "cpuvec"         # TODO(yuyang18): this package will be refactor into common
    "fpga_output"    # Do not check fpga module
    "LtrDataProvider" # The DataProvider Will be refactor later
    "MultiDataProvider"
    "MetricDataProvider"  # The DataProvider will be refactor later
    "picojson.h"
    "Paddle_wrap"	# swig file
    "output"
    "gserver/fpga"
    "gserver/extension/"
    "proto"
    "x86_64-scm-linux-gnu" # Drop toolchain
    "avx_mathfun.h"
    "hl_cuda_cudnn.cu" # This file will be refactor
    "cudnn.h" # This file will be refactor
    "build"
)
LINT_CMD="python scripts/cpplint.py "

LINT_FILTERS=("whitespace/indent"   # paddle do not indent \
                                      public/protected/private in class
              "runtime/references"  # paddle use mutable reference. But it is \
                                      not recommanded.
              "build/include"       # paddle use relative path for include.
              "build/c++11"         # we use <thread>, <mutex>, etc.
              "readability/casting" # paddle use c-style casting. But it is not\
                                    # recommanded.
)



# concat lint filter to lint command
filter_args=""
for filter in ${LINT_FILTERS[*]}
do
  filter_args=$filter_args","-$filter
done
filter_args=`echo $filter_args | sed 's/^,//'`
LINT_CMD="$LINT_CMD --filter=$filter_args"

ERROR_FILE=`mktemp`

function run_lint() {
  local dn=$( dirname "$1" )
  local fn=$( basename "$1" )
  local md5fn=$dn/".lint.$fn.md5"
  md5sum -c --quiet $md5fn 1>/dev/null 2>/dev/null  # cache lint result,
                                                    # if file not modified.
  if [ $? -ne 0 ]; then
    ERROR_MSG=`$LINT_CMD $1 2>&1`
    if [ $? -ne 0 ]; then 
      # ERROR
      echo "Check $fn failed" >> $ERROR_FILE
      >&2 echo $ERROR_MSG
      exit 1
    fi
  fi
  md5sum $1 > $md5fn
}

function run_autoformat(){
  clang-format -i -style=file $1
  if [ $? -ne 0 ]; then
      echo "Format file $1 error!"
      exit 1
  fi
}

# Main
for filename in $SOURCES
do
  isDrop="false"
  
  for pattern in ${DROP_PATTERNS[*]}
  do
    echo "$filename" | grep -q $pattern
    if [ $? -eq 0 ]; then # Pattern Matched.
      isDrop="true"
      break
    fi
  done

  $isDrop
  if [ $? -ne 0 ]; then # Is not Droped
    if [ $MODE -eq 0 ]; then
      run_lint $filename &
    else
      run_autoformat $filename &
    fi
  fi
done

wait

if [[ -s $ERROR_FILE ]]; then
  cat $ERROR_FILE
  rm $ERROR_FILE
  exit 1
else
  rm $ERROR_FILE
fi

if [ $MODE -eq 0 ]; then
echo 'Passed code style check.'
else
echo "Done."
fi

exit 0
