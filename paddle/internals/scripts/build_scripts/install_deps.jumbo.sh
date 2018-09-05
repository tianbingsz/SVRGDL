#!/bin/bash
# Copyright (c) 2016 Baidu, Inc. All Rights Reserved

function abort(){
    echo "An error occurred. Exiting..." 1>&2
    exit 1
}

trap 'abort' 0
set -e

jumbo add_repo http://m1-idl-gpu2-bak31.m1.baidu.com:8088/jumbo/alpha/

jumbo install gcc46 cmake ccache protobuf google-glog\
           google-gflags google-gtest python swig\
           python-pip python-numpy cudnn-v2 git-svn

pip install --index-url=http://pip.baidu.com/pypi/simple\
           --trusted-host=pip.baidu.com -U recommonmark Sphinx sphinx_rtd_theme wheel
echo ""
echo "====================================================================================="
echo "Paddle dependencies installed by jumbo. To build paddle, you need to use gcc46 at least."
echo '  If you want to use gcc46, export PATH=${JUMBO_ROOT}/opt/gcc46/bin:${PATH} .'
echo '  If you want to use gcc48, which is deployed in /opt/, just export PATH=/opt/compiler/gcc-4.8.2/bin:$PATH'
echo "Just pick a directory to run "
echo "   cmake -DCUDNN_ROOT=~/.jumbo/opt/cudnn/ YOUR_PADDLE_SOURCE_DIRECTORY/paddle/"
echo "Then make; make install DESTDIR=YOUR_INSTALL_DIR"
echo ""
echo "To run paddle gpu mode, you need cudnn v2 library and cuda library in your LD_LIBRARY_PATH "
echo "Enjoy yourself." 
echo ""
