#!/bin/sh
# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


script_dir=$(cd `dirname $0` && pwd)
VERSION="0.8.0b0"

rm $script_dir/.tmp -fr &&
mkdir -p $script_dir/.tmp &&
cp $script_dir/paddle $script_dir/.tmp/

MD5=($(md5sum $script_dir/.tmp/paddle))

m4 -P -DVERSION=${VERSION} -DMD5=${MD5}  $script_dir/paddle_main.jumbo.m4 > $script_dir/paddle_main.jumbo

echo "******************************************************"
echo "*****TODO: upload data jumbo server mannually*********"
echo "1. upload project file to jumbo server"
echo "   scp $script_dir/.tmp/paddle \
www@m1-idl-gpu2-bak31.m1:~/var/jumbo/alpha/packages/paddle_main/"
echo "2. upload jumbo file to jumbo server"
echo "   scp paddle_main.jumbo \
www@m1-idl-gpu2-bak31.m1:~/var/jumbo/alpha/installs/"
echo "3. update jumbo list "
echo "   ssh www@m1-idl-gpu2-bak31.m1 \"cd var/jumbo/alpha/installs; tar czvf list.tar.gz *.jumbo; mv list.tar.gz ..\""
echo "******************************************************"
