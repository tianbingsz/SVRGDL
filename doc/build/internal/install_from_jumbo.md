# Install from jumbo

[Jumbo](http://jumbo.baidu.com/) is a package manager used in Baidu development
environment. The root privilege is not needed when using jumbo.
PaddlePaddle is also released by [jumbo](http://jumbo.baidu.com/) in baidu development
environment.

The following guidances are to install PaddlePaddle by jumbo.

## Install jumbo

To install jumbo, just run

    bash -c "$( curl http://jumbo.baidu.com/install_jumbo.sh )"; source ~/.bashrc

, then jumbo is installed.


NOTE: jumbo may crash due to problems caused by LD_LIBRARY_PATH.
LD_LIBRARY_PATH will be reset when using jumbo to install other packages, but PATH will not be reset.
So if your gcc in PATH depends on LD_LIBRARY_PATH, jumbo installation will crash.

To resolve this problem, you can just remove 'unset LD_LIBRARY_PATH' in
${JUMBO_ROOT}/bin/.jumbo.

## Add PaddlePaddle's Jumbo repo

PaddlePaddle uses a private jumbo repo. To add this repo for jumbo, just run

    jumbo add_repo http://m1-idl-gpu2-bak31.m1.baidu.com:8088/jumbo/alpha/

then, the PaddlePaddle repo is added.
Currently, PaddlePaddle in jumbo is in alpha state.

## Install PaddlePaddle

Just run

    jumbo install paddle

After PaddlePaddle  is compiled and installed, just run

    paddle

this dumps a helper message.

    usage: paddle [--help] [<args>]
    These are common paddle commands used in various situations:
        train             Start a paddle_trainer
        gen_answer        Start a paddle_gen_answer
        gen_sequence      Start a paddle_gen_sequence
        merge_model       Start a paddle_merge_model
        pserver           Start a paddle_pserver_main

    'paddle train --help', 'paddle gen_answer --help','paddle gen_sequence --help',
    'paddle merge_model --help', 'paddle pserver --help', list more detail usage for each command

Type `paddle train --help` will help you to run PaddlePaddle for training.
