## Training on clusters

PaddlePaddle is shipped with an easy to use tool to help you train models on
cluster, you can find it under
`${PADDLE_SOURCE_ROOT}/paddle/scripts/cluster_train`.

The tool is a python script which utilizes `fabric` to deploy files over your
cluster, so first you need to make sure fabric is proper installed, if not,
```bash
pip install fabric
```
The solution to train on a cluster is easy to understand, pack everything
that PaddlePaddle needs to train a model in a folder called `workspace`, then
deploy workspace over the machines you want to train on, PaddlePaddle will start
trainers and parameter servers on those machines, the trainers on different
machines will train the model in a distributed manner and save the complete
model on the master node's workspace folder.

Let's focus on the `workspace` first.

### Prepare Workspace

Here is a typical setting for workspace forder:
```common
workspace
├── conf
│   └── trainer_config.conf
├── test
│   ├── dnn_instance_000000
│   ├── dnn_instance_000001
├── test.list
├── train
│   ├── dnn_instance_000000
│   ├── dnn_instance_000001
└── train.list
```

But in this demo you don't need to bother to prepare a workspace, assume you have
successfully run the local train in this demo, which means data preparation is
done, the demo folder to run the local train can be used as a workspace.
One thing that should be noticed is that the cluster train tool doesn't split
data, so you should always split data by yourself. In this demo, to keep it
easy, we put same data on each node.


### Configure Your Cluster

In this section we'll show you how to configure information about the cluster to
run the training job, Just open
`${PADDLE_SOURCE_ROOT}/paddle/script/cluster_train/conf.py`,
fill the names or ips(with account name, for example, root@192.168.0.1)
of the machines in `HOSTS`, the first machine in the list will be treated as
the master node, PaddlePaddle will create a `output` folder under workspace of the machine
and store models in it, and paddle will create `log` folder under every node's
workspace to store log files.

Here is a sample conf.py:
```python

HOSTS = [
    "root@192.168.100.17",
    "root@192.168.100.18",
    ]

'''
workspace configuration
'''
#root dir for workspace, can be set as any director with real user account
ROOT_DIR = "/home/paddle"


'''
network configuration
'''
#pserver nics
PADDLE_NIC = "eth0"
#pserver port
PADDLE_PORT = 7164
#pserver ports num
PADDLE_PORTS_NUM = 2
#pserver sparse ports num
PADDLE_PORTS_NUM_FOR_SPARSE = 2
```
Before launching a cluster training job, go to
`${PADDLE_SOURCE_ROOT}/paddle/script/cluster_train`, open conf.py and:

1. Replace the content of `HOSTS` with the real machine list you want to train on.
   You should make sure that paddle is properly installed on these machines for
   the account.
2. Set a `ROOT_DIR`, on every machine in `HOSTS`, the content of the workspace
   will be copied to `ROOT_DIR`.
3. Set the NIC(Network Interface Card) interface name for cluster communication
   channel, such as eth0 for ethternet, ib0 for infiniband.

In this demo, just modify the 3 entries according to your situation and leave
the rest as default. For more information about the configurations for cluster
training, please refer to
[cluster train](../../cluster/opensource/cluster_train.html).

### Start Training

Except for the configurations in
`${PADDLE_SOURCE_ROOT}/paddle/script/cluster_train/conf.py` and
`${PADDLE_SOURCE_ROOT}/demo/image_classification/vgg_16_cifar.py`
There are still several more arguments to set, below is an example:

```bash
python paddle.py \
  --job_dispatch_package="${PATH_TO_LOCAL_WORKSPACE}" \
  --dot_period=10 \
  --ports_num_for_sparse=2 \
  --log_period=50 \
  --num_passes=10 \
  --trainer_count=4 \
  --saving_period=1 \
  --local=0 \
  --config=./trainer_config.py \
  --save_dir=./output \
  --use_gpu=0
```
In this tutorial, go to
`${PADDLE_SOURCE_ROOT}/paddle/script/cluster_train`, open run.sh and:
1. Update `--job_dispatch_package` to the absolute path of image classification
   demo.
2. Modify `--trainer_count`, `--num_passes`, `--saving_period` and `--use_gpu`
   to the values set in the
   `${ABSOLUTE_PATH_TO_IMAGE_CLASSIFICATION_DEMO}/train.sh`
3. Change `--config` to `vgg_16_cifar.py`.

Leave other arguments as default, and run the script.

After the training is done, you can find model under master node's
`${ROOT_DIR}/JOB${TimestampOfYourLaunching}/output`, and during the running of
the job, you can kill it by Ctrl-C on your submitting machine. For more
information please refer to
[clustr train](../../cluster/opensource/cluster_train.html).
