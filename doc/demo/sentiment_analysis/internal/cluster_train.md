## Cluster Training

### Upload Data to HDFS

All following operations are in `demo/sentiment` directory, which is also the local experiments directory. Assume that you have processed the data.

```
cd demo/sentiment
```

At first, you need to split train data into several parts and upload them to HDFS. In order to avoid more users to operate same directory at the same time, we have uploaded data to `/app/idl/idl-dl/paddle/demo/sentiment` by hdfs_data.sh.

NOTE: the train data parts should be more than number of cluster nodes. 

``` bash
hadoop fs -ls /app/idl/idl-dl/paddle/demo/sentiment
```

you will see that data path in HDFS is as follows:

```
drwxr-xr-x   3 paddle_demo paddle_demo          0 2016-07-13 14:08 /app/idl/idl-dl/paddle/demo/sentiment/test
drwxr-xr-x   3 paddle_demo paddle_demo          0 2016-07-13 14:07 /app/idl/idl-dl/paddle/demo/sentiment/train
```


### Config HDFS Arguments

You need to config hadoop cluster name, ugi, data root path in `trainer_config.py` as follows:

```
# for cluster training
cluster_config(
        fs_name = "hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310",
        fs_ugi = "paddle_demo,paddle_demo",
        work_dir ="/app/idl/idl-dl/paddle/demo/sentiment/",
        )

from sentiment_net import *

# whether this config is used for local training
is_cluster = get_config_arg('is_cluster', bool, False)
# whether this config is used for test
is_test = get_config_arg('is_test', bool, False)
# whether this config is used for prediction
is_predict = get_config_arg('is_predict', bool, False)

data_dir  = "./data/pre-imdb" if not is_cluster else "./"
...         
```
mainly arguments in cluster_config is as follows. For more detailed information, you can refer to the document of Paddle platform.

* fs_name: name of hadoop.
* fs_ugi: hadoop user name and password.
* work_dir: data root path.

### Submit Job

```
./cluster.sh
```

cluster.sh is as follows:

```
dir=thirdparty
cp dataprovider.py $dir/.
cp sentiment_net.py $dir/.
cp data/pre-imdb/dict.txt $dir/.
cp data/pre-imdb/labels.list $dir/.

paddle cluster_train \
    --config=trainer_config.py \
    --use_gpu=cpu \
    --trainer_count=4 \
    --num_passes=10 \
    --log_period=20 \
    --config_args=is_cluster=1 \
    --thirdparty=./thirdparty \
    --num_nodes=2 \
    --job_priority=normal \
    --job_name=paddle_platform_sentiment_demo \
    --time_limit=01:00:00 \
    --submitter=dangqingqing \
    --where=nmg01-idl-dl-cpu-10G_cluster \
```

And you need to write `thirdparty/before_hook.sh` as follows.

```
#this scripts will be executed before launching trainer
dir=thirdparty/thirdparty
mv ./$dir/dataprovider.py .
mv ./$dir/dict.txt .
mv ./$dir/labels.list .
```

Most arguments are same with local training except serveral arguments.

* num_nodes: number of cluster nodes.
* job_priority: specify job priority.
* where: specify cluster name.
* thirdparty: specify directory which will be uploaded to each nodes of cluster.

After submitting successfully you may see the job url as follows.

```
...
[INFO] qsub_f: to stop, pls run: qdel 127993.nmg01-hpc-imaster01.nmg01.baidu.com
[INFO] If you want to get status of the job, pls look at the page: http://nmg01-hpc-controller.nmg01.baidu.com:8090/job/i-127993/
```

And you can view the job log by this url. The mainly logs are in the first node, as follows in cluster `nmg01-idl-dl-cpu-10G_cluster`:

```
workspace
|----log
|    |----paddle_trainer.INFO
|    |----train.log
```

