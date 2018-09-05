## Training on Cluster
### Upload Data to HDFS ###

Assume that [Data Preprocess](#data-preprocess) is finished, we need to split train data into several parts and upload them to HDFS. To do this, simply run the following commands:

```bash
cd demo/seqToseq/data
./hdfs_data.sh
```
NOTE THAT we have alreadly uploaded to the `/app/idl/idl-dl/paddle/demo/seqToseq`, and the train data parts should be more than number of cluster nodes. 

Run the hadoop command:

```bash
hadoop fs -ls /app/idl/idl-dl/paddle/demo/seqToseq 
``` 

You will see messages as follows:
```
drwxr-xr-x   3 paddle_demo paddle_demo          0 2016-07-12 19:31 /app/idl/idl-dl/paddle/demo/seqToseq/test
drwxr-xr-x   3 paddle_demo paddle_demo          0 2016-07-12 19:30 /app/idl/idl-dl/paddle/demo/seqToseq/train
```

### Config HDFS Arguments ###

We need to config hadoop cluster name, ugi, data root path in `train.conf` as follows:

```python
### for cluster training
cluster_config(
    fs_name = "hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310",
    fs_ugi = "paddle_demo,paddle_demo",
    work_dir ="/app/idl/idl-dl/paddle/demo/seqToseq/",
)
...
```

- fs_name: name of hadoop
- fs_ugi: hadoop user name and password
- work_dir: data root path

For more arguments in `cluster_config`, please refer to the Paddle Platform Document.

### Submit Job ###

Before submit job to cluster, we need to write `demo/seqToseq/translation/thirdparty/before_hook.sh`:

```bash
#this scripts will be executed before launching trainer
dir="thirdparty/thirdparty"
mv ./$dir/train.conf .
mv ./$dir/seqToseq_net.py .
mv ./$dir/dataprovider.py .
mv ./$dir/src.dict .
mv ./$dir/trg.dict .
```

Then, we can train the model on cluster by running the command:

```bash
cd demo/seqToseq/translation
./cluster.sh
```

The `cluster.sh` is shown as follows:
```bash
dir="thirdparty"
cp train.conf $dir/.
cp ../dataprovider.py $dir/.
cp ../seqToseq_net.py $dir/.
cp ../data/pre-wmt14/src.dict $dir/.
cp ../data/pre-wmt14/trg.dict $dir/.

paddle cluster_train \
  --config=train.conf \
  --config_args=is_cluster=true \
  --save_dir='output' \
  --use_gpu=cpu \
  --trainer_count=8 \
  --num_passes=16 \
  --log_period=10 \
  --thirdparty=./thirdparty \
  --num_nodes=2 \
  --job_priority=normal \
  --job_name=paddle_platform_translation_demo \
  --time_limit=00:30:00 \
  --submitter=luotao \
  --where=nmg01-idl-dl-cpu-10G_cluster \
```

Most arguments are same with local training except following:
- thirdparty: set directory which will be uploaded to each nodes of cluster
- num_nodes: number of cluster nodes
- job_priority: set job priority
- job_name: set job name 
- time_limit: set the maximize of elasped time
- submitter: set job submitter
