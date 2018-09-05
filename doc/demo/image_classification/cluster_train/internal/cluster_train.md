## Training using clusters
After making sure the training is running successfully locally, you can run it on clusters if you want faster training time.
For installing and configuring PADDLE remote training tools, please refer to the documentaion at .

First, you need to upload your data to HDFS by running `upload_hadoop.sh`.
The following scripts upload the files to the corresponding directory at HDFS.

```bash
hadoop fs -Dhadoop.job.ugi=paddle_demo,paddle_demo -put data/cifar-out/batches/train_batch_* /app/idl/idl-dl/paddle/demo/image_classification/train/
hadoop fs -Dhadoop.job.ugi=paddle_demo,paddle_demo -put data/cifar-out/batches/test_batch_* /app/idl/idl-dl/paddle/demo/image_classification/test/
hadoop fs -Dhadoop.job.ugi=paddle_demo,paddle_demo -put data/cifar-out/batches/batches.meta /app/idl/idl-dl/paddle/demo/image_classification/train_meta
hadoop fs -Dhadoop.job.ugi=paddle_demo,paddle_demo -put data/cifar-out/batches/batches.meta /app/idl/idl-dl/paddle/demo/image_classification/test_meta
```

Then, the cluster configuration needs to be added into the configuration file.
It specifies the HDFS path of the training/testing data, and training/testing meta files.

```python
cluster_config(
        fs_name = "hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310",
        fs_ugi = "paddle_demo,paddle_demo",
        work_dir ="/app/idl/idl-dl/paddle/demo/image_classification/",
        has_meta_data = True
)
```

The data configuration needs to be changed to the following to reflect the data structure on
the cluster:
```python
data_dir='data/cifar-out/batches/' if is_local else "./"
meta_path=data_dir+'batches.meta'
if not is_local:
  meta_path="train_data_dir/image_classification/train_meta"
```

Then, you can run `cluster.sh` to submit job. This script prepares the related dependencies first.
It copies all the python and shared library dependencies into this directory.

```bash
mkdir thirdparty
cp -r ~/.jumbo/lib/python2.7/site-packages/PIL/ ./thirdparty/
cp -r ~/.jumbo/lib/libjpeg.so.9  ./thirdparty/
```


Finally, your can use submit script to script a cluster job.

```bash
SCRIPT_PATH=$PWD
export PYTHONPATH=$PYTHONPATH:$PWD:../../python/

paddle cluster_train \
  --config $SCRIPT_PATH/vgg_16_cifar.cluster.py \
  --use_gpu cpu\
  --time_limit 00:30:00 \
  --submitter wangjiang03 \
  --num_nodes 2 \
  --job_priority normal \
  --trainer_count 4 \
  --num_passes 1 \
  --log_period 1000 \
  --dot_period 100 \
  --saving_period 1 \
  --where nmg01-idl-dl-cpu-10G_cluster \
  --job_name image_classification_cifar \
  --thirdparty $SCRIPT_PATH/thirdparty
```

More details on the meaning of the command line arguments can be found in cluster training documentaion.
The approaches on how to see the training log or errors can also be found in this documentation.

