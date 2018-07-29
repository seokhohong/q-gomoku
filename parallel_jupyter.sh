#!/usr/bin/env bash

# //////////////////////////////////////////////////////////////////////////////
# ATTENTION
# //////////////////////////////////////////////////////////////////////////////
#
# This file is managed through git and is symlinked from a cloned repository
# on this machine. Type `ls -al` to view the source location of this file.
# Any changes made must be checked in to git otherwise they will be overwritten.
# https://wwwin-github.cisco.com/GVS-CS-DSX/server_config
#
# //////////////////////////////////////////////////////////////////////////////

user=`who -m | awk '{print $1;}'`
executors="$1"
cores=1
# arbitrary minimum of a quarter of executors just to give faster starts
min_executors="1"
min_cores="$(($min_executors * $cores))"
max_cores="$(($executors * $cores))"
max_memory="$(($executors * 4))"

name="PySpark ("$user") Max Cores: "$max_cores" / Max Memory: "$max_memory"g"
export MAPR_HOME=/opt/mapr
export MAPR_MAPREDUCE_MODE=yarn
export MASTER=yarn-client
export ANACONDA_HOME=/users/hdpgcsana/anaconda3
export SPARK_HOME=/opt/mapr/spark/spark-2.1.0
export PATH=$ANACONDA_HOME/bin:$SPARK_HOME/bin:$PATH
export PYSPARK_PYTHON=/users/hdpgcsana/anaconda3/bin/python
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS=notebook
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/build:/hdfs/app/GCS_ANA/Analytics/lib/src:/hdfs/app/GCS_ANA/Analytics/admin/python/graphframes/python

# 14540m is the maximum given the 16gb limit

pyspark \
  --master yarn \
  --deploy-mode client \
  --driver-memory 14G \
  --executor-memory 14545m \
  --conf "spark.dynamicAllocation.enabled=true" \
  --conf "spark.executor.instances="$min_executors \
  --conf "spark.dynamicAllocation.maxExecutors="$executors \
  --conf "spark.executor.memory=4g" \
  --name "$name" \
  --files /opt/mapr/spark/spark-2.1.0/conf/hive-site.xml  \
  --executor-cores $cores \
  --properties-file ~/bin/jupyter.conf \

