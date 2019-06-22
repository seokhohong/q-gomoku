export ANACONDA_PATH='/opt/conda/anaconda'
export PYSPARK_DRIVER_PYTHON=${ANACONDA_PATH}/bin/jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook"
export PYSPARK_PYTHON=${ANACONDA_PATH}/bin/python3
pyspark --name "Pyspark Notebook" --num-executors 1 --conf spark.executor.memory='7g' --conf spark.dynamicAllocation.enabled=true \
--conf spark.shuffle.service.enabled=true --conf spark.dynamicAllocation.minExecutors=1 --conf spark.dynamicAllocation.maxExecutors=100 \
--conf spark.dynamicAllocation.executorIdleTimeout='300s' --conf spark.executor.cores=2 --conf spark.dynamicAllocation.initExecutors=1 \
--conf spark.sql.shuffle.partitions=10 --conf spark.kryoserializer.buffer.max=1g --conf spark.driver.memory='16g' \
--conf spark.rpc.message.maxSize='2047' --jars=gs://hadoop-lib/bigquery/bigquery-connector-hadoop2-latest.jar --py-files ~/q-gomoku/src.zip
