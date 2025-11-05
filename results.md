**testing for remap and classification**
```
============< Transformation statistics >============

Scheduling mode = FIFO
Spark Context default degree of parallelism = 96

Aggregated Spark task metrics:
numTasks => 5876
successful tasks => 5876
speculative tasks => 0
taskDuration => 2899094 (48 min)
schedulerDelayTime => 40471 (40 s)
executorRunTime => 2790920 (47 min)
executorCpuTime => 30001 (30 s)
executorDeserializeTime => 67215 (1.1 min)
executorDeserializeCpuTime => 15623 (16 s)
resultSerializationTime => 488 (0.5 s)
jvmGCTime => 72917 (1.2 min)
shuffleFetchWaitTime => 0 (0 ms)
shuffleWriteTime => 2692 (3 s)
gettingResultTime => 0 (0 ms)
resultSize => 4465 (4.4 KB)
diskBytesSpilled => 0 (0 Bytes)
memoryBytesSpilled => 0 (0 Bytes)
peakExecutionMemory => 0
recordsRead => 100872
bytesRead => 0 (0 Bytes)
recordsWritten => 0
bytesWritten => 0 (0 Bytes)
shuffleRecordsRead => 5870
shuffleTotalBlocksFetched => 5870
shuffleLocalBlocksFetched => 1199
shuffleRemoteBlocksFetched => 4671
shuffleTotalBytesRead => 293500 (286.6 KB)
shuffleLocalBytesRead => 59950 (58.5 KB)
shuffleRemoteBytesRead => 233550 (228.1 KB)
shuffleRemoteBytesReadToDisk => 0 (0 Bytes)
shuffleBytesWritten => 293500 (286.6 KB)
shuffleRecordsWritten => 5870
```

**distribution of classes**
```
============< Classification Distribution >============
+--------------+--------------------+-----+-----+-----+
|Classification|Description         |Train|Test |Val  |
+--------------+--------------------+-----+-----+-----+
|1             |Unclassified        |0.56 |0.67 |0.53 |
|2             |Ground              |38.97|40.49|39.1 |
|3             |Vegetation          |56.98|54.09|56.93|
|4             |Building            |2.8  |3.34 |2.8  |
|5             |Water               |0.52 |1.2  |0.49 |
|6             |Bridge              |0.13 |0.16 |0.1  |
|7             |Permanent structures|0.04 |0.03 |0.04 |
|8             |Filtered/Artifacts  |0.01 |0.03 |0.01 |
+--------------+--------------------+-----+-----+-----+
```

**transformed features**
```
============< Transformation statistics >============

Scheduling mode = FIFO
Spark Context default degree of parallelism = 16

Aggregated Spark task metrics:
numTasks => 13
successful tasks => 13
speculative tasks => 0
taskDuration => 5597 (6 s)
schedulerDelayTime => 194 (0.2 s)
executorRunTime => 4224 (4 s)
executorCpuTime => 1653 (2 s)
executorDeserializeTime => 1167 (1 s)
executorDeserializeCpuTime => 977 (1.0 s)
resultSerializationTime => 12 (12 ms)
jvmGCTime => 128 (0.1 s)
shuffleFetchWaitTime => 0 (0 ms)
shuffleWriteTime => 14 (14 ms)
gettingResultTime => 0 (0 ms)
resultSize => 4508 (4.4 KB)
diskBytesSpilled => 0 (0 Bytes)
memoryBytesSpilled => 0 (0 Bytes)
peakExecutionMemory => 0
recordsRead => 243822
bytesRead => 7272813 (6.9 MB)
recordsWritten => 0
bytesWritten => 0 (0 Bytes)
shuffleRecordsRead => 4
shuffleTotalBlocksFetched => 4
shuffleLocalBlocksFetched => 3
shuffleRemoteBlocksFetched => 1
shuffleTotalBytesRead => 218 (218 Bytes)
shuffleLocalBytesRead => 165 (165 Bytes)
shuffleRemoteBytesRead => 53 (53 Bytes)
shuffleRemoteBytesReadToDisk => 0 (0 Bytes)
shuffleBytesWritten => 218 (218 Bytes)
shuffleRecordsWritten => 4
```

**sample of transformed features**
```
=== Train Sample ===
+----------+-----------+------------------+---------+--------------+-----+-----+-----+--------+-------------------+
|         x|          y|            z_norm|Intensity|Classification|  Red|Green| Blue|Infrared|               NDVI|
+----------+-----------+------------------+---------+--------------+-----+-----+-----+--------+-------------------+
|436849.274|6398112.945|2.4510000000000005|      460|             3|11008|12288|14592|    1280|-0.7916666666666666|
|436849.913|6398113.032|             2.649|     1441|             3|11520|12800|15616|    2304|-0.6666666666666666|
|436849.544|6398113.221|2.5870000000000006|     2208|             3|11008|12544|14592|    1536|-0.7551020408163265|
|436848.622|6398114.129|             1.469|      605|             3|11008|12544|15104|    1280|-0.7916666666666666|
|436848.254|6398114.275|1.5130000000000001|      780|             3|11008|12544|15104|    1280|-0.7916666666666666|
+----------+-----------+------------------+---------+--------------+-----+-----+-----+--------+-------------------+
only showing top 5 rows


=== Test Sample ===
+----------+------------------+------------------+---------+--------------+-----+-----+-----+--------+-------------------+
|         x|                 y|            z_norm|Intensity|Classification|  Red|Green| Blue|Infrared|               NDVI|
+----------+------------------+------------------+---------+--------------+-----+-----+-----+--------+-------------------+
|436399.755|       6383149.934| 1.093999999999994|     1647|             3|15104|17408|14336|   28928|  0.313953488372093|
|436399.707|       6383149.596|0.9439999999999884|     2079|             3|15616|18944|16384|   30464|0.32222222222222224|
|436399.662|       6383148.907|0.7979999999999876|     2424|             2|12032|14336|14336|   24576|0.34265734265734266|
|436399.707|6383148.5649999995|0.8799999999999955|     2017|             3|12800|14592|15104|   24064| 0.3055555555555556|
|436399.805|       6383148.229|  0.98599999999999|     1787|             3| 9984|10496|11520|   15616|               0.22|
+----------+------------------+------------------+---------+--------------+-----+-----+-----+--------+-------------------+
only showing top 5 rows


=== Val Sample ===
+----------+------------------+------------------+---------+--------------+-----+-----+-----+--------+--------------------+
|         x|                 y|            z_norm|Intensity|Classification|  Red|Green| Blue|Infrared|                NDVI|
+----------+------------------+------------------+---------+--------------+-----+-----+-----+--------+--------------------+
|436500.225|       6419250.006|12.254999999999995|      479|             4|54272|48384|42752|   30720|-0.27710843373493976|
|436500.173|6419250.6620000005|12.102999999999994|     2708|             4|52480|46080|40192|   26624| -0.3268608414239482|
| 436500.15|       6419250.999|            12.015|     3232|             4|52480|46080|40192|   25600| -0.3442622950819672|
| 436500.13|       6419251.349|11.906999999999996|     3289|             4|51200|44544|38656|   23808| -0.3651877133105802|
|436500.087|       6419251.673|11.876999999999995|     1801|             4|50688|44288|39168|   24832| -0.3423728813559322|
+----------+------------------+------------------+---------+--------------+-----+-----+-----+--------+--------------------+
only showing top 5 rows

```

**for saving files**
============< Transformation statistics >============

Scheduling mode = FIFO
Spark Context default degree of parallelism = 64

Aggregated Spark task metrics:
numTasks => 11746
successful tasks => 11746
speculative tasks => 0
taskDuration => 27714340 (7.7 h)
schedulerDelayTime => 46648 (47 s)
executorRunTime => 27536706 (7.6 h)
executorCpuTime => 18624121 (5.2 h)
executorDeserializeTime => 130653 (2.2 min)
executorDeserializeCpuTime => 83649 (1.4 min)
resultSerializationTime => 333 (0.3 s)
jvmGCTime => 399399 (6.7 min)
shuffleFetchWaitTime => 3 (3 ms)
shuffleWriteTime => 559 (0.6 s)
gettingResultTime => 0 (0 ms)
resultSize => 4508 (4.4 KB)
diskBytesSpilled => 0 (0 Bytes)
memoryBytesSpilled => 0 (0 Bytes)
peakExecutionMemory => 0
recordsRead => 18524936256
bytesRead => 230779129357 (214.9 GB)
recordsWritten => 9262468128
bytesWritten => 175168941247 (163.1 GB)
shuffleRecordsRead => 5870
shuffleTotalBlocksFetched => 5870
shuffleLocalBlocksFetched => 2786
shuffleRemoteBlocksFetched => 3084
shuffleTotalBytesRead => 322658 (315.1 KB)
shuffleLocalBytesRead => 153128 (149.5 KB)
shuffleRemoteBytesRead => 169530 (165.6 KB)
shuffleRemoteBytesReadToDisk => 0 (0 Bytes)
shuffleBytesWritten => 322658 (315.1 KB)
shuffleRecordsWritten => 5870

