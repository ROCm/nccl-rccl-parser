# RCCL Collective Trace Processor

This tool is used to collect RCCL collective traces, visulize topologies for rings and trees in RCCL, and get device grouping information. 

## Requirement 
The tool currently works for applications with RCCL 2.9 or above. However, the collective trace processor function works for an application without multiple device groups in RCCL 2.8 or below.

From ROCm 4.3:
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL only enable collective API trace. Collective trace mode is enabled separately by RCCL_KERNEL_COLL_TRACE_ENABLE=1 which has the outputs in the new format as below:
```
[0] NCCL INFO ## [1703255.821541] [01:00] 000035 KL HWID 4230c540 AllReduceTreeLLSum_f32 nt 256 bi 0 nc 1 busId C3000
```
**Run application and collect RCCL Log:**

```
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL,GRAPH RCCL_KERNEL_COLL_TRACE_ENABLE=1 <application> |& tee nccl_debug_log.txt
```

## Usage
For more information about RCCL collective traces, please go to [here](https://confluence.amd.com/display/MLSE/RCCL+Collective+Trace).

Collect RCCL logs for applications
```shell
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL RCCL_KERNEL_COLL_TRACE_ENABLE=1 <executable>
```
Example command lines:
```shell
python rccl_trace_processor.py --rccl-debug-log rccl_log.txt
```
Notice that since RCCL 2.8 or below has no sufficient inforamtion for device grouping, "--legacy" flag needs to be specified and the number of devices used in the application is also required.
```shell
python rccl_trace_processor.py --rccl-debug-log rccl_2.8.log --legacy --num_devices 8
```

## Example Output
If ROCm 2.8 or above is used, there will be multiple RCCL topology graphs, time tables for each RCCL operations and devices, bandwidth tables for each RCCL operations and devices, and a text file which contains device grouping information. </br>
For example, if there are 6 device groups in an application, there will be 12 (=6*2) output tables in csv files. The numbering of the tables is followed by the line number in device_groups.txt.

![image info](bw_0.png) 


## Copyright
All source code and accompanying documentation are copyright (c) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
