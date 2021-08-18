# Device ID Mapping

The device indices on HIP, ROCM-SMI, and RCCL (cudaDev) are not necessarily matching. The only unique information for devices is PCI bus ID. Therefore, this sub-project is meant to create a debugging tool across different software stacks. 



**On ROCM-SMI (--showbus):**
The following outputs are from this tool on a server with 8 MI60 GPUs.
```
======================= ROCm System Management Interface =======================
================================== PCI Bus ID ==================================
GPU[0]          : PCI Bus: 0000:43:00.0
GPU[1]          : PCI Bus: 0000:23:00.0
GPU[2]          : PCI Bus: 0000:26:00.0
GPU[3]          : PCI Bus: 0000:03:00.0
GPU[4]          : PCI Bus: 0000:E3:00.0
GPU[5]          : PCI Bus: 0000:C3:00.0
GPU[6]          : PCI Bus: 0000:C6:00.0
GPU[7]          : PCI Bus: 0000:83:00.0
================================================================================
============================= End of ROCm SMI Log ==============================
```


**Device ID Mapping**

The following outputs are from this tool on a server with 8 MI60 GPUs. We can use PCI bus id to get device id on HIP by using [hipDeviceGetByPCIBusId](https://rocmdocs.amd.com/en/latest/ROCm_API_References/HIP_API/Initialization-and-Version.html?highlight=hipDeviceGetByPCIBusId#hipdevicegetbypcibusid).

```
====================== Number of HIP visible devices: 8
====================== Number of GPUs on your machine which can be observed by ROCm-SMI: 8
====== ROCm-SMI device ID ======= PCI bus ID ======= HIP device ID ======
                0                0000:43:00.0              0
                1                0000:23:00.0              N/A (cannot map PCI Bus ID: 0000:23:00.0 to a HIP visible device)
                2                0000:26:00.0              N/A (cannot map PCI Bus ID: 0000:26:00.0 to a HIP visible device)
                3                0000:03:00.0              1
                4                0000:e3:00.0              N/A (cannot map PCI Bus ID: 0000:e3:00.0 to a HIP visible device)
                5                0000:c3:00.0              N/A (cannot map PCI Bus ID: 0000:c3:00.0 to a HIP visible device)
                6                0000:c6:00.0              N/A (cannot map PCI Bus ID: 0000:c6:00.0 to a HIP visible device)
                7                0000:83:00.0              2

```

As the results shown above, we confirmed that the device (enumeration) indices on HIP and ROCM-SMI are identical when there is no HIP_VISIBLE_DEVICES used.


When [HIP_VISIBLE_DEVICES](https://rocmdocs.amd.com/en/latest/Other_Solutions/Other-Solutions.html?highlight=HIP_VISIBLE_DEVICES#hip-environment-variables) or [ROCR_VISIBLE_DEVICES](https://rocmdocs.amd.com/en/latest/ROCm_System_Managment/ROCm-System-Managment.html?highlight=ROCR_VISIBLE_DEVICES#rocr-visible-devices) is used, the indices will not match for HIP and ROCM-SMI. The following outputs are from
```
HIP_VISIBLE_DEVICES=0,3,7 ./hip_rocm_smi_mapping
or
ROCR_VISIBLE_DEVICES=0,3,7 ./hip_rocm_smi_mapping
```
```
====================== Number of HIP visible devices: 3
====================== Number of GPUs on your machine which can be observed by ROCm-SMI: 8
=================== ROCm-SMI device ID =================== PCI bus ID =================== HIP device ID ===================
0      --->      0000:43:00.0      --->      0
1      --->      0000:23:00.0      --->      N/A (cannot map PCI Bus ID: 0000:23:00.0 to a HIP visible device)
2      --->      0000:26:00.0      --->      N/A (cannot map PCI Bus ID: 0000:26:00.0 to a HIP visible device)
3      --->      0000:03:00.0      --->      1
4      --->      0000:e3:00.0      --->      N/A (cannot map PCI Bus ID: 0000:e3:00.0 to a HIP visible device)
5      --->      0000:c3:00.0      --->      N/A (cannot map PCI Bus ID: 0000:c3:00.0 to a HIP visible device)
6      --->      0000:c6:00.0      --->      N/A (cannot map PCI Bus ID: 0000:c6:00.0 to a HIP visible device)
7      --->      0000:83:00.0      --->      2

```