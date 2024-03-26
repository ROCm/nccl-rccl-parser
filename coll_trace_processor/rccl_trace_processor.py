###############################################################################
#
# MIT License
#
# Copyright (c) 2022 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###############################################################################
# Please contact Hubert Lu (hubertlu@amd.com) if you have any questions.

import os
import sys
import subprocess
import argparse
import pkg_resources
import re
import time
import numpy as np

coll_op_map = {
            "Broadcast": "broadcast_perf",
            "Reduce": "reduce_perf",
            "AllGather": "all_gather_perf",
            "ReduceScatter": "reduce_scatter_perf",
            "AllReduce": "all_reduce_perf",
            "Gather": "gather_perf",
            "Scatter": "scatter_perf",
            "AllToAll": "alltoall_perf",
#            "AllToAllv": "alltoallv_perf",
            "Send": "sendrecv_perf",
            "Recv": "sendrecv_perf",
            }

reduction_op_map = {
                "0" : "sum",
                "1" : "prod",
                "2" : "max",
                "3" : "min",
                "4" : "all",
               }

data_types_map = {
                "0" : "int8",
                "1" : "uint8",
                "2" : "int32",
                "3" : "uint32",
                "4" : "int64",
                "5" : "uint64",
                "6" : "half",
                "7" : "float",
                "8" : "double",
                "9" : "bf16",
                "10" : "ncclNumTypes"
             }

data_type_bytes_map = {
                    "0" : 1,
                    "1" : 1,
                    "2" : 4,
                    "3" : 4,
                    "4" : 8,
                    "5" : 8,
                    "6" : 2,
                    "7" : 4,
                    "8" : 8,
                    "9" : 2,
                    #"10" : Not sure.
                  }

# n: number of ranks 
# n links of Bandwidth B to perform a operation
# def factor_1(n): 
#     return n
def all_gather_factor(n):
     return (n-1)/n
def reduce_scatter_factor(n):
     return (n-1)/n        
def all_reduce_factor(n):
     return 2*(n-1)/n
def all_to_all_factor(n):
    return 2*(n-1)/n

def coll_op_algobw_factor(coll_type, nranks):
    nranks = float(nranks)
    if coll_type == "AllGather":
        return float(all_gather_factor(nranks))
    elif coll_type == "ReduceScatter":
        return float(reduce_scatter_factor(nranks))
    elif coll_type == "AllReduce":
        return float(all_reduce_factor(nranks))
    elif coll_type == "AllToAll":
        return float(all_to_all_factor(nranks))
    else:
        return float(1)
    
def algobw_factor_times_size(coll_type, nranks, total_bytes):
    nranks = float(nranks)
    if coll_type == "AllGather":
        return all_gather_factor(nranks) * float(total_bytes)
    elif coll_type == "ReduceScatter":
        return reduce_scatter_factor(nranks) * float(total_bytes)
    elif coll_type == "AllReduce":
        return all_reduce_factor(nranks) * float(total_bytes)
    elif coll_type == "AllToAll":
        return all_to_all_factor(nranks) * float(total_bytes)
    else:
        return float(1) * float(total_bytes)   

class DisjointSet(object): # https://stackoverflow.com/questions/3067529/a-set-union-find-algorithm
    def __init__(self):
        self.leader = {} # maps a member to the group's leader
        self.group = {} # maps a group leader to the group (which is a set)

    def add(self, a, b):
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return # nothing to do
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])

                
def get_useful_info(log_file):
    fs = open(log_file, 'r')
    lines = fs.readlines()
    fs.close()

    coll_lines, conn_lines, comm_lines, ring_lines, tree_lines, coll_trace_lines = [], [], [], [], [], []
    for j in range(len(lines)):
        line = lines[j].rstrip()
        if ("opCount" in line and "sendbuff" in line):
            coll_lines.append(line)
        elif ("Channel" in line and "via" in line):
            conn_lines.append(line)
        elif ("Init COMPLETE" in line and "busId" in line):
            comm_lines.append(line)
        elif ("NCCL INFO Ring" in line):
            ring_lines.append(line)
        elif ("NCCL INFO Trees" in line):
            tree_lines.append(line)
        elif ((" ## " in line) and ("KL HWID" in line or "KE" in line or "CE" in line)):  # RCCL From ROCm 4.3
            # Bug: [ 6628.064978] we need to consider the case when there is a spac right after '['
            #                     Everything with split_list[index] will break.
            if "[ " in line:
                line = line.replace("[ ", "[")
            coll_trace_lines.append(line)
            
    return coll_lines, conn_lines, comm_lines, ring_lines, tree_lines, coll_trace_lines

# ixt-sjc2-101:116274:116274 [1] NCCL INFO Broadcast: opCount 49cd sendbuff 0x7f10ce1c0000 recvbuff 0x7f10ce1c0000 count 12582912 datatype 0 op 0 root 0 comm 0x7f1640000eb0 [nranks=4] stream 0x55e2df89c270 devRedOp 0 isPtr 0 scaler 0
def coll_table_build(coll_lines):
    opCount, coll, count, datatype, op_type, root, comm, nranks, data_size = [], [], [], [], [], [], [], [], []
    for line in coll_lines:
        split_list = line.split(" ")
        coll.append(split_list[split_list.index("opCount") - 1][:-1])
        # Store opCount in string
        opCount.append(split_list[split_list.index("opCount") + 1])
        count.append(split_list[split_list.index("count") + 1])
        datatype.append(split_list[split_list.index("datatype") + 1])
        op_type.append(split_list[split_list.index("op") + 1])
        root.append(split_list[split_list.index("root") + 1])
        comm.append(split_list[split_list.index("comm") + 1])
        nranks.append(int(next(item for item in split_list if 'nranks' in item).split("=")[1].replace("]", ""))) ### 
        data_size.append(int(split_list[split_list.index("count") + 1]) * data_type_bytes_map[split_list[split_list.index("datatype") + 1]])
        
    dict_coll = {'coll': coll, 'opCount': opCount, 'datatype': datatype, 'count':count, 'op_type':op_type, 'root':root, 'comm':comm, 'nranks':nranks, 'data_size':data_size}
    table = pd.DataFrame(dict_coll)    
    table['algobw_factor_times_size'] = table.apply(lambda row: 
                                                    algobw_factor_times_size(row['coll'], row['nranks'], row['data_size']), axis=1)
    table = table[['opCount','nranks','algobw_factor_times_size', 'data_size']].drop_duplicates(subset = ['opCount', 'data_size'])
    # table.sort_values(by=['opCount'])
    return table

## It works for connection information like XXX via "P2P/direct pointer%s".(useReadStr),       --> pointer%s will not be collected
##                                                  "P2P/IPC%s".(useReadStr),  
##                                                  "P2P/indirect/%d[%lx]%s".(intermediateRank,comm->peerInfo[intermediateRank].busId, useReadStr) 
##                                                  "direct shared memory"                     --> shared memory
## However, it will not capture the information for CollNet for now.

## xxx [4] NCCL INFO Channel 00 : 0[e3000] -> 1[c3000] via P2P/IPC comm 0x7f53bc000e50 nRanks 04',  #(new output)

def conn_table_build(conn_lines):  # Only works for RCCL 2.9 or above
    def process_string(line):
        split_list = line.split("[")
        return [split_list[0], split_list[1].split("]")[0]]

    start_rank, start_busid, end_rank, end_busid, connection, comm, nranks = [], [], [], [], [], [], []
    for line in conn_lines:
        split_list = line.split(" ")
        sr, sb = process_string(split_list[split_list.index(":") + 1])  # first device
        er, eb = process_string(split_list[split_list.index("->") + 1]) # second device
        start_rank.append(sr)
        start_busid.append(sb)
        end_rank.append(er)
        end_busid.append(eb)
        connection.append(split_list[split_list.index("via") + 1])  # if it is direct, it means the connection is done by direct shared memory
        comm.append(split_list[split_list.index("comm") + 1])
        nranks.append(int(split_list[split_list.index("nRanks") + 1]))

    dict_conn = {'start_rank': start_rank, 'start_busid': start_busid, 'end_rank': end_rank, 'end_busid': end_busid, 
                 'connection': connection, 'comm':comm, 'nranks':nranks, 'conn_line':conn_lines}      
    return pd.DataFrame(dict_conn)

# ixt-sjc2-101:116274:116310 [1] NCCL INFO comm 0x7f1640000eb0 rank 1 nranks 4 cudaDev 1 busId 43000 used 35728 bytes - Init COMPLETE
def comm_table_build(comm_lines):
    comm, rank, nranks, cudaDev, busId = [], [], [], [], []   
    for line in comm_lines:
        split_list = line.rstrip().split(" ")
        comm.append(split_list[split_list.index("comm") + 1])
        rank.append(split_list[split_list.index("rank") + 1])
        nranks.append(int(split_list[split_list.index("nranks") + 1]))
        cudaDev.append(split_list[split_list.index("cudaDev") + 1])
        busId.append(split_list[split_list.index("busId") + 1])
    dict_comm = {'comm':comm, 'rank':rank, 'nranks':nranks, 'cudaDev':cudaDev, 'busId':busId}  
    return pd.DataFrame(dict_comm)


def topo_table_build(topo_lines):
    comm, nranks, busId, topo_line = [], [], [], []
    for line in topo_lines:
        split_list = line.split(" ")
        comm.append(split_list[split_list.index("comm") + 1])
        nranks.append(int(split_list[split_list.index("nRanks") + 1]))
        busId.append(split_list[split_list.index("busId") + 1])
        topo_line.append(line)
    dict_table = {'comm':comm, 'nranks':nranks, 'busId':busId, 'topo_line':topo_line}      
    return pd.DataFrame(dict_table)

def create_table(log_name):
    log_file = os.path.abspath(log_name)
    coll_lines, conn_lines, comm_lines, ring_lines, tree_lines, coll_trace_lines = get_useful_info(log_file)
    return coll_table_build(coll_lines), conn_table_build(conn_lines), comm_table_build(comm_lines), topo_table_build(ring_lines), topo_table_build(tree_lines), coll_trace_lines


def device_grouping(comm_table, conn_table):
    groups = []
    for index, row in comm_table.iterrows():
        temp = [row['busId'], list(conn_table[(conn_table['comm'] == row['comm']) & (conn_table['start_busid'] == row['busId'])]['end_busid'].unique())]
        groups.append(temp)    
    nranks = list(comm_table['nranks'])
    outputs = []
    rank_outputs = []
    tempRank = None
    for id, group in enumerate(groups): 
        if tempRank == None:
            tempRank = nranks[id]
            ds = DisjointSet()
        else:
            if tempRank != nranks[id]:
                for _, v in ds.group.items():
                    if v not in outputs: 
                        outputs.append(v)
                ds = DisjointSet()
                tempRank = nranks[id]
        for node in group[1]:
            ds.add(group[0], node)

        if id == len(groups) - 1: 
            for _, v in ds.group.items():
                if v not in outputs: 
                    outputs.append(v)  
    return outputs

def collect_topo(comm, nranks, ring_table, tree_table, conn_table, lines):
    ring_lines = ring_table[(ring_table['comm'] == comm) & (ring_table['nranks'] == nranks)]['topo_line'].tolist()
    tree_lines = tree_table[(tree_table['comm'] == comm) & (tree_table['nranks'] == nranks)]['topo_line'].tolist()
    conn_lines = conn_table[(conn_table['comm'] == comm) & (conn_table['nranks'] == nranks)]['conn_line'].tolist()
    for line in (ring_lines + tree_lines + conn_lines):
        lines.append(line)
    return lines

def modify_label(in_txt, out_txt, rank_busId_mapping):
    fs = open(in_txt, 'r')
    lines = fs.readlines()
    fs.close()
    with open(out_txt, 'w') as f:
        for line in lines:
            if "fontsize" in line: 
                for m in re.finditer('"([^"]*)"', line):
                    rank_quot = line[m.start(0):m.end(0)]
                    rank = line[m.start(0)+1:m.end(0)-1] # find the rank with no quotation
                    line = line.replace(rank_quot,'\"'+rank_busId_mapping[rank]+'\"', 1)
                    f.write("{}".format(line))
                    break
            else:
                f.write("{}".format(line)) 

# ixt-sjc2-101:116276:116309 [0] NCCL INFO ## [488201.944851] [03:00] 000001 KL HWID 433025b0 BroadcastRingSimpleSum_i8 nt 256 bi 0 nc 4 busId 3000 nRanks 4
# ixt-sjc2-101:116276:116309 [0] NCCL INFO ## [488201.974056] [03:02] 000001 KE busId 3000 nRanks 4
def coll_trace_processor(coll_trace_lines, group_list, coll_table):
    print(group_list)
    def process_string(line):
        split_list = line.split("[")
        return split_list[1].split("]")[0]
    # First pass
    results, func_name = {}, {}
    busIds = []
    RCCL_2_8 = False
    opCount_list = set(coll_table['opCount'].tolist())
    for line in coll_trace_lines:
        if ((" ## " in line) and ("KL HWID" in line or "KE" in line or "CE" in line)):  # RCCL From ROCm 4.3
            split_list = line.split(" ")
            
            if "KL HWID" in line:
                op = str(split_list[split_list.index("KL") - 1].lstrip('0'))
                if op == "":
                    op = "0"
            elif "KE" in line:
                op = str(split_list[split_list.index("KE") - 1].lstrip('0'))
                if op == "":
                    op = "0"
            else:
                op = str(split_list[split_list.index("CE") - 1].lstrip('0'))
                if op == "":
                    op = "0"

            if op not in opCount_list:
                continue
            ####### Kernel Launch #######
            if 'KL' in line:
                t = np.float64(process_string(split_list[split_list.index("KL") - 3])) # seconds.microseconds
                rk, blk = process_string(split_list[split_list.index("KL") - 2]).split(":") # rank:block_id
                if "busId" in split_list:
                    busId = split_list[split_list.index("busId") + 1]
                    nranks = split_list[split_list.index("nRanks") + 1]
                    KL_key = str(op) + "," + str(busId) + "," + str(nranks) + ",t0"
                else:
                    # RCCL 2.8 or older
                    busId = int(rk)
                    RCCL_2_8 = True
                    KL_key = str(op) + "," + str(busId) + ",t0"

                if KL_key not in results or results[KL_key] > t:
                    results[KL_key] = t
                if busId not in busIds:
                     busIds.append(busId)
                if op not in func_name:
                     func_name[op] = split_list[split_list.index("KL") + 3] # only work for KL
                        
            ####### Kernel End #######
            elif 'KE' in line:
                t = np.float64(process_string(split_list[split_list.index("KE") - 3])) # seconds.microseconds
                rk, blk = process_string(split_list[split_list.index("KE") - 2]).split(":") # rank:block_id
                if "busId" in split_list:
                    busId = split_list[split_list.index("busId") + 1]
                    nranks = split_list[split_list.index("nRanks") + 1]
                    KE_key = str(op) + "," + str(busId) + "," + str(nranks) + ",t1"
                else:
                    # RCCL 2.8 or below
                    busId = int(rk)
                    RCCL_2_8 = True
                    KE_key = str(op) + "," + str(busId) + ",t1"
                if KE_key not in results or results[KE_key] < t:
                    results[KE_key] = t
                    
                    
            elif 'CE' in line:
                t = np.float64(process_string(split_list[split_list.index("CE") - 3])) # seconds.microseconds
                rk, blk = process_string(split_list[split_list.index("CE") - 2]).split(":") # rank:block_id
                if "busId" in split_list:
                    busId = split_list[split_list.index("busId") + 1]
                    nranks = split_list[split_list.index("nRanks") + 1]
                    CE_key = str(op) + "," + str(busId) + "," + str(nranks) + ",t1"
                else:
                    # RCCL 2.8 or older
                    busId = int(rk)
                    RCCL_2_8 = True
                    CE_key = str(op) + "," + str(busId) + ",t1"
                    
                if CE_key not in results or results[CE_key] < t:
                    results[CE_key] = t

        elif ((" ## " in line) and ("Abort" in line)):
            raise AssertionError("Abort")
    
    # Second pass
    bw_tables, time_tables = [], []
    for i, group in enumerate(group_list):
        group = list(group)
        bw_list, time_list = [], []
        for op in func_name:
            found = False
            time_start, time_end = np.float64(1e9), np.float64(0)

            if RCCL_2_8:
                for busId in range(len(group)):
                    key = str(op) + "," + str(busId)
                    if (key + ",t0") in results:
                        found = True
                        if busId == 0:
                            temp = [op, func_name[op]] # opCount, function name
                        t_start = results[key + ",t0"]
                        t_end = results[key + ",t1"]
                        temp.append(t_end - t_start) 
                        if t_start < time_start:
                            time_start = t_start
                        if t_end > time_end:
                            time_end = t_end
            else:
                for busId in group:
                    key = str(op) + "," + str(busId) + "," + str(len(group))
                    if (key + ",t0") in results:
                        found = True
                        if busId == group[0]:
                            temp = [op, func_name[op]] # opCount, function name
                        t_start = results[key + ",t0"]
                        t_end = results[key + ",t1"]
                        temp.append(t_end - t_start)
                        if t_start < time_start:
                            time_start = t_start
                        if t_end > time_end:
                            time_end = t_end
            if found:
                # In PyTorch we observed that there are some synchronization used, so may get two lines with same opCount, comm, and datatype, but with different count
                # [0] NCCL INFO AllReduce: opCount 0 sendbuff 0x7f3e5b000000 recvbuff 0x7f3e5b000000 count 1 datatype 7 op 0 root 0 comm 0x7f3d34000eb0 ...
                # [0] NCCL INFO AllReduce: opCount 0 sendbuff 0x7f3e5b000000 recvbuff 0x7f3e5b000000 count 1131072 datatype 7 op 0 root 0 comm 0x7f3d34000eb0 ...
                # Therefore, we can just take the max over them.
                algobw_factor_times_size = max(coll_table[(coll_table['opCount'] == op) & (coll_table['nranks'] == len(group))]['algobw_factor_times_size'].unique())
                data_size = max(coll_table[(coll_table['opCount'] == op) & (coll_table['nranks'] == len(group))]['data_size'].unique())
                bw = temp[:2]

                for k in range(len(group)):
                    bw.append(algobw_factor_times_size / temp[k + 2] /1e9)    ## IndexError: list index out of range
                
                # time_end
                for busId in group:
                    time_end
                bw = bw + [data_size, data_size/(time_end - time_start)/1e9]
                
                # temp = [opCount, Function Name, GPU0, GPU1, ...,]
                busbw_per_op_worst = algobw_factor_times_size/max(temp[2:2+len(group)])/1e9 
                busbw_per_op_best = algobw_factor_times_size/min(temp[2:2+len(group)])/1e9
                temp.append(busbw_per_op_worst)
                temp.append(busbw_per_op_best)
                temp.append(data_size)
                time_list.append(temp)
                bw_list.append(bw)
                
        time_table = pd.DataFrame(time_list, columns = ['opCount', 'Function Name'] + group + ['busBW_worst'] + ['busBW_best'] + ['datasize']).sort_values(by=['opCount'])
        bw_table = pd.DataFrame(bw_list, columns = ['opCount', 'Function Name'] + group + ['data_size', 'algBW']).sort_values(by=['opCount'])
        time_tables.append(time_table)
        bw_tables.append(bw_table)
        
    return time_tables, bw_tables

def rccl_log_process():
    debug_log = os.path.abspath(args.rccl_debug_log)
    if args.legacy:
        coll_lines, conn_lines, comm_lines, ring_lines, tree_lines, coll_trace_lines = get_useful_info(debug_log)
        coll_table = coll_table_build(coll_lines)
        device_group_list = [[i for i in range(args.num_devices)]]
        print("Using ROCm with RCCL 2.8 or below.")
    else:
        device_grouping_output = os.path.join(os.path.dirname(os.path.realpath(__file__)), "device_groups.txt")
        coll_table, conn_table, comm_table, ring_table, tree_table, coll_trace_lines = create_table(debug_log)
        assert len(coll_trace_lines) != 0, "Please run your application with RCCL with the environment variable of \"RCCL_KERNEL_COLL_TRACE_ENABLE=1\""
        device_group_list = device_grouping(comm_table, conn_table)
        with open(device_grouping_output, 'w') as f:
            for mySet in device_group_list:
                f.write("%s\n" % str(mySet))    
    
    ########### Collective trace processor ###########
    time_tables, bw_tables = coll_trace_processor(coll_trace_lines, device_group_list, coll_table)
    for i, (time_table, bw_table) in enumerate(zip(time_tables, bw_tables)):
        time_csv = os.path.join(os.path.dirname(os.path.realpath(__file__)), "time_{}.csv".format(i))
        bw_csv = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bw_{}.csv".format(i))
        time_table.to_csv(time_csv)
        bw_table.to_csv(bw_csv)


def rccl_log_process():
    debug_log = os.path.abspath(args.rccl_debug_log)
    if args.legacy:
        coll_lines, conn_lines, comm_lines, ring_lines, tree_lines, coll_trace_lines = get_useful_info(debug_log)
        coll_table = coll_table_build(coll_lines)
        device_group_list = [[i for i in range(args.num_devices)]]
        print("Using ROCm with RCCL 2.8 or below.")
    else:
        device_grouping_output = os.path.join(os.path.dirname(os.path.realpath(__file__)), "device_groups.txt")
        coll_table, conn_table, comm_table, ring_table, tree_table, coll_trace_lines = create_table(debug_log)
        assert len(coll_trace_lines) != 0, "Please run your application with RCCL with the environment variable of \"RCCL_KERNEL_COLL_TRACE_ENABLE=1\""
        device_group_list = device_grouping(comm_table, conn_table)
        with open(device_grouping_output, 'w') as f:
            for mySet in device_group_list:
                f.write("%s\n" % str(mySet))    
    
    ########### Collective trace processor ###########
    time_tables, bw_tables = coll_trace_processor(coll_trace_lines, device_group_list, coll_table)
    for i, (time_table, bw_table) in enumerate(zip(time_tables, bw_tables)):
        time_csv = os.path.join(os.path.dirname(os.path.realpath(__file__)), "time_{}.csv".format(i))
        bw_csv = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bw_{}.csv".format(i))
        time_table.to_csv(time_csv, float_format='%.6f')
        bw_table.to_csv(bw_csv, float_format='%.6f')
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rccl-debug-log", type=str, required=True, \
                            help="RCCL log after running app with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL RCCL_KERNEL_COLL_TRACE_ENABLE=1 executable")
    parser.add_argument("--legacy", action="store_true", default=False, help="If the application is using ROCm systems with RCCL 2.8 or below, the topology visualizer will not be enabled.")
    parser.add_argument('--num_devices', type=int, default=1, help="If the application is using ROCm systems with RCCL 2.8 or below, the number of devices needs to be specified.")
    args = parser.parse_args()
    #### Requirement check #### 
    required = {'pandas'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    if missing:
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
    import pandas as pd
    rccl_log_process()
