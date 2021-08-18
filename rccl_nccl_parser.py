import os
import sys
import argparse
from collections import defaultdict
import pandas as pd
import networkx as nx

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
                #"10" : "ncclNumTypes Equivalent?"
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

def algobw_factor_times_size(coll_type, nranks, total_bytes):
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

def generate_script(commands, output_script):
    filename = output_script + ".sh"
    fs = open(filename, "w")
    for j in range(len(commands)):
        fs.write(commands[j])
        fs.write("\n")
    fs.close()
    print("INFO: Dumped out the commands in a script named: {}".format(filename))

def dump_counts_map(unique_command_list, counts_list, output_file): ###########################
    filename = output_file + ".csv"
    dict_command_count = {'command': unique_command_list, 'count': counts_list}
    table = pd.DataFrame(dict_command_count)
    table.to_csv(filename)
    print ("INFO: Dumped out the count of each command in a file named: {}".format(filename))    
    
# ts1-sjc2-27:11585:11585 [0] NCCL INFO Broadcast: opCount 0 sendbuff 0x7f988f200400 recvbuff 0x7f988f200400 count 440 datatype 0 op 0 root 0 comm 0x7f905c000dc0 [nranks=4] stream 0x55aa0a8d78f0
def coll_table_build(coll_lines):
    opCount, coll, count, datatype, op_type, root, comm, nranks, data_size = [], [], [], [], [], [], [], [], []
    for line in coll_lines:
        split_list = line.split()
        coll.append(split_list[4][:-1])
        opCount.append(int(split_list[6], 16))
        count.append(split_list[12])
        datatype.append(split_list[14])
        op_type.append(split_list[16])
        root.append(split_list[18])
        comm.append(split_list[20])
        nranks.append(int(next(item for item in split_list if 'nranks' in item).split("=")[1].replace("]", ""))) ### 
        data_size.append(int(split_list[split_list.index("count") + 1]) * data_type_bytes_map[split_list[split_list.index("datatype") + 1]])

    dict_coll = {'coll': coll, 'opCount': opCount, 'datatype': datatype, 'count':count, 'op_type':op_type, 'root':root, 'comm':comm, 'nranks':nranks, 'data_size':data_size}
    table = pd.DataFrame(dict_coll)    
    table['algobw_factor_times_size'] = table.apply(lambda row: 
                                                    algobw_factor_times_size(row['coll'], row['nranks'], row['data_size']), axis=1)
    table['raw_command'] = coll_lines
    return table

def conn_table_build(conn_lines, legacy_device_grouping):  # Only works for RCCL 2.9 or above
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
        if not legacy_device_grouping:
            if "comm" not in line:
                raise AssertionError("This NCCL/RCCL log is from an older version. Please use RCCL 2.9 or above.")
            comm.append(split_list[split_list.index("comm") + 1])
            nranks.append(int(split_list[split_list.index("nRanks") + 1]))
    
    if not legacy_device_grouping:
        dict_conn = {'start_rank': start_rank, 'start_busid': start_busid, 'end_rank': end_rank, 'end_busid': end_busid, 
                    'connection': connection, 'comm':comm, 'nranks':nranks, 'conn_line':conn_lines}      
    else:
        dict_conn = {'start_rank': start_rank, 'start_busid': start_busid, 'end_rank': end_rank, 'end_busid': end_busid, 
                    'connection': connection, 'conn_line':conn_lines}
    return pd.DataFrame(dict_conn)
    
def comm_table_build(comm_lines):
    comm, rank, nranks, cudaDev, busId = [], [], [], [], []   
    for line in comm_lines:
        split_list = line.rstrip().split(" ")
        comm.append(split_list[5])
        rank.append(split_list[7])
        nranks.append(int(split_list[9]))
        cudaDev.append(split_list[11])
        busId.append(split_list[13])
    dict_comm = {'comm':comm, 'rank':rank, 'nranks':nranks, 'cudaDev':cudaDev, 'busId':busId}  
    return pd.DataFrame(dict_comm)


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
                
def buildGraph(graphs, connectionList): # add an input of connection list
    G = nx.DiGraph()
    node_pool = []
    for graph in graphs:
        if graph[0][0] not in node_pool: 
            G.add_node(graph[0][0])
            node_pool.append(graph[0][0])
        for node in graph[1]:
            if node not in node_pool: 
                G.add_node(node)
                node_pool.append(node)
            label = connectionList[(connectionList['start_busid'] == graph[0][0]) & (connectionList['end_busid'] == node)]['connection'].unique()[0]
            G.add_edge(graph[0][0], node, label=label)

    to_remove = []
    for edge in G.edges():
        if (G.has_edge(edge[1], edge[0]) == False): to_remove.append([edge[0], edge[1]])
    for pair in to_remove: G.remove_edge(pair[0], pair[1])
    edges = G.to_undirected().edges()
    
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='#ffff00')
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, edge_color='b', arrows = True)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=nx.get_edge_attributes(G,'label'))
    
    parents = {}
    ds = DisjointSet()
    for edge in edges: ds.add(edge[0], edge[1])
    
    outputs = []
    for k, v in ds.group.items():
        outputs.append(v)
    return outputs
        
    
def hip_busId_mapping(path_to_deviceIdMapping):
    def processBusId(busId):
        split_list = busId.split(':')
        temp = split_list[2].split('.')
        return split_list[1].lstrip('0') + temp[0] + temp[1]
        
    fs = open(path_to_deviceIdMapping, 'r')
    lines = fs.readlines()
    fs.close()
    busId_HIP_map = {}
    for j in range(len(lines)):
        line = lines[j].rstrip()
        if "=" not in line:
            split_list = line.split()
            busId_HIP_map[processBusId(split_list[1])] = split_list[2]
    return busId_HIP_map

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

def generate_topo_script(commands, topo_info, busId_HIP_map, output_script):
    filename = output_script + ".sh"
    fs = open(filename, "w")
    for j in range(len(commands)):
        fs.write("echo '==========================================================' \n")
        for device_set in topo_info[j]:
            device_setting = "HIP_VISIBLE_DEVICES="
            for k, device in enumerate(list(device_set)):
                if k == len(device_set) - 1:
                    device_setting = device_setting + str(busId_HIP_map[device]) + " "
                else:                   
                    device_setting = device_setting + str(busId_HIP_map[device]) + ","
            fs.write(device_setting + commands[j])
            fs.write("\n")
    fs.write("echo '==========================================================' \n")
    fs.close()
    print("INFO: Dumped out the commands with device assignment in a script named: {}".format(filename))

def generate_topo(busId_HIP_map, command_list, raw_command_list, coll_table, conn_table, comm_table, legacy_device_grouping, output_name):
    all_info = pd.merge(coll_table, comm_table, on=['comm','nranks'])
    topo_info = []
    if legacy_device_grouping:
        for command in raw_command_list:
            split_list = command.split()
            coll = split_list[4][:-1]
            opCount = int(split_list[6], 16)
            count = split_list[12]
            datatype = split_list[14]
            op_type = split_list[16]
            nranks = int(next(item for item in split_list if 'nranks' in item).split("=")[1].replace("]", ""))        
            #### Filter
            selected_info = all_info[(all_info['coll'] == coll) & (all_info['opCount'] == opCount)
                                    & (all_info['datatype'] == datatype) & (all_info['count'] == count) 
                                    & (all_info['nranks'] == nranks) & (all_info['op_type'] == op_type)] 
            stage_2 = pd.merge(selected_info, conn_table, left_on=['rank','busId'], right_on=['start_rank','start_busid'],how='left')
            graphs = []
            for _ , subgroup in stage_2.groupby(['busId']):
                graphs.append([list(subgroup['busId'].unique()), list(subgroup['end_busid'].unique())])
            outputs = buildGraph(graphs, conn_table)
            topo_info.append(outputs)
        generate_topo_script(command_list, topo_info, busId_HIP_map, output_name)
    else:
        device_group_list = device_grouping(comm_table, conn_table)
        for command in command_list:
            split_list = command.split()
            nranks = int(split_list[split_list.index("-g") + 1])
            temp = []
            for deviceSet in device_group_list:
                if len(deviceSet) == nranks:
                    temp.append(deviceSet)
            topo_info.append(temp)
        generate_topo_script(command_list, topo_info, busId_HIP_map, output_name)

def get_commands(coll_table, unique):
    def nccl_rccl_tests_command(row):
        test_cmd = "./build/" + coll_op_map[row['coll']] + " -d " + data_types_map[row['datatype']] \
                    + " -b " + str(row['data_size']) + " -e " + str(row['data_size']) \
                    + " -o " + reduction_op_map[row['op_type']] + " -g " + str(row['nranks'])
        return test_cmd
    
    command_list = []
    raw_command_list = []
    counts_list = []
    if unique:
        unique_coll_table = coll_table.drop_duplicates(subset = ['coll','datatype', 'op_type', 'nranks', 'data_size'])
        for _, row in unique_coll_table.iterrows():
            test_cmd = nccl_rccl_tests_command(row)
            command_list.append(test_cmd)
            count = coll_table[(coll_table['coll'] == row['coll']) & (coll_table['datatype'] == row['datatype']) & 
                               (coll_table['op_type'] == row['op_type']) & (coll_table['nranks'] == row['nranks']) & 
                               (coll_table['data_size'] == row['data_size'])].shape[0]
            assert count % row['nranks'] == 0
            counts_list.append(int(count / row['nranks']))
            raw_command_list.append(row['raw_command'])
    else:
        for _, row in coll_table.iterrows():
            test_cmd = nccl_rccl_tests_command(row)
            command_list.append(test_cmd)
    return command_list, counts_list, raw_command_list


def main():
    log_file = os.path.abspath(args.nccl_debug_log)
    coll_lines, conn_lines, comm_lines, ring_lines, tree_lines, coll_trace_lines = get_useful_info(log_file)
    coll_table = coll_table_build(coll_lines)
    conn_table = conn_table_build(conn_lines, args.legacy_device_grouping)
    comm_table = comm_table_build(comm_lines)
    command_list, counts_list, raw_command_list = get_commands(coll_table, args.unique)
    path_to_deviceIdMapping = os.path.join(os.path.dirname(os.path.realpath(__file__)), "deviceIdMapping/busId_HIP_map.txt")
    if os.path.exists(path_to_deviceIdMapping) == False:
        raise AssertionError("Remember to run 'sh install.sh' before using this tool.")
    busId_hip_map = hip_busId_mapping(path_to_deviceIdMapping)
    if (args.unique):
        dump_counts_map(command_list, counts_list, args.output_script_name + "_counts")
        generate_topo(busId_hip_map, command_list, raw_command_list, coll_table, conn_table, 
                            comm_table, args.legacy_device_grouping, args.output_script_name + "_unique_topo")
    else:
        generate_script(command_list, args.output_script_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nccl-debug-log", type=str, required=True, help="RCCL log after running app with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL RCCL_KERNEL_COLL_TRACE_ENABLE=1 <executable>")
    parser.add_argument("--legacy-device-grouping", action="store_true", default=False, help="If the application is using CUDA systems or ROCm systems with RCCL 2.8 or below.") # 
    parser.add_argument("--output-script-name", type=str, required=False, default="net_nccl_rccl", help="Output command script")
    parser.add_argument("--unique", action="store_true", default=False, help="Get only the unique commands.")

    args = parser.parse_args()
    main()

# python rccl_nccl_parser_new.py --nccl-debug-log gpt2_rccl_mp4_log.txt --output-script-name net
# python rccl_nccl_parser_new.py --nccl-debug-log gpt2_rccl_mp4_log.txt --output-script-name net --unique --legacy-device-grouping
# python rccl_nccl_parser_new.py --nccl-debug-log gpt2_rccl_mp4_log_newPR.txt --output-script-name net --unique 
