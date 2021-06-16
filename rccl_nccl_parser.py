import os
import sys
import argparse

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
                
def get_useful_info(log_file):
    fs = open(log_file, 'r')
    lines = fs.readlines()
    fs.close()

    useful_lines = []
    for j in range(len(lines)):
        line = lines[j].rstrip()
        if ("opCount" in line and "sendbuff" in line):
            useful_lines.append(line)

    return useful_lines

def parse_nccl_log(nccl_lines):
    
    commands = []
    for j in range(len(nccl_lines)):
        line = nccl_lines[j]
        split_list = line.split(" ")
        comm = split_list[split_list.index("INFO") + 1].replace(":", "")
        count = split_list[split_list.index("count") + 1]
        datatype = split_list[split_list.index("datatype") + 1]
        op_type = split_list[split_list.index("op") + 1]
        root = split_list[split_list.index("root") + 1]
        nnranks = next(item for item in split_list if 'nranks' in item).split("=")[1].replace("]", "")

        #print (comm)
        #print (count)
        #print (datatype)
        #print (op_type)
        #print (root)
        #print (nnranks)

        total_bytes = int(count) * data_type_bytes_map[datatype]

        test_cmd = "./build/" + coll_op_map[comm] + " -d " + data_types_map[datatype] + \
                       " -b " + str(total_bytes) + " -e " + str(total_bytes) + \
                       " -o " + reduction_op_map[op_type] + " -g " + str(nnranks)
        #print (test_cmd)
        commands.append((test_cmd, int(nnranks)))

    return commands

def generate_script(commands, output_script):
    filename = output_script + ".sh"
    fs = open(filename, "w")
    for j in range(len(commands)):
        fs.write(commands[j])
        fs.write("\n")
    fs.close()
    print("INFO: Dumped out the commands in a script named: {}".format(filename))

def dump_counts_map(counts_map, output_file):
    filename = output_file + ".csv"
    fs = open(filename, 'w')
    fs.write("sep=|")
    fs.write("\n")
    keys = counts_map.keys()
    for key in keys:
        fs.write(key + "|" + str(counts_map[key]))
        fs.write("\n")
    fs.close()
    print ("INFO: Dumped out the count of each command in a file named: {}".format(filename))

def get_unique_commands(commands_and_nranks):
    unique_values = []
    counts_map = {}
    nranks_map = {}
    for c_and_nr in commands_and_nranks:
        cmd = c_and_nr[0]
        nranks = c_and_nr[1]
        if (cmd not in unique_values):
            counts_map[cmd] = 1
            nranks_map[cmd] = nranks
            unique_values.append(cmd)
        else:
            counts_map[cmd] = counts_map[cmd] + 1
    assert len(counts_map) == len(nranks_map)
    for cmd in counts_map.keys():
        assert counts_map[cmd] % nranks_map[cmd] == 0
        counts_map[cmd] = int(counts_map[cmd] / nranks_map[cmd])
    return unique_values, counts_map

def get_topo_info(log_file):
    fs = open(log_file, 'r')
    lines = fs.readlines()
    fs.close()

    useful_lines = []
    found = False
    j = 0
    
    while j < len(lines):
        line = lines[j].rstrip()
        if ("=== System : maxWidth" in line and found == False):
            topo_lines = []
            found = True
        elif(found):
            if ("Pattern" in line or "search.cc" in line):
                useful_lines.append(topo_lines)
                found = False
            elif ("=== System : maxWidth" in line):
                useful_lines.append(topo_lines)
                topo_lines = []
            else:
                topo_lines.append(line)
        j += 1
    return useful_lines

def generate_topo_script(commands, topo_info, output_script):
    filename = output_script + ".sh"
    fs = open(filename, "w")
    for j in range(len(commands)):
        fs.write(commands[j])
        fs.write("\n :' \n")
        for line in topo_info[j]:
            fs.write(line)
            fs.write("\n")
        fs.write("' \n")
    fs.close()
    print("INFO: Dumped out the commands in a script named: {}".format(filename))

def main():
    log_file = os.path.abspath(args.nccl_debug_log)
    nccl_lines = get_useful_info(log_file)
    commands_and_nranks = parse_nccl_log(nccl_lines)
    #generate_script(commands, args.output_script_name)
    if (args.unique):
        new_commands, counts_map = get_unique_commands(commands_and_nranks)
        generate_script(new_commands, args.output_script_name + "_unique")
        dump_counts_map(counts_map, args.output_script_name + "_counts")
        if (args.topology):
            topo_info = get_topo_info(log_file)
            generate_topo_script(new_commands, topo_info, args.output_script_name + "_unique_topo")
    else:
        commands = list(zip(*commands_and_nranks))[0]
        generate_script(commands, args.output_script_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nccl-debug-log", type=str, required=True, help="Log from app with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL")
    parser.add_argument("--output-script-name", type=str, required=False, default="net_nccl_rccl", help="Output command script")
    parser.add_argument("--unique", action="store_true", default=False, help="Get only the unique commands.")
    parser.add_argument("--topology", action="store_true", default=False, help="Must be used with --unique to output topology info as comments in command script.")
    args = parser.parse_args()
    main()
