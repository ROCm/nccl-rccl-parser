import os
import sys
import argparse
import re

def get_script_commands(script_file):
    fs = open(script_file, 'r')
    lines = fs.readlines()
    fs.close()

    commands = []
    for j in range(len(lines)):
        line = lines[j].rstrip()
        commands.append(line)

    return commands

def parse_useful_information(log_file):
    fs = open(log_file, 'r')
    lines = fs.readlines()
    fs.close()

    useful_lines = []
    for j in range(len(lines)):
        line = lines[j].rstrip()
        if ("time" in line and "algbw" in line and "busbw" in line):
            perf_line = lines[j+2]
            if ("Avg bus bandwidth" in lines[j+5]):
                perf_line = perf_line + lines[j + 5]
            elif ("Avg bus bandwidth" in lines[j+4]):
                perf_line = perf_line + lines[j+4]
            useful_lines.append(perf_line)
    return useful_lines

def parse_nccl_performance(useful_lines, commands):
    
    perf_lines = []
    perf_lines.append("sep=|")
    header = "size|count|type|redop|root|time-oplace(us)|algbw(gb/s)-oplace|busbw(gb/s)-oplace|error|" + \
             "time-iplace(us)|algbw(gb/s)-iplace|busbw(gb/s)-iplace|error|avg_bus_bw|commands"
    #print(header)
    num_fields = len(header.split("|"))
    perf_lines.append(header)
    for j in range(len(useful_lines)):
        line = useful_lines[j]
        line = line.replace("# Avg bus bandwidth    : ", "")
        
        split_list = line.split()
        perf_line = ""
        field_index = 0
        for i in range(len(split_list)):
            perf_line = perf_line + split_list[i] + "|"
            # Some collectives do not involve a redop
            if field_index==2 and "reduce" not in commands[j].lower():
                perf_line = perf_line + "|"
                field_index = field_index + 1
            # Only broadcast and reduce involve a root
            if (
               field_index==3 and
               re.search(r'\Wreduce_perf', commands[j]) is None and
               re.search(r'\Wbroadcast_perf', commands[j]) is None
            ):
                perf_line = perf_line + "|"
                field_index = field_index + 1
            field_index = field_index + 1
        #print (perf_line + commands[j])
        perf_line = perf_line + commands[j]
        assert len(perf_line.split("|")) == num_fields
        perf_lines.append(perf_line)

    return perf_lines

def get_counts_from_file(count_file):
    fs = open(count_file, 'r')
    lines = fs.readlines()
    fs.close()

    counts = []
    for j in range(1, len(lines)):
        line = lines[j].rstrip()
        counts.append(line.split("|")[1])
    return counts

def update_perf_lines(perf_lines, counts):
    
    updated_lines = []
    updated_lines.append("sep=|")
    updated_lines.append(perf_lines[1] + "|count")
    for j in range(2, len(perf_lines)):
        perf_line = perf_lines[j] + "|" +  counts[j-2]
        updated_lines.append(perf_line)

    return updated_lines
        
def generate_output_file(out_file, perf_lines):
    fs = open(out_file, 'w')
    for j in range(len(perf_lines)):
        fs.write(perf_lines[j])
        fs.write('\n')
    fs.close()
    print ("INFO: Dumped out the performance.")

def main():
    log_file = os.path.abspath(args.log_file)
    out_file = args.output_file_name + ".csv"
    script_file = os.path.abspath(args.script_file)

    commands = get_script_commands(script_file)
    useful_lines = parse_useful_information(log_file)
    perf_lines = parse_nccl_performance(useful_lines, commands)
    if args.count_file:
        counts = get_counts_from_file(os.path.abspath(args.count_file))
        perf_lines = update_perf_lines(perf_lines, counts)
    generate_output_file(out_file, perf_lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, required=True, help="Log file generated while running rccl-tests")
    parser.add_argument("--output-file-name", type=str, required=False, default="net_summary")
    parser.add_argument("--script-file", type=str, required=True, help="Script file to run NCCL/RCCL Tests")
    parser.add_argument("--count-file", type=str, required=False, help="net_count file generated while running unique option in parser.")

    args = parser.parse_args()
    main()
