import os
import sys
import argparse

def main():
    debug_log = os.path.abspath(args.nccl_debug_log)
    
    ## Generate a script to run nccl/rccl tests.
    if args.legacy_device_grouping:
        gen_cmd = gen_cmd = "python rccl_nccl_parser.py --nccl-debug-log " + debug_log + " --output-script-name net --unique --legacy-device-grouping"
    else:
        gen_cmd = "python rccl_nccl_parser.py --nccl-debug-log " + debug_log + " --output-script-name net --unique"
    if os.system(gen_cmd):
        print ("ERROR: Failed to parse the log.")
        sys.exit(1)
      
    ## change directory to rccl-tests/nccl-tests
    if args.rocm:
        rccl_tests_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rccl-tests")
        os.system("cp net_unique_topo.sh " + rccl_tests_path)
        os.chdir(rccl_tests_path)
        if os.system("./install.sh > /dev/null 2>&1"):
            print("ERROR: Failed to install rccl-tests.")
            sys.exit(1)
        
        os.system("cat net_unique_topo.sh")
        run_script_cmd = "HSA_FORCE_FINE_GRAIN_PCIE=1 sh net_unique_topo.sh | tee topo_rccl_tests.txt"

        if os.system(run_script_cmd):
            print ("ERROR: Unable to run rccl-tests properly.")
            sys.exit(1)
        os.system("mv topo_rccl_tests.txt ../")
        os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

        print (os.getcwd())
        summary_cmd = "python generate_summary.py --log-file topo_rccl_tests.txt --output-file-name net_summary --count-file net_counts.csv"
                
        os.system(summary_cmd)
        print ("INFO: Finished dumping all data.")
        
        

    if args.cuda:
        nccl_tests_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nccl-tests")
        os.system("cp net_unique_topo.sh " + nccl_tests_path)
        os.chdir(nccl_tests_path)
        if os.system("./install.sh > /dev/null 2>&1"):
            print("ERROR: Failed to install nccl-tests.")
            sys.exit(1)
        
        os.system("cat net_unique_topo.sh")
        run_script_cmd = "HSA_FORCE_FINE_GRAIN_PCIE=1 sh net_unique_topo.sh | tee topo_rccl_tests.txt"
        if os.system(run_script_cmd):
            print ("ERROR: Unable to run nccl-tests properly.")
            sys.exit(1)
        os.system("mv topo_rccl_tests.txt ../")
        os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

        print (os.getcwd())
        summary_cmd = "python generate_summary.py --log-file topo_rccl_tests.txt --output-file-name net_summary --count-file net_counts.csv"
                
        os.system(summary_cmd)
        print ("INFO: Finished dumping all data.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nccl-debug-log", type=str, required=True, \
                            help="NCCL/RCCL log after running app with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL")
    parser.add_argument("--rocm", action="store_true", default=False, help="Run the tests on ROCm using rccl-tests")
    parser.add_argument("--cuda", action="store_true", default=False, help="Run the tests on CUDA using nccl-tests")
    parser.add_argument("--legacy-device-grouping", action="store_true", default=False, help="NCCL/RCCL log after running app with NCCL or RCCL 2.8 or below")

    args = parser.parse_args()
    main()
# python run_parser_and_generate_summary.py --nccl-debug-log gpt2_rccl_mp4_log.txt --rocm --legacy-device-grouping
# python run_parser_and_generate_summary.py --nccl-debug-log gpt2_rccl_mp4_log_newPR.txt --rocm