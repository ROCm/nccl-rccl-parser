#!/bin/bash
sudo apt-get install -y gawk
sudo apt-get install -y graphviz
pip install pandas
pip install networkx
cd deviceIdMapping; make
./hip_rocm_smi_mapping > busId_HIP_map.txt
