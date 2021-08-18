echo '==========================================================' 
HIP_VISIBLE_DEVICES=6,4,5,7 ./build/broadcast_perf -d int8 -b 40 -e 40 -o sum -g 4
HIP_VISIBLE_DEVICES=3,1,0,2 ./build/broadcast_perf -d int8 -b 40 -e 40 -o sum -g 4
echo '==========================================================' 
HIP_VISIBLE_DEVICES=6,2 ./build/broadcast_perf -d int8 -b 25952256 -e 25952256 -o sum -g 2
HIP_VISIBLE_DEVICES=3,7 ./build/broadcast_perf -d int8 -b 25952256 -e 25952256 -o sum -g 2
HIP_VISIBLE_DEVICES=0,4 ./build/broadcast_perf -d int8 -b 25952256 -e 25952256 -o sum -g 2
HIP_VISIBLE_DEVICES=1,5 ./build/broadcast_perf -d int8 -b 25952256 -e 25952256 -o sum -g 2
echo '==========================================================' 
HIP_VISIBLE_DEVICES=6,2 ./build/broadcast_perf -d int8 -b 2097152 -e 2097152 -o sum -g 2
HIP_VISIBLE_DEVICES=3,7 ./build/broadcast_perf -d int8 -b 2097152 -e 2097152 -o sum -g 2
HIP_VISIBLE_DEVICES=0,4 ./build/broadcast_perf -d int8 -b 2097152 -e 2097152 -o sum -g 2
HIP_VISIBLE_DEVICES=1,5 ./build/broadcast_perf -d int8 -b 2097152 -e 2097152 -o sum -g 2
echo '==========================================================' 
HIP_VISIBLE_DEVICES=6,2 ./build/broadcast_perf -d int8 -b 2048 -e 2048 -o sum -g 2
HIP_VISIBLE_DEVICES=3,7 ./build/broadcast_perf -d int8 -b 2048 -e 2048 -o sum -g 2
HIP_VISIBLE_DEVICES=0,4 ./build/broadcast_perf -d int8 -b 2048 -e 2048 -o sum -g 2
HIP_VISIBLE_DEVICES=1,5 ./build/broadcast_perf -d int8 -b 2048 -e 2048 -o sum -g 2
echo '==========================================================' 
HIP_VISIBLE_DEVICES=6,2 ./build/broadcast_perf -d int8 -b 1572864 -e 1572864 -o sum -g 2
HIP_VISIBLE_DEVICES=3,7 ./build/broadcast_perf -d int8 -b 1572864 -e 1572864 -o sum -g 2
HIP_VISIBLE_DEVICES=0,4 ./build/broadcast_perf -d int8 -b 1572864 -e 1572864 -o sum -g 2
HIP_VISIBLE_DEVICES=1,5 ./build/broadcast_perf -d int8 -b 1572864 -e 1572864 -o sum -g 2
echo '==========================================================' 
HIP_VISIBLE_DEVICES=6,2 ./build/broadcast_perf -d int8 -b 1536 -e 1536 -o sum -g 2
HIP_VISIBLE_DEVICES=3,7 ./build/broadcast_perf -d int8 -b 1536 -e 1536 -o sum -g 2
HIP_VISIBLE_DEVICES=0,4 ./build/broadcast_perf -d int8 -b 1536 -e 1536 -o sum -g 2
HIP_VISIBLE_DEVICES=1,5 ./build/broadcast_perf -d int8 -b 1536 -e 1536 -o sum -g 2
echo '==========================================================' 
HIP_VISIBLE_DEVICES=6,2 ./build/broadcast_perf -d int8 -b 524288 -e 524288 -o sum -g 2
HIP_VISIBLE_DEVICES=3,7 ./build/broadcast_perf -d int8 -b 524288 -e 524288 -o sum -g 2
HIP_VISIBLE_DEVICES=0,4 ./build/broadcast_perf -d int8 -b 524288 -e 524288 -o sum -g 2
HIP_VISIBLE_DEVICES=1,5 ./build/broadcast_perf -d int8 -b 524288 -e 524288 -o sum -g 2
echo '==========================================================' 
HIP_VISIBLE_DEVICES=6,4,5,7 ./build/broadcast_perf -d int8 -b 32 -e 32 -o sum -g 4
HIP_VISIBLE_DEVICES=3,1,0,2 ./build/broadcast_perf -d int8 -b 32 -e 32 -o sum -g 4
echo '==========================================================' 
HIP_VISIBLE_DEVICES=6,4,5,7 ./build/broadcast_perf -d int8 -b 65600 -e 65600 -o sum -g 4
HIP_VISIBLE_DEVICES=3,1,0,2 ./build/broadcast_perf -d int8 -b 65600 -e 65600 -o sum -g 4
echo '==========================================================' 
HIP_VISIBLE_DEVICES=6,4,5,7 ./build/all_reduce_perf -d half -b 16777216 -e 16777216 -o sum -g 4
HIP_VISIBLE_DEVICES=3,1,0,2 ./build/all_reduce_perf -d half -b 16777216 -e 16777216 -o sum -g 4
echo '==========================================================' 
HIP_VISIBLE_DEVICES=6,4,5,7 ./build/all_reduce_perf -d float -b 32768 -e 32768 -o max -g 4
HIP_VISIBLE_DEVICES=3,1,0,2 ./build/all_reduce_perf -d float -b 32768 -e 32768 -o max -g 4
echo '==========================================================' 
HIP_VISIBLE_DEVICES=6,4,5,7 ./build/all_reduce_perf -d float -b 32768 -e 32768 -o sum -g 4
HIP_VISIBLE_DEVICES=3,1,0,2 ./build/all_reduce_perf -d float -b 32768 -e 32768 -o sum -g 4
echo '==========================================================' 
HIP_VISIBLE_DEVICES=6,4,5,7 ./build/all_gather_perf -d int8 -b 4194304 -e 4194304 -o sum -g 4
HIP_VISIBLE_DEVICES=3,1,0,2 ./build/all_gather_perf -d int8 -b 4194304 -e 4194304 -o sum -g 4
echo '==========================================================' 
HIP_VISIBLE_DEVICES=6,2 ./build/all_reduce_perf -d half -b 179429376 -e 179429376 -o sum -g 2
HIP_VISIBLE_DEVICES=3,7 ./build/all_reduce_perf -d half -b 179429376 -e 179429376 -o sum -g 2
HIP_VISIBLE_DEVICES=0,4 ./build/all_reduce_perf -d half -b 179429376 -e 179429376 -o sum -g 2
HIP_VISIBLE_DEVICES=1,5 ./build/all_reduce_perf -d half -b 179429376 -e 179429376 -o sum -g 2
echo '==========================================================' 
HIP_VISIBLE_DEVICES=6,2 ./build/all_reduce_perf -d uint8 -b 1 -e 1 -o max -g 2
HIP_VISIBLE_DEVICES=3,7 ./build/all_reduce_perf -d uint8 -b 1 -e 1 -o max -g 2
HIP_VISIBLE_DEVICES=0,4 ./build/all_reduce_perf -d uint8 -b 1 -e 1 -o max -g 2
HIP_VISIBLE_DEVICES=1,5 ./build/all_reduce_perf -d uint8 -b 1 -e 1 -o max -g 2
echo '==========================================================' 
HIP_VISIBLE_DEVICES=6,4,5,7 ./build/all_reduce_perf -d uint8 -b 1 -e 1 -o max -g 4
HIP_VISIBLE_DEVICES=3,1,0,2 ./build/all_reduce_perf -d uint8 -b 1 -e 1 -o max -g 4
echo '==========================================================' 
