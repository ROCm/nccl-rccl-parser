HIP_PATH?= $(wildcard /opt/rocm/hip)

ifeq (,$(HIP_PATH))
$(error Please set environment variable HIP_PATH manually.)
endif

ROCM_PATH?= $(wildcard /opt/rocm)
ifeq (,$(HIP_PATH))
$(error Please set environment variable ROCM_PATH manually.)
endif

HIPCC=$(HIP_PATH)/bin/hipcc


EXE=hip_rocm_smi_mapping

files=$(EXE).cpp

CXXFLAGS=-g -O3 -I$(ROCM_PATH)/rocm_smi/include -lrocm_smi64 -L$(ROCM_PATH)/rocm_smi/lib

all: $(EXE)

$(EXE): $(files)
		$(HIPCC) $(CXXFLAGS) $^ -o $@


clean:
		rm -f *.o $(EXE)