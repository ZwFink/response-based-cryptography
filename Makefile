NVCC=nvcc
vpath %.cu src
vpath %.cpp src
vpath %.cu test
vpath %.h include
vpath %.h test
vpath %.hpp lib

NUM_THREADS=256
EXECUTABLES=sbench # test_rbc gbench 
#GENCODE = -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_75,code=sm_75
GENCODE = -gencode=arch=compute_60,code=sm_60
AES_FILES=AES.cu AES.h BlockCipher.h AES_encrypt.cu 
AES_PER_ROUND_FILES=aes_per_round.cu aes_per_round.h
UINT_FILES=uint256_t.cu uint256_t.h 
UINT_ITER_FILES=uint256_iterator.cu uint256_iterator.h
AES_UTIL_FILES=util.cu util.h
SBOX_FILES=sbox.cu sbox.h 
UTIL_FILES=perm_util.cu perm_util.h cuda_utils.h
UTIL_MAIN_FILES=util_main.cu util_main.h cuda_utils.h
AES_CPU_FILES=aes_cpu.cpp aes_cpu.h
GENERAL_OBJECTS=objs/aes_per_round.o objs/sbox.o objs/uint_iter.o objs/uint.o objs/util.o objs/aes_cpu.o objs/aes_util.o
CCFLAGS := -O3 --ptxas-options=-v -Xptxas -dlcm=ca $(GENCODE) -DITERCOUNT=1 -DRANDOM=1 -DEARLY_EXIT=1 -DTHREADS_PER_BLOCK=$(NUM_THREADS) -DNUM_THREADS=$(NUM_THREADS) \
-Xcompiler -fPIC -rdc=true -Xcompiler -fopenmp -std=c++11 -Iinclude/ -Itabs/ -DUSE_CONSTANT -DUSE_SMEM
DEBUGFLAGS := -O0 -g --ptxas-options=-v -Xptxas -dlcm=ca $(GENCODE) \
-Xcompiler -fPIC -rdc=true -Xcompiler -fopenmp -std=c++11 -Iinclude/ -Itabs/

LFLAGS := -lcrypto -lssl
CCTESTFLAGS := -Itest/ -Ilib/ -Isrc/
TT?=256
MODE?=HYBRID


all: $(EXECUTABLES)

test_rbc: $(GENERAL_OBJECTS) objs/test.o objs/util_main.o
	$(NVCC) $(CCFLAGS) -o $@ $^

objs/test.o: test.cu catch.hpp test_utils.h 
	$(NVCC) $(CCFLAGS) $(CCTESTFLAGS) -DTTABLE=$(TT) -D$(MODE) -c -o $@ $<

objs/aes_per_round.o: $(AES_PER_ROUND_FILES) cuda_defs.h
	$(NVCC) $(CCFLAGS) -c -o $@ $<

objs/aes_util.o: $(AES_UTIL_FILES)
	$(NVCC) $(CCFLAGS) -c -o $@ $<

objs/sbox.o: $(SBOX_FILES)
	$(NVCC) $(CCFLAGS) -c -o $@ $<

objs/aes_cpu.o: $(AES_CPU_FILES) 
	$(NVCC) $(CCFLAGS) -c -o $@ $<

objs/uint_iter.o: $(UINT_ITER_FILES) cuda_defs.h
	$(NVCC) $(CCFLAGS) -c -o $@ $<

objs/util_main.o: $(UTIL_MAIN_FILES) 
	$(NVCC) $(CCFLAGS) -c -o $@ $< 

objs/uint.o: $(UINT_FILES) cuda_defs.h
	$(NVCC) $(CCFLAGS) -c -o $@ $<

objs/util.o: $(UTIL_FILES)
	$(NVCC) $(CCFLAGS) -c -o $@ $<

objs/AES_gmem.o: $(AES_FILES) $(UINT_FILES)
	$(NVCC) $(CCFLAGS) -DTTABLE=$(TT) -D$(MODE) -c -o $@ $<

objs/AES_smem.o: $(AES_FILES) $(UINT_FILES)
	$(NVCC) $(CCFLAGS) -DTTABLE=$(TT) -D$(MODE) -DUSE_SMEM -c -o $@ $<

objs/benchmark.o: main.cu main.h
	@mkdir -p $(@D)
	$(NVCC) $(CCFLAGS) -DMAX_HAMMING_DIST=16 -c -o $@ $< 

gbench: objs/AES_gmem.o objs/uint.o objs/sbox.o objs/aes_per_round.o objs/uint_iter.o objs/benchmark.o objs/util.o
	$(NVCC) $(CCFLAGS) -o $@ $^

sbench: objs/benchmark.o objs/util_main.o $(GENERAL_OBJECTS) 
	$(NVCC) $(CCFLAGS) -o $@ $^ $(LFLAGS)

.PHONY: clean clobber debug

clean:
	rm -rf objs $(EXECUTABLES)

clobber: clean
	rm *~

debug: clean
debug: CCFLAGS=$(DEBUGFLAGS)
debug: sbench
