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
GENCODE = -gencode=arch=compute_70,code=sm_70
AES_FILES=AES.cu AES.h BlockCipher.h AES_encrypt.cu 
AES_PER_ROUND_FILES=aes_per_round.cu aes_per_round.h
UINT_FILES=uint256_t.cu uint256_t.h 
UINT_ITER_FILES=uint256_iterator.cu uint256_iterator.h
AES_UTIL_FILES=util.cu util.h
SBOX_FILES=sbox.cu sbox.h 
UTIL_FILES=perm_util.cu perm_util.h cuda_utils.h
UTIL_MAIN_FILES=util_main.cu util_main.h cuda_utils.h
AES_CPU_FILES=aes_cpu.cpp aes_cpu.h
GENERAL_OBJECTS=aes_per_round.o sbox.o uint_iter.o uint.o util.o aes_cpu.o aes_util.o
CCFLAGS := -O3 --ptxas-options=-v -Xptxas -dlcm=ca $(GENCODE) -DITERCOUNT=1 -DRANDOM=0 -DEARLY_EXIT=1 -DTHREADS_PER_BLOCK=$(NUM_THREADS) -DNUM_THREADS=$(NUM_THREADS) \
-Xcompiler -fPIC -rdc=true -Xcompiler -fopenmp -std=c++11 -Iinclude/ -Itabs/ -DUSE_CONSTANT -DUSE_SMEM
DEBUGFLAGS := -O0 -g --ptxas-options=-v -Xptxas -dlcm=ca $(GENCODE) \
-Xcompiler -fPIC -rdc=true -Xcompiler -fopenmp -std=c++11 -Iinclude/ -Itabs/

LFLAGS := -lcrypto -lssl
CCTESTFLAGS := -Itest/ -Ilib/ -Isrc/
TT?=128
MODE?=HYBRID

all: $(EXECUTABLES)

test_rbc: $(GENERAL_OBJECTS) test.o util_main.o
	$(NVCC) $(CCFLAGS) -o $@ $^

test.o: test.cu catch.hpp test_utils.h 
	$(NVCC) $(CCFLAGS) $(CCTESTFLAGS) -DTTABLE=$(TT) -D$(MODE) -c -o $@ $<

aes_per_round.o: $(AES_PER_ROUND_FILES) cuda_defs.h
	$(NVCC) $(CCFLAGS) -c -o $@ $<

aes_util.o: $(AES_UTIL_FILES)
	$(NVCC) $(CCFLAGS) -c -o $@ $<

sbox.o: $(SBOX_FILES)
	$(NVCC) $(CCFLAGS) -c -o $@ $<

aes_cpu.o: $(AES_CPU_FILES) 
	$(NVCC) $(CCFLAGS) -c -o $@ $<

uint_iter.o: $(UINT_ITER_FILES) cuda_defs.h
	$(NVCC) $(CCFLAGS) -c -o $@ $<

util_main.o: $(UTIL_MAIN_FILES) 
	$(NVCC) $(CCFLAGS) -c -o $@ $< 

uint.o: $(UINT_FILES) cuda_defs.h
	$(NVCC) $(CCFLAGS) -c -o $@ $<

gbench: AES_gmem.o uint.o sbox.o aes_per_round.o uint_iter.o benchmark.o util.o
	$(NVCC) $(CCFLAGS) -o $@ $^

sbench: benchmark.o util_main.o $(GENERAL_OBJECTS)
	$(NVCC) $(CCFLAGS) -o $@ $^ $(LFLAGS)

util.o: $(UTIL_FILES)
	$(NVCC) $(CCFLAGS) -c -o $@ $<

AES_gmem.o: $(AES_FILES) $(UINT_FILES)
	$(NVCC) $(CCFLAGS) -DTTABLE=$(TT) -D$(MODE) -c -o $@ $<

AES_smem.o: $(AES_FILES) $(UINT_FILES)
	$(NVCC) $(CCFLAGS) -DTTABLE=$(TT) -D$(MODE) -DUSE_SMEM -c -o $@ $<

benchmark.o: main.cu main.h
	$(NVCC) $(CCFLAGS) -DMAX_HAMMING_DIST=16 -c -o $@ $< 

.PHONY: clean clobber debug

clean:
	$(RM) $(EXECUTABLES) *.o

clobber: clean
	rm *~

debug: clean
debug: CCFLAGS=$(DEBUGFLAGS)
debug: sbench
