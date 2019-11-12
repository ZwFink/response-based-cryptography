NVCC=nvcc
vpath %.cu src
vpath %.cu test
vpath %.h include
vpath %.hpp lib

EXECUTABLES=gbench sbench test_rbc # aes aes_ecb benchmark benchmark_async benchmark_con benchmark_cpb benchmark_con_cpb
#OBJECTS=AES.o AES_benchmark.o AES_benchmark_con.o AES_benchmark_con_cpb.o AES_benchmark_cpb.o benchmark.o benchmark_async.o main.o main_ecb.o
#GENCODE = -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_75,code=sm_75
GENCODE = -gencode=arch=compute_60,code=sm_60
AES_FILES=AES.cu AES.h BlockCipher.h AES_encrypt.cu 
UINT_FILES=uint256_t.cu uint256_t.h 
UINT_ITER_FILES=uint256_iterator.cu uint256_iterator.h
CCFLAGS := -O3 --ptxas-options=-v -Xptxas -dlcm=ca $(GENCODE) \
-Xcompiler -fPIC -rdc=true -Xcompiler -fopenmp -std=c++11 -Iinclude/ -Itabs/
CCTESTFLAGS := -Itest/ -Ilib/ -Isrc/
TT?=128
MODE?=HYBRID

all: $(EXECUTABLES)

test_rbc: AES_smem.o catch.o uint.o uint_iter.o
	$(NVCC) $(CCFLAGS) -o $@ $^

test.o: catch.hpp test_utils.h test.cu
	$(NVCC) $(CCFLAGS) $(CCTESTFLAGS) -DTTABLE=$(TT) -D$(MODE) -c -o $@ $<

uint_iter.o: $(UINT_ITER_FILES)
	$(NVCC) $(CCFLAGS) -c -o $@ $<

uint.o: $(UINT_FILES) 
	$(NVCC) $(CCFLAGS) -c -o $@ $<

gbench: AES_gmem.o benchmark.o uint.o
	$(NVCC) $(CCFLAGS) -o $@ $^

sbench: AES_smem.o benchmark.o uint.o
	$(NVCC) $(CCFLAGS) -o $@ $^

AES_gmem.o: $(AES_FILES) $(UINT_FILES)
	$(NVCC) $(CCFLAGS) -DTTABLE=$(TT) -D$(MODE) -c -o $@ $<

AES_smem.o: $(AES_FILES) $(UINT_FILES) 
	$(NVCC) $(CCFLAGS) -DTTABLE=$(TT) -D$(MODE) -DUSE_SMEM -c -o $@ $<

catch.o: test.cu catch.hpp
	$(NVCC) $(CCFLAGS) $(CCTESTFLAGS) -c -o $@ $<

benchmark.o: benchmark.cu main.h
	$(NVCC) $(CCFLAGS) -c -o $@ $<

clean:
	$(RM) $(EXECUTABLES) *.o

clobber: clean
	rm *~
