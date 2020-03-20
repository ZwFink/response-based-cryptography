CC = g++
DEBUG = -g
CFLAGS = -fpermissive -c $(DEBUG) 
LFLAGS = -lcrypto -lssl $(DEBUG)
OBJS = opensslAES.o uint256_t.o uint256_iterator.o perm_util.o

all : rbc

rbc : opensslAES.o uint256_t.o uint256_iterator.o perm_util.o
	$(CC) -O3 -fopenmp $(OBJS) $(LFLAGS) -o rbc -lgmp

opensslAES.o : opensslAES.cpp opensslAES.h
	$(CC) -fopenmp $(CFLAGS) -DMAX_HAMMING_DIST=5 -DNTHREADS=16 opensslAES.cpp

uint256_t.o : uint256_t.cpp uint256_t.h
	$(CC) $(CFLAGS) uint256_t.cpp

uint256_iter.o : uint256_iterator.cpp uint256_iterator.h
	$(CC) $(CFLAGS) uint256_iterator.cpp

perm_util.o : perm_util.cpp perm_util.h
	$(CC) $(CFLAGS) perm_util.cpp

clean:
	\rm *.o rbc
