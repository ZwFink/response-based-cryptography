CC=g++
CFLAGS=-fpermissive -std=c++11 -O3
LFLAGS=-lcrypto -lssl
OBJS=opensslAES.o uint256_t.o uint256_iterator.o perm_util.o

all: rbc

rbc: $(OBJS)
	$(CC) $(CFLAGS) -fopenmp -std=c++11 -o $@ $^ $(LFLAGS)

opensslAES.o: opensslAES_v2.0.cpp opensslAES.h
	$(CC) -fopenmp $(CFLAGS) -DMAX_HAMMING_DIST=5 -DNTHREADS=3 -c -o $@ $<

uint256_t.o: uint256_t.cpp uint256_t.h
	$(CC) $(CFLAGS) -c -o $@ $<

uint256_iterator.o: uint256_iterator.cpp uint256_iterator.h
	$(CC) -std=c++11 $(CFLAGS) -c -o $@ $<

perm_util.o: perm_util.cpp perm_util.h
	$(CC) $(CFLAGS) -c -o $@ $<

.PHONY: clean rbc opensslAES.o

clean:
	rm -f *.o rbc
