CC=g++
CFLAGS=-fpermissive -std=c++11 -O3
LFLAGS=-lcrypto -lssl
OBJS=main.o uint256_t.o uint256_iterator.o perm_util.o

all: rbc

rbc: $(OBJS)
	$(CC) $(CFLAGS) -fopenmp -std=c++11 -o $@ $^ $(LFLAGS)

main.o: main.cpp main.h
	$(CC) -fopenmp $(CFLAGS) -DMAX_HAMMING_DIST=5 -DNTHREADS=16 -c -o $@ $<

uint256_t.o: uint256_t.cpp uint256_t.h
	$(CC) $(CFLAGS) -c -o $@ $<

uint256_iterator.o: uint256_iterator.cpp uint256_iterator.h
	$(CC) -std=c++11 $(CFLAGS) -c -o $@ $<

perm_util.o: perm_util.cpp perm_util.h
	$(CC) $(CFLAGS) -c -o $@ $<

.PHONY: clean rbc main.o

clean:
	rm -f *.o rbc
