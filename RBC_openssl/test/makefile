CC = g++
DEBUG = -g
CFLAGS = -fpermissive -c $(DEBUG) 
LFLAGS = $(DEBUG)
OBJS = test.o uint256_t.o

all : test

test : test.o uint256_t.o
	$(CC) -O3 -fopenmp $(OBJS) $(LFLAGS) -o test

test.o : test.cpp test.h
	$(CC) -fopenmp $(CFLAGS) test.cpp

uint256_t.o : uint256_t.cpp uint256_t.h
	$(CC) $(CFLAGS) uint256_t.cpp

uint256_iter.o : uint256_iterator.cpp uint256_iterator.h
	$(CC) $(CFLAGS) uint256_iterator.cpp

.PHONY: clean

clean:
	rm -f *.o test
