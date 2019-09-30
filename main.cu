#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include "AES.h"
#include "main.h"

int main(int argc, char **argv) {
    if(argc < 2) {
        printf("USAGE: benchmark FILE\n");
        return 1;
    }

    uchar key[] = { 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00
    };
    uint keySize = 32;
    //uchar pt_debug[] = { 0x03, 0x02, 0x01, 0x00, 0x07, 0x06, 0x05, 0x04, 0x0b, 0x0a, 0x09, 0x08, 0x0f, 0x0e, 0x0d, 0x0c };
    
    uint *ct, *pt;
	
    FILE *f = fopen(argv[1], "rb");
    if(f == NULL) {
	printf("File not found.\n");
	return 1;
    }

    fseek(f, 0, SEEK_END);
    uint f_size = ftell(f);
    rewind(f);

    if(f_size % 4*sizeof(uint) != 0) {
	printf("Plaintext size must be a multiple of AES block size.\n");
	return 1;
    }

    uint ptSize = f_size / sizeof(uint);

#ifdef ASYNC
    cudaMallocHost((void**)&pt, f_size);
    cudaMallocHost((void**)&ct, f_size);
#else
    pt = (uint*)malloc(f_size);
    ct = (uint *)malloc(f_size);
#endif

    fread(pt, sizeof(uint), ptSize, f);
    fclose(f);

    AES *aes = new AES();
    aes->makeKey(key, keySize << 3, DIR_ENCRYPT);

    /*
    uchar pt_debug[] = { 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf };
    pt = (uint*)pt_debug;
    aes->encrypt_ecb(pt, ct, 4);
    printHexArray(ct, 4);
    */
    aes->encrypt_ecb(pt, ct, ptSize);
    //printHexArray(ct, ptSize);

    return 0;
}