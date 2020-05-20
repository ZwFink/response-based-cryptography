#ifndef MAIN_HH_INCLUDED
#define MAIN_HH_INCLUDED


// built-in declarations
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <omp.h>
#include <openssl/conf.h>
#include <openssl/evp.h>
#include <openssl/err.h>


// local declarations
#include "AES.h"
#include "aes_per_round.h"
#include "cuda_utils.h"
#include "util_main.h"
#include "uint256_t.h"
#include "aes_cpu.h"
#include "perm_util.h"


// structs
struct ClientData
{
    uint256_t key;
    unsigned char *plaintext;
    int plaintext_len;
    unsigned char ciphertext[128];
    int ciphertext_len;
};


// utility functions
void make_client_data( struct ClientData *ret, int num_fragments );
void rand_flip_n_bits(uint256_t *server_key, uint256_t *client_key, int n);
void select_middle_key( uint256_t *server_key, int hamming_dist, int num_ranks, int n_gpus, int key_size_bits );
int encrypt(unsigned char *plaintext, int plaintext_len, unsigned char *key,
            unsigned char *iv, unsigned char *ciphertext);
void handleErrors(void);
unsigned char flip_n_bits( unsigned char val, int n );
void print_prelim_info(ClientData client, uint256_t server_key);
void print_rbc_info(long unsigned int num_blocks,
                    long unsigned int keys_per_thread,
                    long long unsigned int total_keys,
                    long unsigned int last_thread_numkeys,
                    int extra_keys,
                    int h);


#endif // MAIN_HH_INCLUDED
