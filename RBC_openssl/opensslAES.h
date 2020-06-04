// opensslAES header file 
#ifndef OPENSSL_AES_HH_INCLUDED
#define OPENSSL_AES_HH_INCLUDED


// built-in declarations
#include <openssl/conf.h>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <string.h>
#include <stdio.h> 
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <omp.h>


// local declarations
#include "uint256_t.h"
#include "uint256_iterator.h"
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
void rand_flip_n_bits(uint256_t *server_key, uint256_t *client_key, int n);
void select_middle_key( uint256_t *server_key, int hamming_dist, int num_ranks );
void generate256bitKey(unsigned char * genString);
void handleErrors(void);
int encrypt(unsigned char *plaintext, int plaintext_len, unsigned char *key,
            unsigned char *iv, unsigned char *ciphertext);
int decrypt(unsigned char *ciphertext, int ciphertext_len, unsigned char *key,
            unsigned char *iv, unsigned char *plaintext);
ClientData make_client_data();
void print_prelim_info(ClientData client, uint256_t server_key);
void print_rbc_info(int mismatches,
                    long long unsigned int num_keys, 
                    long long unsigned int keys_per_thread, 
                    long unsigned int extra_keys);


#endif // OPENSSL_AES_HH_INCLUDED