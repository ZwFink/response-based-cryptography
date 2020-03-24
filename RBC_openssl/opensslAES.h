// opensslAES header file 

// built-in declarations
#include <openssl/conf.h>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <string.h>
#include <stdio.h> 
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>

// local declarations
#include "omp.h"
#include "uint256_t.h"
#include "uint256_iterator.h"
#include "perm_util.h"

// utility functions
void rand_flip_n_bits(uint256_t *server_key, uint256_t *client_key, int n);
void generate256bitKey(unsigned char * genString);
void handleErrors(void);
int encrypt(unsigned char *plaintext, int plaintext_len, unsigned char *key,
            unsigned char *iv, unsigned char *ciphertext);
int decrypt(unsigned char *ciphertext, int ciphertext_len, unsigned char *key,
            unsigned char *iv, unsigned char *plaintext);