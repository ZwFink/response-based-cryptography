#ifndef MAIN_HH_INCLUDED
#define MAIN_HH_INCLUDED


#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include <stdlib.h>
#include <sys/time.h>
#include "AES.h"
#include "aes_per_round.h"
#include "cuda_utils.h"
#include "main_util.h"
#include "uint256_t.h"

uint stringToUcharArray(char *str, uchar **array);
uint stringToUcharArray(char *str, uint **array);
void printHexArray(uint *array, uint size);

#define ROTL8(x,shift) ((uint8_t) ((x) << (shift)) | ((x) >> (8 - (shift))))

typedef struct key_256
{
    uint8_t bits[32];
} key_256;

typedef struct key_128
{
    uint8_t bits[16];
} key_128;

typedef struct message_128
{
    uint8_t bits[16];
} message_128;

int char2int(char input);
void hex2bin(const char* src, uint8_t * target);
void print_message(message_128 message);
void print_sbox(uint8_t sbox[256]);
void print_key_128(key_128 key);
void print_key_256(key_256 key);
void print_expanded_key(uint8_t expanded_key[240]);

// used for aes encryption on CPU to obtain client's cipher text
uint8_t rcon(int in);
void rotate(uint8_t in[4]);
void schedule_core(uint8_t in[4], uint8_t i, uint8_t sbox[256]);
void expand_key(uint8_t in[240], uint8_t sbox[256]);
void key_gen(key_128 key_set[15], key_256 key, uint8_t sbox[256]);
void initialize_aes_sbox(uint8_t sbox[256]);
void gmix_column(uint8_t r[4]);
void mix_columns(message_128 *message);
void sub_bytes(message_128 *message, uint8_t sbox[256]);
void shift_rows(message_128 *message);
void xor_key(message_128 *message, key_128 key);

#endif // MAIN_HH_INCLUDED
