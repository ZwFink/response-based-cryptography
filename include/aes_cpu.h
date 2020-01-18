#ifndef AES_CPU_HH_INCLUDED
#define AES_CPU_HH_INCLUDED

#include <cstdint>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

using uint = unsigned int;
using uchar = unsigned char;

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
void print_message(message_128 message);
void print_sbox(uint8_t sbox[256]);
void print_key_128(key_128 key);
void print_key_256(key_256 key);
void print_expanded_key(uint8_t expanded_key[240]);
void hex2bin(const char* src, uint8_t * target);



namespace aes_cpu
{

    // Calculate the rcon used in key expansion
    // from https://www.samiam.org/key-schedule.html 
    uint8_t rcon(int in);

    // This function assumes src to be a zero terminated sanitized string with
    // an even number of [0-9a-f] characters, and target to be sufficiently large
    void gmix_column(uint8_t r[4]) ;
    void xor_key(message_128 *message, key_128 key);
    void shift_rows(message_128 *message);
    void sub_bytes(message_128 *message, uint8_t sbox[256]);
    void mix_columns(message_128 *message);
    //From Wikepedia Rijndael mix columns
    //Straight from Wikiepedia on Rijndael S-box
    void initialize_aes_sbox(uint8_t sbox[256]);
    void key_gen(key_128 key_set[15], key_256 key, uint8_t sbox[256]);
    // from https://www.samiam.org/key-schedule.html
    void expand_key(uint8_t in[240], uint8_t sbox[256]);
    // This is the core key expansion, which, given a 4-byte value,
    // does some scrambling from https://www.samiam.org/key-schedule.html
    void schedule_core(uint8_t in[4], uint8_t i, uint8_t sbox[256]) ;
    //from https://www.samiam.org/key-schedule.html
    void rotate(uint8_t in[4]);
    void encrypt_ecb( message_128 *cipher,
                      key_256 *key
                    );


};

#endif // AES_CPU_HH_INCLUDED
