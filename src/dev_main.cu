
#include "AES.h"
#include "main.h"
#include "main_util.cu"
#include "uint256_t.h"
#include "cuda_utils.h"

using namespace std;

#define ROTL8(x,shift) ((uint8_t) ((x) << (shift)) | ((x) >> (8 - (shift))))

int main(int argc, char * argv[])
{
    if( argc != 3 )
    {
        printf("\nERROR: must enter 3 args only [ uid, key, mismatches ]");
        return 0;
    }

    // parse args
    char * uid     = argv[1];
    char * key     = argv[2];
    int mismatches = atoi(argv[3]);

    uint8_t key_hex[32];
    uint8_t uid_hex[16];

    hex2bin(key,key_hex);
    hex2bin(uid, uid_hex);

    key_256 bit_key;
    for (int i = 0 ; i < 32; i++)
    {
        bit_key.bits[i] = (uint8_t) key_hex[i];
    }

    message_128 cipher;
    for (int i = 0 ; i < 16; i++)
    {
        cipher.bits[i] = (uint8_t) uid_hex[i];
    }

    print_message(cipher);

    print_key_256(bit_key);
    

    // make the sbox
    uint8_t sbox[256];
    initialize_aes_sbox(sbox);
    //print_sbox(sbox);
    
    printf("sbox initialized\n");

    key_128 key_set[15];
    key_gen(key_set, bit_key, sbox);

    printf("keys initialized\n");

    xor_key(&cipher, key_set[0]);

    //print_message(cipher);

    for(unsigned int i = 0; i < 13; i++){
        //printf("ROUND: %u\n", i+1);
        //print_key_128(key_set[i]);
        //only working with 256 bit aes
        sub_bytes(&cipher, sbox);
        
        //print_message(cipher);

        shift_rows(&cipher);

        //print_message(cipher);

        mix_columns(&cipher);

        //print_message(cipher);

        xor_key(&cipher, key_set[i+1]);

        //print_message(cipher);
    }
    // printf("ROUND: %u\n", 14);
    sub_bytes(&cipher, sbox);

    shift_rows(&cipher);

    xor_key(&cipher, key_set[14]);

    print_message(cipher);

    // corrupt bit_key by number of mismatches
    key_256 staging_key;
    for (int i = 0; i < 32; i++)
    {
        staging_key.bits[i] = (uint8_t) bit_key.bits[i];
    }
    // this is subject to change...
    for (int i = 0; i < mismatches*2; i=i+2)
    {
        // flip the third bit of the first 6 even numbered blocks 
        staging_key.bits[i] ^= (1 << 3); 
    }

    /* ok, we now have:
       - uid:          client's 128 bit message to encrypt
       - cipher:       client's encrypted cipher text to check against 
       - staging_key:  corrupted version of bit_key
    */
        
    // send userid, cipher, and corrupted key to GPU global memory
    message_128 * dev_uid = nullptr;
    message_128 * dev_cipher = nullptr;
    uint256_t * dev_key = nullptr, * host_key = nullptr;
    uint256_t * dev_found_key = nullptr;
    uint256_t host_found_key;
    for( uint8_t i=0; i < 32; i++ )
    { 
        host_key->set( staging_key.bits[i], i );
    }

    cudaMalloc( (void**) &dev_uid, sizeof( message_128 ) );
    cudaMalloc( (void**) &dev_cipher, sizeof( message_128 ) );
    cudaMalloc( (void**) &dev_key, sizeof( uint256_t ) );
    cudaMalloc( (void**) &dev_found_key, sizeof( uint256_t ) );

    if( cuda_utils::HtoD( dev_uid, uid, sizeof( message_128 ) ) != cudaSuccess )
        {
            std::cout << "Failure to transfer uid to device\n";
        }

    if( cuda_utils::HtoD( dev_cipher, &cipher, sizeof( message_128 ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer cipher to device\n";
        }

    if( cuda_utils::HtoD( dev_key, host_key, sizeof( uint256_t ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer corrupted_key to device\n";
        }

    if( cuda_utils::HtoD( dev_found_key, &host_found_key, sizeof( uint256_t ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer client_key_to_find to device\n";
        }

	 cudaDeviceSynchronize();

    for( int i=0; i < mismatches; i++ )
    {
        // kernel invocation here    
    }

    if( cuda_utils::DtoH( &host_found_key, dev_found_key, sizeof( uint256_t ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer client_key_to_find to host \n";
        }

    return 0;
}


/* Calculate the rcon used in key expansion
from https://www.samiam.org/key-schedule.html */
uint8_t rcon(int in) {
    uint8_t c = 1;
    if(in == 0) {return 0;}
    
    while(in != 1) {
        uint8_t b;
        b = c & 0x80;
        c <<= 1;
        if(b == 0x80) {
            c ^= 0x1b;
        }
        in--;
    }
    return c;
}

//from https://www.samiam.org/key-schedule.html
void rotate(uint8_t in[4]) {
    uint8_t a,c;
    a = in[0];
    for(c=0;c<3;c++) 
            in[c] = in[c + 1];
    in[3] = a;
    return;
}

/* This is the core key expansion, which, given a 4-byte value,
 * does some scrambling from https://www.samiam.org/key-schedule.html*/
 void schedule_core(uint8_t in[4], uint8_t i, uint8_t sbox[256]) {
    uint8_t a;
    /* Rotate the input 8 bits to the left */
    rotate(in);
    /* Apply Rijndael's s-box on all 4 bytes */
    for(a = 0; a < 4; a++) {
        uint8_t less_nibble = in[a] & 0x0f;
        uint8_t more_nibble = (in[a] & 0xf0) >> 4;
        in[a] = sbox[less_nibble + more_nibble*16];
    } 
    /* On just the first byte, add 2^i to the byte */
    in[0] ^= rcon(i);
}

// from https://www.samiam.org/key-schedule.html
void expand_key(uint8_t in[240], uint8_t sbox[256]) {
    uint8_t t[4];
    uint8_t c = 32;
    uint8_t i = 1;
    uint8_t a;
    while(c < 240) {

        /* Copy the temporary variable over */
        for(a = 0; a < 4; a++) {
            t[a] = in[a + c - 4]; 
        }
                
        /* Every eight sets, do a complex calculation */
        if(c % 32 == 0) {
            schedule_core(t,i,sbox);
            i++;
            }

        /* For 256-bit keys, we add an extra sbox to the
            * calculation */
        if(c % 32 == 16) {
            for(a = 0; a < 4; a++) {
                uint8_t less_nibble = t[a] & 0x0f;
                uint8_t more_nibble = (t[a] & 0xf0) >> 4;
                t[a] = sbox[less_nibble + more_nibble*16];
            } 
        }

        for(a = 0; a < 4; a++) {
            in[c] = in[c - 32] ^ t[a];
            c++;
        }
    }
}

void key_gen(key_128 key_set[15], key_256 key, uint8_t sbox[256]){
    
    uint8_t expanded_key[240];
    for(int i = 0; i < 32; i++){
        expanded_key[i] = key.bits[i];
    }
    
    expand_key(expanded_key, sbox);

    //print_expanded_key(expanded_key);

    for (int i = 0; i < 15; i++){
        for (int j = 0; j < 16; j++){
            key_set[i].bits[j] = expanded_key[j+i*16];
        }
    }

}

//Straight from Wikiepedia on Rijndael S-box
void initialize_aes_sbox(uint8_t sbox[256]) {
	uint8_t p = 1, q = 1;
	
	/* loop invariant: p * q == 1 in the Galois field */
	do {
		/* multiply p by 3 */
		p = p ^ (p << 1) ^ (p & 0x80 ? 0x1B : 0);

		/* divide q by 3 (equals multiplication by 0xf6) */
		q ^= q << 1;
		q ^= q << 2;
		q ^= q << 4;
		q ^= q & 0x80 ? 0x09 : 0;

		/* compute the affine transformation */
		uint8_t xformed = q ^ ROTL8(q, 1) ^ ROTL8(q, 2) ^ ROTL8(q, 3) ^ ROTL8(q, 4);

		sbox[p] = xformed ^ 0x63;
	} while (p != 1);

	/* 0 is a special case since it has no inverse */
	sbox[0] = 0x63;
}

//From Wikepedia Rijndael mix columns
void gmix_column(uint8_t r[4]) {
    uint8_t a[4];
    uint8_t b[4];
    uint8_t c;
    uint8_t h;
    /* The array 'a' is simply a copy of the input array 'r'
     * The array 'b' is each element of the array 'a' multiplied by 2
     * in Rijndael's Galois field
     * a[n] ^ b[n] is element n multiplied by 3 in Rijndael's Galois field */ 
    for (c = 0; c < 4; c++) {
        a[c] = r[c];
        /* h is 0xff if the high bit of r[c] is set, 0 otherwise */
        h = (uint8_t)((signed char)r[c] >> 7); /* arithmetic right shift, thus shifting in either zeros or ones */
        b[c] = r[c] << 1; /* implicitly removes high bit because b[c] is an 8-bit char, so we xor by 0x1b and not 0x11b in the next line */
        b[c] ^= 0x1B & h; /* Rijndael's Galois field */
    }

    r[0] = b[0] ^ a[3] ^ a[2] ^ b[1] ^ a[1]; /* 2 * a0 + a3 + a2 + 3 * a1 */
    r[1] = b[1] ^ a[0] ^ a[3] ^ b[2] ^ a[2]; /* 2 * a1 + a0 + a3 + 3 * a2 */
    r[2] = b[2] ^ a[1] ^ a[0] ^ b[3] ^ a[3]; /* 2 * a2 + a1 + a0 + 3 * a3 */
    r[3] = b[3] ^ a[2] ^ a[1] ^ b[0] ^ a[0]; /* 2 * a3 + a2 + a1 + 3 * a0 */
}

void mix_columns(message_128 *message){
    //mix each set of 4 bytes
    message_128 temp = *message;
    
    uint8_t r[4];

    //first col
    r[0] = temp.bits[0];
    r[1] = temp.bits[1];
    r[2] = temp.bits[2];
    r[3] = temp.bits[3];

    gmix_column(r);

    message->bits[0] = r[0];
    message->bits[1] = r[1];
    message->bits[2] = r[2];
    message->bits[3] = r[3];

    //sec col
    r[0] = temp.bits[4];
    r[1] = temp.bits[5];
    r[2] = temp.bits[6];
    r[3] = temp.bits[7];

    gmix_column(r);

    message->bits[4] = r[0];
    message->bits[5] = r[1];
    message->bits[6] = r[2];
    message->bits[7] = r[3];

    //third col
    r[0] = temp.bits[8];
    r[1] = temp.bits[9];
    r[2] = temp.bits[10];
    r[3] = temp.bits[11];

    gmix_column(r);

    message->bits[8] = r[0];
    message->bits[9] = r[1];
    message->bits[10] = r[2];
    message->bits[11] = r[3];

    //fourth col
    r[0] = temp.bits[12];
    r[1] = temp.bits[13];
    r[2] = temp.bits[14];
    r[3] = temp.bits[15];

    gmix_column(r);

    message->bits[12] = r[0];
    message->bits[13] = r[1];
    message->bits[14] = r[2];
    message->bits[15] = r[3];

}

void sub_bytes(message_128 *message, uint8_t sbox[256]){
    //take the bytes and seperate the nibbles out
    for (int i = 0; i < 16; i++){
        uint8_t less_nibble  = message->bits[i] & 0x0f;
        uint8_t more_nibble = (message->bits[i] & 0xf0) >> 4;
        message->bits[i] = sbox[less_nibble + more_nibble*16];
    } 

}

void shift_rows(message_128 *message){

    uint8_t temp;

    //shift row 2 by 1
    temp = message->bits[1];
    message->bits[1] = message->bits[5];
    message->bits[5] = message->bits[9];
    message->bits[9] = message->bits[13];
    message->bits[13] = temp;

    //shift row 3 by 2
    temp = message->bits[2];
    message->bits[2] = message->bits[10];
    message->bits[10] = temp;
    temp = message->bits[6];
    message->bits[6] = message->bits[14];
    message->bits[14] = temp;

    //shift row 4 by 3
    temp = message->bits[3];
    message->bits[3] = message->bits[15];
    message->bits[15] = message->bits[11];
    message->bits[11] = message->bits[7];
    message->bits[7] = temp;

}

void xor_key(message_128 *message, key_128 key){

    for (int j = 0; j < 16; j++){
        message->bits[j] ^= key.bits[j];
    }
}

/* 
    start utility prints and conversion methods
*/

int char2int(char input)
{
  if(input >= '0' && input <= '9')
    return input - '0';
  if(input >= 'A' && input <= 'F')
    return input - 'A' + 10;
  if(input >= 'a' && input <= 'f')
    return input - 'a' + 10;
  return 0;
}

// This function assumes src to be a zero terminated sanitized string with
// an even number of [0-9a-f] characters, and target to be sufficiently large
void hex2bin(const char* src, uint8_t * target)
{
  while(*src && src[1])
  {
    *(target++) = char2int(*src)*16 + char2int(src[1]);
    src += 2;
  }
}

void print_message(message_128 message)
{
    printf("Hex Message: ");
    for(int i = 0; i < 16; i++){
        printf("%02hhX ", message.bits[i]);
    }
    printf("\n");
}
void print_sbox(uint8_t sbox[256])
{
    printf("sbox:");
    for(int i = 0; i < 16; i++)
    {
        printf("\n");
        for(int j = 0; j < 16; j++)
        {
            printf("%02hhX ", sbox[i*16+j]);
        }
    }
    printf("\n");
}

void print_key_128(key_128 key)
{
    printf("128 key: ");
    for(int i = 0; i < 16; i++)
    {
        printf("%02hhX ", key.bits[i]);
    }
    printf("\n");
}

void print_key_256(key_256 key)
{
    printf("256 key: ");
    for(int i = 0; i < 32; i++)
    {
        printf("%02hhX ", key.bits[i]);
    }
    printf("\n");
}

void print_expanded_key(uint8_t expanded_key[240])
{
    printf("Expanded key: ");
    for(int i = 0; i < 15; i++)
    {
        printf("\n");
        for(int j = 0; j < 16; j++)
        {
            printf("%02hhX ", expanded_key[i*16+j]);
        }
    }
    printf("\n");
}

uint stringToUcharArray(char *str, uchar **array) {
    uint i, len  = strlen(str) >> 1;
    *array = (uchar *)malloc(len * sizeof(uchar));
	
    for(i=0; i<len; i++)
	sscanf(str + i*2, "%02X", *array+i);

    return len;
}

uint stringToUcharArray(char *str, uint **array) {
    uint i, len  = strlen(str) >> 3;
    *array = (uint *)malloc(len * sizeof(uint));
	
    for(i=0; i<len; i++)
	sscanf(str + i*8, "%08X", *array+i);

    return len;
}

void printHexArray(uint *array, uint size) {
    uint i;
    for(i=0; i<size; i++)
	printf("%08X", array[i]);
    printf("\n");
}


