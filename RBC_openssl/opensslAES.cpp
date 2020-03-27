//example from https://wiki.openssl.org/index.php/EVP_Symmetric_Encryption_and_Decryption

#include "opensslAES.h"


int main(int argc, char **argv)
{    

    /* Parse args */

    if( argc != 3 )
    {
        fprintf(stderr,"Must run program as follows: ./rbc [hamming-distance] [verbose: 0 or 1]\n");
        return -1;
    } 

    int hamming_dist = atoi(argv[1]);
    bool verbose = atoi(argv[2]);

    if( hamming_dist < 0 || hamming_dist > MAX_HAMMING_DIST )
    {
        fprintf(stderr,"Hamming distance must be between 0 and %d inclusive\n",MAX_HAMMING_DIST);
        return -2;
    }


    /* Make Client Data */    

    struct ClientData client = make_client_data();


    /* Do RBC Authentication */

      // stuff for encryption/decryption
    unsigned char *iv = (unsigned char *)"0123456789012345";
    unsigned char decryptedtext[128];
    int decryptedtext_len;

      // get server-side key (simulated server-side PUF image)
    uint256_t server_key( 0 );
    server_key.copy( client.key );
    rand_flip_n_bits( &server_key, &client.key, hamming_dist );

      // initializations
    struct timeval start, end;
    uint256_t starting_perm(0), ending_perm(0);
    uint256_t auth_key( 0 );
    unsigned char server_ciphertext[128];
    int server_ciphertext_len;
    long long unsigned int count = 0;
    int mismatches = hamming_dist;
    long long unsigned int num_keys = get_bin_coef( 256, hamming_dist );
    long unsigned int extra_keys = num_keys % NTHREADS;
    long long unsigned int keys_per_thread = num_keys / NTHREADS; 
    if( verbose )
    {
        print_prelim_info( client, server_key );
        print_rbc_info( mismatches, num_keys, keys_per_thread, extra_keys );
    }

    omp_set_num_threads( NTHREADS );
    gettimeofday(&start, NULL);
 
    #pragma omp parallel private(starting_perm,ending_perm,server_ciphertext,server_ciphertext_len) reduction(+:count)
    {

        uint16_t tid = omp_get_thread_num();

        get_perm_pair( &starting_perm, 
                       &ending_perm, 
                       tid, 
                       NTHREADS,
                       hamming_dist,
                       keys_per_thread,
                       extra_keys
                     );

        uint256_iter iter( server_key, starting_perm, ending_perm );

        while( !iter.end() )
        {
            // encrypt
            server_ciphertext_len = encrypt( client.plaintext,
                                             client.plaintext_len,
                                             iter.corrupted_key.get_data_ptr(),
                                             iv,
                                             server_ciphertext
                                           );

            // check for match! 
            if(equal(server_ciphertext,server_ciphertext+16,client.ciphertext))
            {
                printf("\n  *** Client key found! ***");
                auth_key.copy(iter.corrupted_key);
            }

            // get next key
            iter.next();

            // update total keys iterated
            count++;
        }
    }

    gettimeofday(&end, NULL);


    if( verbose ) 
    {
        printf("\n\nResulting Authentication Key:\n");
        auth_key.dump();
    }


    double elapsed = ((end.tv_sec*1000000.0 + end.tv_usec) -
            (start.tv_sec*1000000.0 + start.tv_usec)) / 1000000.00;

    printf("\nTime to compute %Ld keys: %f (keys/second: %f)\n", count, elapsed, count*1.0/(elapsed));

    if( verbose ) printf("------------------------------\n\n");

    

    /////////////////////////////////

    //Uncomment this to print the cipher text and decrypt
    
    //Do something useful with the ciphertext here
    
    /*
    //Decrypt the ciphertext
    decryptedtext_len = decrypt(client_ciphertext, ciphertext_len, client.key.get_data_ptr(), iv,
                                decryptedtext);

    //Add a NULL terminator. We are expecting printable text 
    decryptedtext[decryptedtext_len] = '\0';

    // Show the decrypted text 
    printf("Decrypted text is:\n");
    printf("%s\n", decryptedtext);    
    */

    return 0;
}

void print_rbc_info(int mismatches,
                    long long unsigned int num_keys, 
                    long long unsigned int keys_per_thread, 
                    long unsigned int extra_keys)
{
    printf("\n------------------------------");
    printf("\nBegin RBC");
    printf("\n------------------------------");
    printf("\n  Hamming Distance: %d",mismatches);
    printf("\n  Keys to Iterate = %Ld",num_keys);
    //printf("\n  Keys Per Thread = %Ld",keys_per_thread);
    //printf("\n  Extra Keys = %lu",extra_keys);
}

void print_prelim_info(ClientData client, uint256_t server_key)
{
    printf("\n------------------------------");
    printf("\nPreliminary Information");
    printf("\n------------------------------");
    printf("\nClient Key:\n");
    client.key.dump();
    printf("\nServer Corrupted Key:\n");
    server_key.dump();
    printf("\nClient Cipher Text (shared):\n");
    for(int i=0; i<16; ++i) fprintf(stderr,"0x%02X ",client.ciphertext[i]);
    printf("\n\nClient Plain Text (shared):\n");
    printf("%s",client.plaintext);
    printf("\n------------------------------\n\n");
}

ClientData make_client_data()
{
    ClientData ret;

    // random 256 bit key - used by the client for encryption
    srand(7236); // for randomly generating keys 
    for( uint8_t i=0; i<UINT256_SIZE_IN_BYTES; ++i)
    {
        uint8_t temp = rand() % 10;
        ret.key.set(temp,i);
    }

    // 128 bit IV (initialization vector)
    unsigned char *iv = (unsigned char *)"0123456789012345";

    // message to be encrypted - from the client
    ret.plaintext = (unsigned char *)"00000000000000001111111111111111";

    ret.plaintext_len = strlen( (char *)ret.plaintext );

    // buffer for ciphertext
    // - ensure the buffer is long enough for the ciphertext which may
    //   be longer than the plaintext, depending on the algorithm and mode

    // last private ciphertext len so if we fix the key we can validate 
    // decryption works
    int ciphertext_len;

    // encrypt plaintext with our random key 
    ciphertext_len = encrypt(ret.plaintext,
                             ret.plaintext_len,
                             ret.key.get_data_ptr(),
                             iv,
                             ret.ciphertext);

    return ret;
}

void rand_flip_n_bits(uint256_t *server_key, uint256_t *client_key, int n)
{
    srand(238); // for randomly generating keys 

    int hamming_dist = n;
    int i=0;

    while( i<hamming_dist ) // loop until we flipped hamming_dist number of bits
    { 
        uint8_t bit_idx = rand() % 256;
        uint8_t block = bit_idx / 8;

        server_key->set_bit( bit_idx ); // bitwise OR operation

        // only increment if we successfully flipped the bit
        if( server_key->at(block) != client_key->at(block) )
            i++;
        else
            srand(239);
    }
}

void generate256bitKey(unsigned char * genString)
{
    unsigned char alphabet[10] = {'0','1','2','3','4','5','6','7','8','9'};
    // unsigned char genString[32];

    genString[32] = '\0';

    
    for (int i=0; i<32; i++) {
        int temp = rand() % 10;
        genString[i] = alphabet[temp];
    }

    
    // printf("\n%s",genString);
    
}

void handleErrors(void)
{
    ERR_print_errors_fp(stderr);
    abort();
}

int encrypt(unsigned char *plaintext, int plaintext_len, unsigned char *key,
            unsigned char *iv, unsigned char *ciphertext)
{
    EVP_CIPHER_CTX *ctx;

    int len;

    int ciphertext_len;

    /* Create and initialise the context */
    if(!(ctx = EVP_CIPHER_CTX_new()))
        handleErrors();

    /*
     * Initialise the encryption operation. IMPORTANT - ensure you use a key
     * and IV size appropriate for your cipher
     * In this example we are using 256 bit AES (i.e. a 256 bit key). The
     * IV size for *most* modes is the same as the block size. For AES this
     * is 128 bits
     */

    //MG- comment CBC
    // if(1 != EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv))
    //     handleErrors();
    //ECB mode  
    if(1 != EVP_EncryptInit_ex(ctx, EVP_aes_256_ecb(), NULL, key, iv))
        handleErrors();  

    /*
     * Provide the message to be encrypted, and obtain the encrypted output.
     * EVP_EncryptUpdate can be called multiple times if necessary
     */
    if(1 != EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, plaintext_len))
        handleErrors();
    ciphertext_len = len;

    /*
     * Finalise the encryption. Further ciphertext bytes may be written at
     * this stage.
     */
    if(1 != EVP_EncryptFinal_ex(ctx, ciphertext + len, &len))
        handleErrors();
    ciphertext_len += len;

    /* Clean up */
    EVP_CIPHER_CTX_free(ctx);

    return ciphertext_len;
}


int decrypt(unsigned char *ciphertext, int ciphertext_len, unsigned char *key,
            unsigned char *iv, unsigned char *plaintext)
{
    EVP_CIPHER_CTX *ctx;

    int len;

    int plaintext_len;

    /* Create and initialise the context */
    if(!(ctx = EVP_CIPHER_CTX_new()))
        handleErrors();

    /*
     * Initialise the decryption operation. IMPORTANT - ensure you use a key
     * and IV size appropriate for your cipher
     * In this example we are using 256 bit AES (i.e. a 256 bit key). The
     * IV size for *most* modes is the same as the block size. For AES this
     * is 128 bits
     */

    //MG- comment CBC  
    // if(1 != EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv))
    //     handleErrors();
    if(1 != EVP_DecryptInit_ex(ctx, EVP_aes_256_ecb(), NULL, key, iv))
        handleErrors();  

    /*
     * Provide the message to be decrypted, and obtain the plaintext output.
     * EVP_DecryptUpdate can be called multiple times if necessary.
     */
    if(1 != EVP_DecryptUpdate(ctx, plaintext, &len, ciphertext, ciphertext_len))
        handleErrors();
    plaintext_len = len;

    /*
     * Finalise the decryption. Further plaintext bytes may be written at
     * this stage.
     */
    if(1 != EVP_DecryptFinal_ex(ctx, plaintext + len, &len))
        handleErrors();
    plaintext_len += len;

    /* Clean up */
    EVP_CIPHER_CTX_free(ctx);

    return plaintext_len;
}
