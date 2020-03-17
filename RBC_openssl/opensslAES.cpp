//example from https://wiki.openssl.org/index.php/EVP_Symmetric_Encryption_and_Decryption

#include "opensslAES.h"
#include "uint256_t.h"

#define N 10000000

int main(int argc, char **argv)
{    
    
    srand(7236); // for randomly generating keys 

    /* A random 256 bit key */
    uint256_t my_key( 0 );
    for( uint8_t i=0; i<UINT256_SIZE_IN_BYTES; ++i)
    {
        uint8_t temp = rand() % 10;
        my_key.set(temp,i);
    }
    unsigned char *key = (unsigned char *) my_key.get_data_ptr();

    printf("\nMy key: ");
    for(int i=0; i<UINT256_SIZE_IN_BYTES; ++i)
        printf( "0x%02x ", my_key.at(i) );
    printf("\n");

    //unsigned char *key = (unsigned char *)"01234567890123456789012345678901";



    /* A 128 bit IV */
    unsigned char *iv = (unsigned char *)"0123456789012345";

    /* Message to be encrypted */
    unsigned char *plaintext =
        (unsigned char *)"00000000000000001111111111111111";
        //(unsigned char *)"0111111111111111";


     /*
     * Buffer for ciphertext. Ensure the buffer is long enough for the
     * ciphertext which may be longer than the plaintext, depending on the
     * algorithm and mode.
     */
    unsigned char ciphertext[128];

    /* Buffer for the decrypted text */
    unsigned char decryptedtext[128];

    int decryptedtext_len;
    int plaintext_len=strlen ((char *)plaintext);
    printf("plaintext length = %d\n",plaintext_len);

    /* Create and initialize the contexts */
    EVP_CIPHER_CTX *ctx;
    if(!(ctx = EVP_CIPHER_CTX_new()))
        handleErrors();
    
    // last private ciphertext len so if we fix the key we can validate 
    // decryption works
    int ciphertext_len;
    //ciphertext_len = encrypt(plaintext, plaintext_len, key, iv, ciphertext);
    


    /* Start our RBC machine */

    struct timeval start, end;

    gettimeofday(&start, NULL);

    
    for (int i=0; i<N; i++)
    {
        ciphertext_len = encrypt(plaintext, plaintext_len, key, iv, ciphertext);

        //int len;
        //
        //// Initialise the encryption operation. IMPORTANT - ensure you use a key
        //// and IV size appropriate for your cipher
        //// In this example we are using 256 bit AES (i.e. a 256 bit key). The
        //// IV size for *most* modes is the same as the block size. For AES this
        //// is 128 bits
        //if(1 != EVP_EncryptInit_ex(ctx, EVP_aes_256_ecb(), NULL, key, iv))
        //    handleErrors(); 

        ////Provide the message to be encrypted, and obtain the encrypted output.
        ////EVP_EncryptUpdate can be called multiple times if necessary
        //if(1 != EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, plaintext_len))
        //    handleErrors();
        //
        //ciphertext_len = len;

        ////Finalise the encryption. Further ciphertext bytes may be written at
        ////this stage.    
        //if(1 != EVP_EncryptFinal_ex(ctx, ciphertext + len, &len))
        //    handleErrors();

        //ciphertext_len += len;
    }

    gettimeofday(&end, NULL);


    double  elapsed = ((end.tv_sec*1000000.0 + end.tv_usec) -
            (start.tv_sec*1000000.0 + start.tv_usec)) / 1000000.00;

    printf("Time to compute %d keys: %f (keys/second: %f)\n", N, elapsed, N*1.0/(elapsed));

    

    /////////////////////////////////

    //Uncomment this to print the cipher text and decrypt
    
    //Do something useful with the ciphertext here
    
    printf("Ciphertext is:\n");
    BIO_dump_fp (stdout, (const char *)ciphertext, ciphertext_len);





    //Decrypt the ciphertext
    decryptedtext_len = decrypt(ciphertext, ciphertext_len, key, iv,
                                decryptedtext);

    //Add a NULL terminator. We are expecting printable text 
    decryptedtext[decryptedtext_len] = '\0';

    // Show the decrypted text 
    printf("Decrypted text is:\n");
    printf("%s\n", decryptedtext);    
    /*
    */

    return 0;
}


void generate256bitKey(unsigned char * genString)
{
    unsigned char alphabet[10] = {'0','1','2','3','4','5','6','7','8','9'};
    // unsigned char genString[32];

    genString[32] = '\0';

    
    for (int i=0; i<32; i++) {
        fprintf(stderr,"\nhere");
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
