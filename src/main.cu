#include "main.h"

#include <iostream>

#define ROTL8(x,shift) ((uint8_t) ((x) << (shift)) | ((x) >> (8 - (shift))))
#define OPS_PER_THREAD 8192 // titan, gp100
//#define OPS_PER_THREAD 1024 // v100


int main(int argc, char * argv[])
{
    
    /* Parse args */
    
    if( argc != 4 )
    {
        printf("\nERROR: must enter 3 args only [ hamming-distance, verbose, num-devices ]\n");
        return -1;
    }

    int hamming_dist = atoi(argv[1]);
    int verbose  = atoi(argv[2]);
    int num_gpus = atoi(argv[3]);

    if( hamming_dist < 0 || hamming_dist > MAX_HAMMING_DIST )
    {
        fprintf(stderr,"Hamming distance must be between 0 and %d inclusive\n",MAX_HAMMING_DIST);
        return -2;
    }


    /* Make Client Data */

    struct ClientData client = make_client_data();
    aes_per_round::message_128 cipher;
    uint server_plaintext[ 4 ] = {0,0,0,0};
    uint server_ciphertext[ 4 ] = {0,0,0,0};
      // convert from big endian to little endian
    for( int i = 0; i < 16; i +=4 )
        {
            cipher.bits[ i ] = (uint8_t) client.ciphertext[ i + 3 ];
            cipher.bits[ i + 1 ] = (uint8_t) client.ciphertext[ i + 2 ];
            cipher.bits[ i + 2 ] = (uint8_t) client.ciphertext[ i + 1 ];
            cipher.bits[ i + 3 ] = (uint8_t) client.ciphertext[ i + 0 ];
        }
    server_plaintext[ 0 ] = bytes_to_int( client.plaintext );
    server_plaintext[ 1 ] = bytes_to_int( client.plaintext + 4 );
    server_plaintext[ 2 ] = bytes_to_int( client.plaintext + 8 );
    server_plaintext[ 3 ] = bytes_to_int( client.plaintext + 12 );

    server_ciphertext[ 0 ] = bytes_to_int( cipher.bits );
    server_ciphertext[ 1 ] = bytes_to_int( cipher.bits + 4 );
    server_ciphertext[ 2 ] = bytes_to_int( cipher.bits + 8 );
    server_ciphertext[ 3 ] = bytes_to_int( cipher.bits + 12 );
   

    /* Do RBC Authentication */

      // initializations
    omp_set_num_threads( num_gpus );
    struct timeval start, end;
    int dev=0, h=0, i=0;
    std::uint32_t ops_per_block = THREADS_PER_BLOCK * OPS_PER_THREAD;
    std::uint64_t total_keys[ hamming_dist ];
    std::uint64_t num_blocks[ hamming_dist ];
    std::uint64_t total_threads[ hamming_dist ];  
    std::uint64_t keys_per_thread[ hamming_dist ];
    std::uint64_t last_th_numkeys[ hamming_dist ];
    std::uint32_t extra_keys[ hamming_dist ];
        // multi-gpu calculations
    long long unsigned int total_iterations = 0;
    int blocks_per_gpu[ hamming_dist ];
    int offset[ hamming_dist ][ num_gpus ]; 
    int uprbnd[ hamming_dist ][ num_gpus ];
    #pragma omp parallel for private(h)
    for( h=0; h<hamming_dist; h++ )
    {
        total_keys[h]      = get_bin_coef( UINT256_SIZE_IN_BITS, h+1 );
        num_blocks[h]      = ( total_keys[h] / ops_per_block ) + 1;
        total_threads[h]   = num_blocks[h] * THREADS_PER_BLOCK;
        keys_per_thread[h] = total_keys[h] / total_threads[h];
        last_th_numkeys[h] = keys_per_thread[h] + total_keys[h] - (keys_per_thread[h] * total_threads[h]);
        extra_keys[h]      = last_th_numkeys[h] - keys_per_thread[h];
        blocks_per_gpu[h]  = (num_blocks[h]%num_gpus==0) ? (num_blocks[h]/num_gpus) : (num_blocks[h]/num_gpus)+1;
        offset[h][0]       = 0;
        uprbnd[h][0]       = (total_threads[h]/num_gpus) + (total_threads[h]%num_gpus);
        for( int g=1; g<num_gpus; ++g ) 
        {
            offset[h][g] = uprbnd[h][0] + (g-1)*(total_threads[h]/num_gpus);
            uprbnd[h][g] = offset[h][g] + (total_threads[h]/num_gpus);
        }
    }
        // host variables 
    uint256_t server_key( 0 );
    server_key.copy( client.key );
    //rand_flip_n_bits( &server_key, hamming_dist, UINT256_SIZE_IN_BITS );
    select_middle_key( &server_key, hamming_dist, total_threads[hamming_dist-1], num_gpus );
    uint256_t *host_server_key = &server_key;
    uint256_t *auth_key[ num_gpus ];
    std::uint64_t *total_iter_count[ num_gpus ];
        // device variables
    uint256_t *dev_server_key[ num_gpus ];
    uint * dev_plaintext[ num_gpus ];
    uint * dev_cipher[ num_gpus ];
    
    if( verbose ) 
    {
        print_prelim_info( client, server_key );
        for( h=0; h<hamming_dist; h++ )
            print_rbc_info( num_blocks[h], 
                            keys_per_thread[h], total_keys[h], 
                            last_th_numkeys[h], extra_keys[h], h+1 );
    }

      // turn on gpu
    if( verbose ) printf("\n\nTurning on the GPUs...\n");
    for( int dev=0; dev<num_gpus; ++dev ) warm_up_gpu( dev, verbose );
    if( verbose ) printf("\n");
        

    gettimeofday(&start, NULL);

      // allocate and set device variables
    #pragma omp parallel for private(dev)
    for( int dev=0; dev<num_gpus; ++dev )
    {
        cudaSetDevice( dev );
        cudaMalloc( (void**) &dev_plaintext[dev], 4*sizeof( uint ) );
        cudaMalloc( (void**) &dev_cipher[dev], 4*sizeof( uint ) );
        cudaMalloc( (void**) &dev_server_key[dev], sizeof( uint256_t ) );
        cudaMallocManaged( (void**) &total_iter_count[dev], sizeof( std::uint64_t ) );
        *total_iter_count[dev] = 0;
        cudaMallocManaged( (void**) &auth_key[dev], sizeof( uint256_t ) );

        if( cuda_utils::HtoD( dev_plaintext[dev], &server_plaintext, 4*sizeof( uint ) ) != cudaSuccess )
            {
                std::cout << "Failure to transfer uid to device\n";
            }
        if( cuda_utils::HtoD( dev_cipher[dev], &server_ciphertext, 4*sizeof( uint ) ) != cudaSuccess )
            {
                std::cout << "Failure to transfer cipher to device\n";
            }
        if( cuda_utils::HtoD( dev_server_key[dev], host_server_key, sizeof( uint256_t ) ) != cudaSuccess )
            {
                std::cout << "Failure to transfer corrupted_key to device\n";
            }
    }

      // run rbc kernel 
    for( h=1; h<=hamming_dist; ++h )
    {
        #pragma omp parallel for private(i)
        for( i=0; i<num_gpus; ++i ) *total_iter_count[i]=0;

        #pragma omp parallel for private(dev)
        for( dev=0; dev<num_gpus; dev++ )
        {
            cudaSetDevice( dev );

            kernel_rbc_engine<<<blocks_per_gpu[h-1],THREADS_PER_BLOCK>>>( dev_server_key[dev],
                                                                          auth_key[dev],
                                                                          h,
                                                                          dev_plaintext[dev],
                                                                          dev_cipher[dev],
                                                                          num_blocks[h-1],
                                                                          THREADS_PER_BLOCK,
                                                                          keys_per_thread[h-1],
                                                                          total_keys[h-1],
                                                                          extra_keys[h-1],
                                                                          total_iter_count[dev],
                                                                          offset[h-1][dev],
                                                                          uprbnd[h-1][dev]
                                                                        );
            cudaDeviceSynchronize();
        }

        for( int dev=0; dev<num_gpus; ++dev ) total_iterations += *total_iter_count[dev];
    }

    gettimeofday(&end, NULL);


    double elapsed = ((end.tv_sec*1000000.0 + end.tv_usec) -
                     (start.tv_sec*1000000.0 + start.tv_usec)) / 1000000.00;

    if( verbose )
    {
        printf("\nResulting Authentication Keys:\n");
        for( int dev=0; dev<num_gpus; ++dev ) auth_key[dev]->dump();
    }

    printf("\nTime to compute %Ld keys: %f (keys/second: %f)\n",total_iterations,elapsed,total_iterations*1.0/(elapsed));

    int success = 0;
    for( int dev=0; dev<num_gpus; ++dev ) if( *auth_key[dev] == client.key ) success=1;

    if( success )
        {
            std::cout << "SUCCESS: The keys match!\n";
        }
    else
        {
            std::cout << "ERROR: The keys do not match.\n";
        }


    return 0;
} 

unsigned char flip_n_bits( unsigned char val, int n )
{
    int hamming_dist = n;
    unsigned char mask_left = 0xFF << ( 8 - hamming_dist );
    unsigned char mask_right = 0xFF >> ( hamming_dist );

    return ( mask_left & ~val ) | ( mask_right & val );
                      
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
    ret.plaintext = (unsigned char *)"0000000011111111";
    //ret.plaintext = (unsigned char *)"0000000000000000";

    ret.plaintext_len = strlen( (char *)ret.plaintext );

    // buffer for ciphertext
    // - ensure the buffer is long enough for the ciphertext which may
    //   be longer than the plaintext, depending on the algorithm and mode

    // last private ciphertext len so if we fix the key we can validate 
    // decryption works
    uint8_t key[ 32 ];

    for( int idx = 0; idx < 8; ++idx )
        {
            int offset = idx * 4;
            int value = ret.key.data[ idx ];
            key[ offset + 0 ] = ((value>>24)&0xFF);
            key[ offset + 1 ] = ((value>>16)&0xFF);
            key[ offset + 2 ] = ((value>>8)&0xFF);
            key[ offset + 3 ] = ((value)&0xFF);
        }

    // encrypt plaintext with our random key 
    ret.ciphertext_len = encrypt(ret.plaintext,
                                 ret.plaintext_len,
                                 key,
                                 iv,
                                 ret.ciphertext);

    return ret;
}

void select_middle_key( uint256_t *server_key, int hamming_dist, int num_ranks, int n_gpus )
{
    // get key space metrics
    uint32_t ranks_per_gpu = num_ranks / n_gpus; // this implies that the key is always assigned to the first device
    uint64_t num_keys = get_bin_coef( 256, hamming_dist );
    uint32_t extra_keys = num_keys % num_ranks;
    uint64_t keys_per_thread = num_keys / num_ranks;
    
    // get our target ordinal for creating our target permutation
    uint32_t target_rank = ( ranks_per_gpu%2==0 ? (ranks_per_gpu/2)-1 : (ranks_per_gpu/2) );
    uint64_t target_ordinal = 0;
    if( num_ranks > num_keys ) // edge case, when hamming distance == 1
    {
        target_ordinal = num_keys%2==0 ? (num_keys/2)-1 : num_keys/2;
    }
    else
    {
        uint64_t target_rank_num_keys = keys_per_thread%2==0 ? (keys_per_thread/2)-1 : (keys_per_thread/2);
            // handle the case where we have extra keys
        if( target_rank < extra_keys )
            target_ordinal = target_rank*keys_per_thread + target_rank_num_keys;
        else
            target_ordinal = target_rank*keys_per_thread + extra_keys + target_rank_num_keys;
    }

    //fprintf(stderr,"\n\nTARGET: rank = %lu, ordinal = %llu\n\n",target_rank,target_ordinal);

    // get our target permutation
    uint256_t target_perm( 0 );
    decode_ordinal( &target_perm, target_ordinal, hamming_dist ); 

    // set server key to the middle of our key space distribution
    *server_key = *server_key ^ target_perm;
}

void rand_flip_n_bits(uint256_t *server_key, int n, int key_size_bits)
{
    srand((unsigned) time(0)); // for randomly generating keys 

    uint64_t num_keys = get_bin_coef( key_size_bits, n );
    uint64_t rand_ord = rand() % num_keys;

    // get our target perm
    uint256_t target_perm( 0 );
    decode_ordinal( &target_perm, rand_ord, n );

    // randomly corrupt the server key
    *server_key = *server_key ^ target_perm;
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

void print_prelim_info(ClientData client, uint256_t server_key)
{ 
    printf("\n----------------------------");
    printf("\nPreliminary Information");
    printf("\n----------------------------");
    printf("\nClient Key:\n");
    client.key.dump();
    printf("\nServer Corrupted Key:\n");
    server_key.dump();
    printf("\nClient Cipher Text (shared):\n");
    for(int i=0; i<16; ++i) printf("0x%02X ",client.ciphertext[i]);
    printf("\n\nClient Plain Text (shared):\n");
    printf("%s",client.plaintext);
    printf("\n----------------------------\n\n");
}

void print_rbc_info(long unsigned int num_blocks,
                    long unsigned int keys_per_thread,
                    long long unsigned int total_keys,
                    long unsigned int last_thread_numkeys,
                    int extra_keys,
                    int h)
{
    printf("\n----------------------------");
    printf("\nRBC Kernel Information for Hamming Distance %d", h);
    printf("\n----------------------------");

    printf("\n  Number of blocks: %lu",num_blocks);
    printf("\n  Number of threads per block: %d",THREADS_PER_BLOCK);
    printf("\n  Number of keys per thread: %lu",keys_per_thread);
    printf("\n  Total number of keys: %llu",total_keys);
    printf("\n  Last thread's number of keys: %lu\n",last_thread_numkeys);

    if( last_thread_numkeys != keys_per_thread )
    {
        printf("    Warning: num keys not divisible by num threads");
        printf("\n    Extra keys = %d\n", extra_keys);
    }
    printf("----------------------------\n");
}

