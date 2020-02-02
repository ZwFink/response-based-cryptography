#include <iostream>
#include <omp.h>
#include "main.h"
#include "perm_util.h"

#define ROTL8(x,shift) ((uint8_t) ((x) << (shift)) | ((x) >> (8 - (shift))))
#define OPS_PER_THREAD 12800

unsigned char flip_n_bits( unsigned char val,
                           int n
)
{
    int mismatches = n;
    unsigned char mask_left = 0xFF << ( 8 - mismatches );
    unsigned char mask_right = 0xFF >> ( mismatches );

    return ( mask_left & ~val ) | ( mask_right & val );
                      
}

int main(int argc, char * argv[])
{
    if( argc != 4 )
    {
        printf("\nERROR: must enter 3 args only [ uid, key, mismatches ]\n");
        return 0;
    }

    // parse args
    char * uid     = argv[1];
    char * key     = argv[2];
    int mismatches = atoi(argv[3]);

    // partition key space
    std::uint64_t extra_keys    = 0;
    std::uint64_t total_keys    = get_bin_coef(UINT256_SIZE_IN_BITS,mismatches);
    std::uint32_t ops_per_block = THREADS_PER_BLOCK * OPS_PER_THREAD; 
    std::uint64_t num_blocks    = total_keys / ops_per_block; // kernel arg
    ++num_blocks;
   
    std::uint64_t total_threads    = num_blocks * THREADS_PER_BLOCK;
    std::uint64_t keys_per_thread  = total_keys / total_threads;
    std::uint64_t last_thread_numkeys = keys_per_thread + total_keys - keys_per_thread*total_threads;
    
    printf("\nNumber of blocks: %d",num_blocks);
    printf("\nNumber of threads per block: %d",THREADS_PER_BLOCK);
    printf("\nNumber of keys per thread: %d",keys_per_thread);
    printf("\nTotal number of keys: %d",total_keys);
    printf("\nLast thread's number of keys: %d",last_thread_numkeys);

    if( last_thread_numkeys != keys_per_thread )
    {
        extra_keys = last_thread_numkeys - keys_per_thread;

        printf("\n\nWarning: num keys not divisible by num threads");
        printf("\nLeftover keys = %d", extra_keys);
        printf("\nThe first %d threads will handle leftover keys",extra_keys);
    }

    else
    {
        printf("\nTotal number of threads evenly divides total number of keys");   
        printf("\nLeftover keys = %d",extra_keys);
    }

    ////////////////
    // turn on gpu
    printf("\nTurning on the GPU...\n");
    warm_up_gpu( 0 );

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
    message_128 uid_msg;
    for (int i = 0 ; i < 16; i++)
    {
        cipher.bits[i] = (uint8_t) uid_hex[i];
        uid_msg.bits[i] = (uint8_t) uid_hex[i];
    }

    //print_message(cipher);

    //print_key_256(bit_key);
    

    // make the sbox
    uint8_t sbox[256];
    aes_cpu::initialize_aes_sbox(sbox);
    //print_sbox(sbox);
    
    aes_cpu::encrypt_ecb( &cipher, &bit_key );
    //print_message(cipher);

    // corrupt bit_key by number of mismatches
    key_256 staging_key;
    for (int i = 0; i < 32; i++ )
    {
        staging_key.bits[i] = (uint8_t) bit_key.bits[i];
    }
    // this is subject to change...
    staging_key.bits[ 31 ] = flip_n_bits( bit_key.bits[ 31 ], mismatches );

    /* ok, we now have:
       - uid:          client's 128 bit message to encrypt
       - cipher:       client's encrypted cipher text to check against 
       - staging_key:  corrupted version of bit_key
    */
    printf("\nBegin authentication");
    printf("\n====================\n\n");
    double start_time = omp_get_wtime();
        
    // send userid, cipher, and corrupted key to GPU global memory
    uint256_t host_key_value;
    aes_per_round::message_128 * dev_uid = nullptr;
    aes_per_round::message_128 * dev_cipher = nullptr;
    uint256_t *dev_key = nullptr, * host_key = nullptr;
    host_key = &host_key_value;
    uint256_t *dev_found_key = nullptr;
    uint256_t host_found_key;
    for( uint8_t i=0; i < 32; i++ )
    { 
        host_key->set( staging_key.bits[i], i );
    }

    cudaMalloc( (void**) &dev_uid, sizeof( aes_per_round::message_128 ) );
    cudaMalloc( (void**) &dev_cipher, sizeof( aes_per_round::message_128 ) );
    cudaMalloc( (void**) &dev_key, sizeof( uint256_t ) );
    cudaMalloc( (void**) &dev_found_key, sizeof( uint256_t ) );

    std::uint64_t *total_iter_count = nullptr;
    cudaMallocManaged( (void**) &total_iter_count, sizeof( std::uint64_t ) );
    *total_iter_count = 0;


    if( cuda_utils::HtoD( dev_uid, &uid_msg, sizeof( aes_per_round::message_128 ) ) != cudaSuccess )
        {
            std::cout << "Failure to transfer uid to device\n";
        }

    if( cuda_utils::HtoD( dev_cipher, &cipher, sizeof( aes_per_round::message_128 ) ) != cudaSuccess)
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

	
    for( int i=mismatches; i <= mismatches; i++ ) // fixed
    {
       kernel_rbc_engine<<<num_blocks, THREADS_PER_BLOCK>>>( dev_key,
                                                             dev_found_key,
                                                             i,
                                                             dev_uid,
                                                             dev_cipher,
                                                             UINT256_SIZE_IN_BITS,
                                                             num_blocks,
                                                             THREADS_PER_BLOCK,
                                                             keys_per_thread,
                                                             total_keys,
                                                             extra_keys,
                                                             total_iter_count
                                                           );
       cudaDeviceSynchronize();
    }

    cudaError_t res = cudaSuccess;
    
    std::cout << "\nNum keys: " << total_keys << "\n";
    std::cout << "Iterated: " << *total_iter_count << "\n";
    std::cout << "Num_keys - Iterated: " << total_keys-*total_iter_count << "\n\n";
    if( ( res = cuda_utils::DtoH( &host_found_key, dev_found_key, sizeof( uint256_t ) ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer client_key_to_find to host \n";
            std::cout << "Failed with code: " << res << "\n";
        }

    double end_time = omp_get_wtime() ;

    std::cout << "Elapsed: " << end_time - start_time << "\n";

    host_found_key.dump();

    return 0;

} 


