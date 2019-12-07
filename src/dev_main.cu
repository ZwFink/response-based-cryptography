#include <iostream>
#include <omp.h>
#include "main.h"
#include "perm_util.h"

#define ROTL8(x,shift) ((uint8_t) ((x) << (shift)) | ((x) >> (8 - (shift))))
#define THREADS_PER_BLOCK 256
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

    std::uint32_t ops_per_block = THREADS_PER_BLOCK * OPS_PER_THREAD; 
    std::uint64_t num_blocks = get_bin_coef( UINT256_SIZE_IN_BITS, mismatches ) / ops_per_block;
    ++num_blocks;
    
    printf("\nNumber of blocks: %d",num_blocks);
    printf("\nNumber of threads per block: %d",THREADS_PER_BLOCK);
                                             
    ////////////////
    //Turn on gpu
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
    for (int i = 0 ; i < 16; i++)
    {
        cipher.bits[i] = (uint8_t) uid_hex[i];
    }

    print_message(cipher);

    print_key_256(bit_key);
    

    // make the sbox
    uint8_t sbox[256];
    aes_cpu::initialize_aes_sbox(sbox);
    //print_sbox(sbox);
    
    printf("sbox initialized\n");

    key_128 key_set[15];
    aes_cpu::key_gen(key_set, bit_key, sbox);

    printf("keys initialized\n");

    aes_cpu::xor_key(&cipher, key_set[0]);

    //print_message(cipher);

    for(unsigned int i = 0; i < 13; i++){
        //printf("ROUND: %u\n", i+1);
        //print_key_128(key_set[i]);
        //only working with 256 bit aes
        aes_cpu::sub_bytes(&cipher, sbox);
        
        //print_message(cipher);

        aes_cpu::shift_rows(&cipher);

        //print_message(cipher);

        aes_cpu::mix_columns(&cipher);

        //print_message(cipher);

        aes_cpu::xor_key(&cipher, key_set[i+1]);

        //print_message(cipher);
    }
    // printf("ROUND: %u\n", 14);
    aes_cpu::sub_bytes(&cipher, sbox);

    aes_cpu::shift_rows(&cipher);

    aes_cpu::xor_key(&cipher, key_set[14]);

    print_message(cipher);

    // corrupt bit_key by number of mismatches
    key_256 staging_key;
    for (int i = 0; i < 32; i++)
    {
        staging_key.bits[i] = (uint8_t) bit_key.bits[i];
    }
    // this is subject to change...
    staging_key.bits[ 31 ] = flip_n_bits( bit_key.bits[ 31 ], mismatches );
    for( int x = 0; x < 32; ++x )
        {
            printf( "0x%02X ", staging_key.bits[ x ] );

        }
    printf( "\n" );
    for( int x = 0; x < 32; ++x )
        {
            printf( "0x%02X ", bit_key.bits[ x ] );

        }
    printf( "\n" );


    /* ok, we now have:
       - uid:          client's 128 bit message to encrypt
       - cipher:       client's encrypted cipher text to check against 
       - staging_key:  corrupted version of bit_key
    */
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


    if( cuda_utils::HtoD( dev_uid, uid, sizeof( aes_per_round::message_128 ) ) != cudaSuccess )
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

	 cudaDeviceSynchronize();
	
    //for( int i=0; i <= mismatches; i++ )
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
                                                             total_iter_count
                                                           );
       cudaDeviceSynchronize();
    }

    cudaError_t res = cudaSuccess;
    std::cout << "Num keys: " << *total_iter_count << "\n";
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












