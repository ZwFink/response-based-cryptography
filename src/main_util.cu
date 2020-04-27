// utility file for host driver funciton (main)

#include "main_util.h"
#include "aes_tables.h"
#include "util.h"
#include <stdio.h>

__global__ void kernel_rbc_engine( uint256_t *key_for_encryp,
                                   uint256_t *key_to_find,
                                   const int mismatch,
                                   const aes_per_round::message_128 *user_id,
                                   const aes_per_round::message_128 *auth_cipher,
                                   const std::size_t key_sz_bits,
                                   const std::size_t num_blocks,
                                   const std::size_t threads_per_block,
                                   const std::size_t keys_per_thread,
                                   std::uint64_t num_keys,
                                   std::uint16_t extra_keys,
                                   std::uint64_t *iter_count
                                 )
{
    unsigned int tid = threadIdx.x + ( blockIdx.x * blockDim.x );

    uint256_t starting_perm, ending_perm;

    int result = 0;

    // only run thread if tid is less than cardinality of current keyspace
    if( tid < num_keys )
    {
        get_perm_pair( &starting_perm, 
                       &ending_perm, 
                       (uint64_t) tid, 
                       (uint64_t) num_blocks * threads_per_block,
                       mismatch,
                       keys_per_thread,
                       extra_keys
                     );
        
        result = validator( &starting_perm,
                            &ending_perm,
                            key_for_encryp,
                            key_to_find,
                            user_id,
                            auth_cipher
                          );

        atomicAdd( (unsigned long long int*) iter_count, result );
    }

}

__host__ __device__ uint bytes_to_int( const std::uint8_t *bytes )
{
    uint ret_val;
    ret_val =  ((int)bytes[3] << 24) | ((int)bytes[2] << 16) | ((int)bytes[1] << 8) | ((int)bytes[0]);
    return ret_val;
}
__device__ int validator( uint256_t *starting_perm,
                          uint256_t *ending_perm,
                          uint256_t *key_for_encryp,
                          uint256_t *key_to_find,
                          const aes_per_round::message_128 *user_id,
                          const aes_per_round::message_128 *auth_cipher 
                        )
{
    // aes_per_round::message_128 encrypted;

    uint userid[ 4 ];
    uint cyphertext[ 4 ];
    uint authcipher[ 4 ];

    userid[ 0 ] = bytes_to_int( user_id->bits );
    userid[ 1 ] = bytes_to_int( user_id->bits + 4 );
    userid[ 2 ] = bytes_to_int( user_id->bits + 8 );
    userid[ 3 ] = bytes_to_int( user_id->bits + 12 );

    authcipher[ 0 ] = bytes_to_int( auth_cipher->bits );
    authcipher[ 1 ] = bytes_to_int( auth_cipher->bits + 4 );
    authcipher[ 2 ] = bytes_to_int( auth_cipher->bits + 8 );
    authcipher[ 3 ] = bytes_to_int( auth_cipher->bits + 12 );

    aes_tables tabs;
    std::uint8_t idx = 0;
    std::uint8_t match = 0;
    int total = 0;

    #ifdef USE_SMEM
    __shared__ std::uint8_t sbox[ SBOX_SIZE_IN_BYTES ];
    if( threadIdx.x < SBOX_SIZE_IN_BYTES )
        {
            #if THREADS_PER_BLOCK == 128 

            sbox[ 2 * threadIdx.x ] = Tsbox_256[ 2 * threadIdx.x ];
            sbox[ ( 2 * threadIdx.x ) + 1 ] = Tsbox_256[ ( 2 * threadIdx.x ) + 1 ];

            #elif THREADS_PER_BLOCK == 256

            sbox[ threadIdx.x ] = Tsbox_256[ threadIdx.x ];

            #endif
        }
    __shared__ uint Te0[256], Te1[256], Te2[256], Te3[256];
    load_smem(Te0, cTe0, Te1, cTe1, Te2, cTe2, Te3, cTe3);
    // NOTE: __syncthreads not used here because it's called in
    // util::load_smem

    tabs.Te0 = Te0;
    tabs.Te1 = Te1;
    tabs.Te2 = Te2;
    tabs.Te3 = Te3;

    #else
    // just get a reference to it
    uint8_t *sbox = Tsbox_256;

    tabs.Te0 = cTe0;
    tabs.Te1 = cTe1;
    tabs.Te2 = cTe2;
    tabs.Te3 = cTe3;

    #endif 

    tabs.sbox = sbox;

    uint256_iter iter ( *key_for_encryp,
                        *starting_perm,
                        *ending_perm
                      );

    while( !iter.end() )
        {

            ++total;
            // encrypt
            aes_gpu::encrypt( userid,
                              cyphertext,
                              (uint*)(iter.corrupted_key.data),
                              &tabs
                            );

            // check for match! 
            for( idx = 0; idx < 4; ++idx )
                {
                    match += ( cyphertext[ idx ] == authcipher[ idx ] );
                }

           if( match == 4 )
                {
                    *key_to_find = iter.corrupted_key;
                    printf( "I found it!\n" );
                }

            match = 0;

            // get next key
            iter.next();

        }
    return total;
}

void warm_up_gpu( int device )
{
    cudaSetDevice( device ); 		
    // initialize all ten integers of a device_vector to 1 
    thrust::device_vector<int> D(10, 1); 
    // set the first seven elements of a vector to 9 
    thrust::fill(D.begin(), D.begin() + 7, 9); 
    // initialize a host_vector with the first five elements of D 
    thrust::host_vector<int> H(D.begin(), D.begin() + 5); 
    // set the elements of H to 0, 1, 2, 3, ... 
    thrust::sequence(H.begin(), H.end()); // copy all of H back to the beginning of D 
    thrust::copy(H.begin(), H.end(), D.begin()); 
    // print D 

    printf("\nDevice: %d\n",device);

    for(int i = 0; i < D.size(); i++) 
        std::cout << " D[" << i << "] = " << D[i]; 


    // empty the vector
    D.clear();

    // deallocate any capacity which may currently be associated with vec
    D.shrink_to_fit();

    printf("\n");

    return;
}


