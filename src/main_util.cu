// utility file for host driver funciton (main)

#include "main_util.h"
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
                                   std::uint64_t *iter_count
                                 )
{
    unsigned int tid = threadIdx.x + ( blockIdx.x * blockDim.x );

    uint256_t starting_perm, ending_perm;

    uint64_t num_keys = 0;
    int result        = 0;

    num_keys = get_bin_coef( key_sz_bits, mismatch ); 
   
    // only run thread if tid is less than cardinality of current keyspace
    if( tid < num_keys )
    {
        get_perm_pair( &starting_perm, 
                       &ending_perm, 
                       (std::size_t) tid, 
                       (std::size_t) num_blocks * threads_per_block,
                       mismatch,
                       keys_per_thread,
                       key_sz_bits
                     );
        
        result = validator( &starting_perm,
                            &ending_perm,
                            key_for_encryp,
                            user_id,
                            auth_cipher
                          );

        // if result is 1 then we found a key matching client's private key
        // signal all threads to stop
        // if( result )
        //     {
        //         *key_to_find = *key_for_encryp; 
        //     }
        atomicAdd( (unsigned long long int*) iter_count, result );
    }

}

__device__ int validator( uint256_t *starting_perm,
                          uint256_t *ending_perm,
                          uint256_t *key_for_encryp,
                          const aes_per_round::message_128 *user_id,
                          const aes_per_round::message_128 *auth_cipher 
                        )
{
    aes_per_round::message_128 encrypted;
    int idx = 0;
    std::uint8_t match = 0;
    std::uint8_t match2 = 0;
    int total = 0;

    for( idx = 0; idx < 4; ++idx )
        {
            ((uint32_t*)&(encrypted.bits))[ idx ] = 0;
        }

    #ifdef USE_SMEM
    __shared__ std::uint8_t sbox[ SBOX_SIZE_IN_BYTES ];
    if( threadIdx.x < SBOX_SIZE_IN_BYTES )
        {
            sbox[ threadIdx.x ] = Tsbox_256[ threadIdx.x ];
        }

    __syncthreads();

    #else
    // just get a reference to it
    uint8_t *sbox = Tsbox_256;
    #endif 

    uint256_iter iter ( *key_for_encryp,
                        *starting_perm,
                        *ending_perm
                      );
    while( !iter.end() )
        {

            ++total;
            // encrypt
            aes_per_round::roundwise_encrypt( &encrypted,
                                              &iter.corrupted_key,
                                              user_id,
                                              sbox
                                            );

            // check for match! 
            for( idx = 0; idx < 16; ++idx )
                {
                    match += ( encrypted.bits[ idx ] == auth_cipher->bits[ idx ] );
                }
            match2 += match == 16; // if all 16 bytes matched, we have a match!

            if( match == 16 )
                {
                    *key_for_encryp = iter.corrupted_key;
                    printf( "I found it!\n" );
                }

            match = 0;

            // get next key
            iter.next();

            for( idx = 0; idx < 4; ++idx )
                {
                    ((uint32_t*)&(encrypted.bits))[ idx ] = 0;
                }

        }
    return total;
    // return match2;
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


