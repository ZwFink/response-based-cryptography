// utility file for host driver funciton (main)

#ifndef MAIN_UTIL_CU_
#define MAIN_UTIL_CU_

#include "perm_util.cu"
#include "uint256_iterator.h"
#include "AES.h"

__device__ int validator( uint256_t *starting_perm,
                          uint256_t *ending_perm,
                          uint256_t *key_for_encryp,
                          uint256_t user_id,
                          uint256_t auth_cipher 
                        );

__global__ void kernel_rbc_engine( uint256_t *key_for_encryp,
                                   size_t first_mismatch,
                                   size_t last_mismatch,
                                   const uint256_t user_id,
                                   const uint256_t auth_cipher,
                                   const size_t key_sz_bytes,
                                   const size_t key_sz_bits
                                 )
{
    unsigned int tid = threadIdx.x + ( blockIdx.x * blockDim.x );

    uint256_t starting_perm, ending_perm;

    size_t mismatch   = 0;
    uint64_t num_keys = 0;
    int result        = 0;

    for( mismatch = first_mismatch; mismatch <= last_mismatch; mismatch++ )
    {
        num_keys = get_bin_coef( key_sz_bits, mismatch ); 
   
        // only run thread if tid is less than cardinality of current keyspace
        if( tid < num_keys )
        {
            get_perm_pair( &starting_perm, 
                           &ending_perm, 
                           (size_t) tid, 
                           (size_t) NBLOCKS*BLOCKSIZE,
                           mismatch,
                           key_sz_bytes,
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
        }
    }

}

__device__ int validator( uint256_t *starting_perm,
                          uint256_t *ending_perm,
                          uint256_t *key_for_encryp,
                          const uint256_t user_id,
                          const uint256_t auth_cipher 
                        )
{

}


#endif
