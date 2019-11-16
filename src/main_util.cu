// utility file for host driver funciton (main)

#ifndef MAIN_UTIL_CU_
#define MAIN_UTIL_CU_

#include "perm_util.cu"
#include "AES.h"

__global__ void kernel_rbc_engine( uint256_t *key_for_encryp
                                   size_t start_mismatches,
                                   size_t end_mismatches,
                                   const uint256_t *user_id,
                                   const uint256_t *auth_cipher,
                                   const size_t key_sz_bytes,
                                   const size_t key_sz_bits
                                 )
{
    unsigned int tid = threadIdx.x + ( blockIdx.x * blockDim.x );
    uint256_t starting_perm, ending_perm;
   
    get_perm_pair( &starting_perm, 
                   &ending_perm, 
                   (size_t) tid, 
                   NUM_THREADS,    
                   mismatches,
                   key_sz_bytes,
                   key_sz_bits
                 );
    
    result = validator( key_to_be_found, 
                        &starting_perm,
                        &ending_perm,
                        key_for_encryp,
                        user_id,
                        auth_cipher
                      );

    // if result is 1 then we found a key matching client's private key
    // signal all threads to stop
}

__device__ int validator( const uint256_t *starting_perm,
                          const uint256_t *ending_perm,
                          uint256_t *key_for_encryp,
                          uint256_t *user_id,
                          uint256_t *auth_cipher 
                        )
{

}


#endif
