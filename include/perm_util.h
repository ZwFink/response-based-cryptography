
#ifndef PERM_UTIL_HH_INCLUDED
#define PERM_UTIL_HH_INCLUDED

#include "uint256_t.h"
#include "cuda_defs.h"

CUDA_CALLABLE_MEMBER uint64_t get_bin_coef(size_t n, size_t k);
__device__ void decode_ordinal( uint256_t *perm, 
                                const uint64_t ordinal, 
                                size_t mismatches, 
                                int key_sz_bits    
                              );
__device__ void assign_first_permutation( uint256_t *perm, int mismatches );
__device__ void assign_last_permutation( uint256_t *perm,
                                         int mismatches,
                                         int key_sz_bits );
__device__ void get_perm_pair( uint256_t *starting_perm, 
                               uint256_t *ending_perm,
                               size_t pair_index,        // thread id
                               size_t pair_count,        // num threads
                               int mismatches,           
                               const std::size_t keys_per_thread,
                               size_t key_sz_bits        
                             );


#endif // PERM_UTIL_HH_INCLUDED

