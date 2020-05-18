
#ifndef PERM_UTIL_HH_INCLUDED
#define PERM_UTIL_HH_INCLUDED

#include "uint256_t.h"
#include "cuda_defs.h"

__device__ __host__ void decode_ordinal( uint256_t *perm, const uint64_t ordinal, int mismatches );
__device__ void assign_first_permutation( uint256_t *perm, int mismatches );
__device__ void assign_last_permutation( uint256_t *perm, int mismatches );
__device__ void get_perm_pair( uint256_t *starting_perm, 
                               uint256_t *ending_perm,
                               uint64_t tid,        
                               uint64_t num_threads,
                               const uint8_t mismatches,           
                               const std::uint64_t keys_per_thread,
                               const std::uint32_t extra_keys
                             );
CUDA_CALLABLE_MEMBER uint64_t get_bin_coef(uint16_t n, uint16_t k);

#endif // PERM_UTIL_HH_INCLUDED

