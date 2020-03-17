#ifndef PERM_UTIL_HH_INCLUDED
#define PERM_UTIL_HH_INCLUDED

#include "uint256_t.h"

void decode_ordinal( uint256_t *perm, 
                     const uint64_t ordinal, 
                     size_t mismatches, 
                     int key_sz_bits    
                   );
void assign_first_permutation( uint256_t *perm, int mismatches );
void assign_last_permutation( uint256_t *perm, int mismatches,int key_sz_bits );
void get_perm_pair( uint256_t *starting_perm, 
                    uint256_t *ending_perm,
                    std::size_t pair_index,        // thread num
                    std::size_t pair_count,        // num threads
                    const int mismatches,           
                    const std::size_t keys_per_thread,
                    std::size_t key_sz_bits,        
                    const std::uint64_t extra_keys,
                    const std::uint64_t total_perms
                  );
uint64_t get_bin_coef(size_t n, size_t k);



#endif // PERM_UTIL_HH_INCLUDED

