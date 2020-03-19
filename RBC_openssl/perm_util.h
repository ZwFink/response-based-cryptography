#ifndef PERM_UTIL_HH_INCLUDED
#define PERM_UTIL_HH_INCLUDED

#include "uint256_t.h"

void decode_ordinal( uint256_t *perm, 
                     const uint64_t ordinal, 
                     size_t mismatches
                   );
void assign_first_permutation( uint256_t *perm, int mismatches );
void assign_last_permutation( uint256_t *perm, int mismatches );
void get_perm_pair( uint256_t *starting_perm, 
                    uint256_t *ending_perm,
                    std::size_t tid,
                    std::size_t num_threads,
                    const int mismatches,           
                    const std::size_t keys_per_thread,
                    const std::uint64_t extra_keys
                  );
uint64_t get_bin_coef(size_t n, size_t k);


#endif // PERM_UTIL_HH_INCLUDED
