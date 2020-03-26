#ifndef PERM_UTIL_HH_INCLUDED
#define PERM_UTIL_HH_INCLUDED

#include "uint256_t.h"

void decode_ordinal( uint256_t *perm, const uint64_t ordinal, uint8_t mismatches );
void assign_first_permutation( uint256_t *perm, int mismatches );
void assign_last_permutation( uint256_t *perm, int mismatches );
void get_perm_pair( uint256_t *starting_perm, 
                    uint256_t *ending_perm,
                    uint16_t tid,        
                    uint16_t num_threads,
                    const uint8_t mismatches,           
                    const std::uint64_t keys_per_thread,
                    const std::uint16_t extra_keys
                  );
uint64_t get_bin_coef(uint16_t n, uint16_t k);


#endif // PERM_UTIL_HH_INCLUDED
