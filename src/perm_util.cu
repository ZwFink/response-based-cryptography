// utility file for permutation delimination

#ifndef PERM_UTIL_CU_
#define PERM_UTIL_CU_

#include "uint256_t.h"

__device__ uint64_t get_bin_coef(size_t n, size_t k);

// COMPLETED
__device__ void decode_ordinal( uint256_t *perm, 
                                const uint64_t ordinal, 
                                size_t mismatches, // 0-6
                                int key_sz_bits    // 256
                              )
{
   uint64_t binom = 0;
   uint64_t wkg_ord = ordinal;
   perm->set_all( 0 );

   for( size_t bit = key_sz_bits-1; mismatches > 0; bit-- )
   {
      binom = get_bin_coef( bit, mismatches );

      if ( wkg_ord >= binom )
      {
         wkg_ord = wkg_ord - binom;

         perm->set_bit( bit );

         mismatches--;
      }
   }
}

// COMPLETED
__device__ void assign_first_permutation( uint256_t *perm, int mismatches )
{
   // set the value of perm to 1
   perm->set_bit( 0 );

   *perm = *perm << mismatches; // shift left
	
   perm->add( *perm, UINT256_NEGATIVE_ONE ); // add negative one
}

// COMPLETED
__device__ void assign_last_permutation( uint256_t *perm,
                                         int mismatches,
                                         int key_sz_bits )
{

   // set perm to the first key
   assign_first_permutation( perm, mismatches );

   // Then find the last key by shifting the first
   // Equiv to: perm << (key_length - mismatches)
   // E.g. if key_length = 256 and mismatches = 5,
   //      we want to shift left 256 - 5 = 251 times.
   *perm = *perm << (key_sz_bits - mismatches);
}

// COMPLETED
// Precondition: starting_perm and ending_perm have been initialized
__device__ void get_perm_pair( uint256_t *starting_perm, 
                               uint256_t *ending_perm,
                               size_t pair_index,        // thread num
                               size_t pair_count,        // num threads
                               int mismatches,           // 5
                               size_t key_size_bytes,    // 32  (key_size)
                               size_t key_sz_bits        // 256 (key_sz_bits)
                             )
{
   uint64_t total_perms      = 0;
   uint64_t starting_ordinal = 0;
   uint64_t ending_ordinal   = 0;

   total_perms = get_bin_coef( key_sz_bits, mismatches );

   if( pair_index == 0 )
   {
      assign_first_permutation( starting_perm, mismatches );
   } 
   else
   {
      starting_ordinal = floorf( total_perms / pair_count ) * pair_index;

      decode_ordinal(starting_perm, starting_ordinal, mismatches, key_sz_bits);
   }

   if( pair_index == pair_count - 1 )
   {
      assign_last_permutation( ending_perm, mismatches, key_sz_bits );
   } 
   else
   {
      ending_ordinal = floorf( total_perms / pair_count ) * (pair_index + 1);
   
      decode_ordinal(ending_perm, ending_ordinal, mismatches, key_sz_bits);
   }
}

// Returns value of Binomial Coefficient C(n, k)  
// ref: https://www.geeksforgeeks.org/space-and-time-efficient-binomial-coefficient/
__device__ uint64_t get_bin_coef(size_t n, size_t k)
{  
    int ret = 1;  
  
    // Since C(n, k) = C(n, n-k)  
    if ( k > n - k )  
        k = n - k;  
  
    // Calculate value of  
    // [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]  
    // IMPLEMENT: pipelining technique to increase the number of operations per cycle
    // increases register usage (dropping occupancy) := unroll "4" specifically
    for (int i = 0; i < k; ++i)  
    {  
        ret *= (n - i);  
        ret /= (i + 1);  
    }  
  
    return ret;  
}  

// we don't need this here -- should be used in main before kernel invocation.
__device__ void get_random_permutation( uint256_t perm,
                                        int mismatches,
                                        int key_sz_bits )
{


}

// we don't need this here -- should be used in main before kernel invocation.
__device__ void get_benchmark_permutation( uint256_t perm,
                                           int mismatches,
                                           int key_sz_bits )
{


}
   
#endif

                             
