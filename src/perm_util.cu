// utility file for permutation delimination

#ifndef PERM_UTIL_CU_
#define PERM_UTIL_CU_

#include "uint256_t.h"

// COMPLETED
__device__ void decode_ordinal( uint256_t perm, 
                                const uint256_t ordinal, 
                                size_t mismatches, // 0-6
                                int key_sz_bits    // 256
                              )
{
   uint256_t binom, wkg_ordinal;
   uint64_t tmp_binom    = 0;
   uint64_t tmp_curr_ord = 0;
   wkg_ordinal.copy( ordinal );
   perm.set_all( 0 );

   for( size_t bit = key_sz_bits-1; mismatches > 0; bit-- )
   {
      tmp_binom = get_bin_coef( bit, mismatches );
      binom( 0 );
      binom( tmp_binom, 2 );

      if ( wkg_ordinal > binom || wkg_ordinal == binom )
      {
         wkg_ordinal = wkg_ordinal - binom;

         perm.set_bit( bit );

         mismatches--;
      }
   }
}

// COMPLETED
__device__ void assign_first_permutation( uint256_t *perm, int mismatches )
{
   // set perm to first key

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
   uint256_t total_perms();
   uint256_t starting_ordinal();
   uint256_t ending_ordinal();
   uint64_t tmp_tot_perms; 
   uint64_t tmp_starting_ord; 
   uint64_t tmp_ending_ord; 

   tmp_tot_perms = get_bin_coef( key_sz_bits, mismatches );
   total_perms( tmp_tot_perms, 2 ); 

   if( pair_index == 0 )
   {
      assign_first_permutation( starting_perm, mismatches );
   } 
   else
   {
      tmp_starting_ord = floor( tmp_tot_perms / pair_count );
      tmp_starting_ord = tmp_starting_ord * pair_index;
      // copy 64 bit tmp into uint256_t at index 2 
      // uint256_t is big endian - most significant byte first
      starting_ordinal( tmp_starting_ord, 2 );

      decode_ordinal(starting_perm, starting_ordinal, mismatches, key_sz_bits);
   }

   if( pair_index == pair_count - 1 )
   {
      assign_last_permutation( ending_perm, mismatches, key_sz_bits );
   } 
   else
   {
      tmp_ending_ord = floor( tmp_tot_perms / pair_count );
      tmp_starting_ord = tmp_ending_ord * (pair_index + 1);
      starting_ordinal( tmp_starting_ord, 2 ); // copy into uint256_t
   
      decode_ordinal(ending_perm, ending_ordinal, mismatches, key_sz_bits);
   }
}

// compute the binomial coefficient:
// get the number of k-element subsets of an n-element set
__device__ uint64_t get_bin_coef(size_t n, size_t r)
{
   int i;
   uint64_t b;

   if( (r < 0) || (n < r) ) 
      return 0;

   if( (2*r) > n ) 
      r = n-r;

   b = 1;

   if( r>0 )
   {
      for( i=0; i<=r-1; i++ )
      {
         b = ( b*(n-i) ) / (i+1);
      }
   }

   return b;
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

                             
