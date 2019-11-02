// utility file for permutation delimination

#ifndef PERM_UTIL_CU_
#define PERM_UTIL_CU_

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 30
#endif
#define NUM_THREADS 1024
#define MAX_RK_SIZE 182

#include "uint256_t.h"

__device__ void decode_ordinal( uint256_t perm, 
                                const uint256_t ordinal, 
                                int mismatches, 
                                int subkey_length )
{
   uint256_t binom, curr_ordinal;   
   // TODO: create copy constructor for uint256_t class
   // Does not work as is: 
   // curr_ordinal.set_all(ordinal);   
   // perm.set_all(0);

   for( unsigned long bit = subkey_length-1; mismatches > 0; bit-- )
   {
      // compute the binomial coefficient n over k 
      // TODO: create binom_coef method for uint256_t class
   }
}
                            
__device__ void get_random_permutation( uint256_t perm,
                                        int mismatches,
                                        int subkey_length )
{


}

__device__ void get_benchmark_permutation( uint256_t perm,
                                           int mismatches,
                                           int subkey_length )
{


}

__device__ void assign_first_permutation( uint256_t *perm, int mismatches )
{
   // may need to store shift result in new data, then copy over to perm
   perm = perm << mismatches; 
}

__device__ void assign_last_permutation( uint256_t *perm,
                                         int mismatches,
                                         int subkey_length )
{
   // First set the value to the first permutation
   assign_first_permutation( perm, mismatches );

   // Equiv to: perm << (key_length - mismatches)
   // E.g. if key_length = 256 and mismatches = 5,
   // then we want to shift left 256 - 5 = 251 times.
   perm = perm << (subkey_length - mismatches);
}

                                
