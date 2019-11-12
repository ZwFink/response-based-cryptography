// utility file for permutation delimination

#ifndef PERM_UTIL_CU_
#define PERM_UTIL_CU_

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 30
#endif
#define NUM_THREADS 1024
#define MAX_RK_SIZE 182

#include "uint256_t.h"

// NOT NEEDED
// macro for min of two ints; used in binomial function
#define min(a, b) ((a) > (b))? (b): (a)

__device__ void decode_ordinal( uint256_t perm, 
                                const uint256_t ordinal, 
                                int mismatches, 
                                int subkey_length )
{
   uint256_t binom, curr_ordinal;   
   curr_ordinal.copy( ordinal );

   for( int bit = subkey_length-1; mismatches > 0; bit-- )
   {
      // compute the binomial coefficient n over k 
      // TODO: create binom_coef method for uint256_t class
      
   }
}

// COMPLETE
__device__ void assign_first_permutation( uint256_t *perm, int mismatches )
{
   // set perm to first key

   *perm = *perm << mismatches; // shift left
	
   perm->add( *perm, UINT256_NEGATIVE_ONE ); // add negative one
}

// COMPLETE
__device__ void assign_last_permutation( uint256_t *perm,
                                         int mismatches,
                                         int subkey_length )
{

   // set perm to the first key
   assign_first_permutation( perm, mismatches );

   // Then find the last key by shifting the first
   // Equiv to: perm << (key_length - mismatches)
   // E.g. if key_length = 256 and mismatches = 5,
   //      we want to shift left 256 - 5 = 251 times.
   *perm = *perm << (subkey_length - mismatches);
}

__device__ void get_perm_pair( uint256_t *starting_perm, 
                               uint256_t *ending_perm,
                               size_t pair_index,        // thread num
                               size_t pair_count,        // num threads
                               int mismatches,           // 5
                               size_t key_size_in_bytes, // 32
                               size_t key_size_in_bits   // 256 
                             )
{
   uint256_t total_perms();
   uint256_t starting_ordinal();
   uint256_t ending_ordinal();

   // TODO: total_perms = binomial_coef(subkey_length, mismatches)

   if( pair_index == 0 )
   {
      assig// C++ program for binomial coefficients 
#include <stdio.h> 
  
// Function to print binomial table 
int printbinomial(int max) 
{ 
    for (int m = 0; m <= max; m++) { 
        printf("%2d", m); 
        int binom = 1; 
        for (int x = 0; x <= m; x++) { 
  
            // B(m, x) is 1 if either m or x is 
            // is 0. 
            if (m != 0 && x != 0) 
  
                // Otherwise using recursive formula 
                // B(m, x) = B(m, x - 1) * (m - x + 1) / x 
                binom = binom * (m - x + 1) / x; 
  
            printf("%4d", binom); 
        } 
        printf("\n"); 
    } 
} 
  
// Driver Function 
int main() 
{ 
    int max = 10; 
    printbinomial(max); 
    return 0; 
} n_first_permutation( starting_perm, mismatches );
   } 
   else
   {
      
   }
}
// compute the binomial coefficient:
// get the number of k-element subsets of an n-element set
__device__ uint256_t get_bin_coef( int n, int k )  
{
   cudaError_t result = cudaSuccess;    
   uint256_t ret = UINT256_ZERO;
   //int sz = k + 1;
   //const C[sz];

   //C[0] = 1;

   //for( int i=1; i <= n; i++ )
   //{
   //   for( int j=min(i,k); j>0; j-- )
   //   {  
   //      C[j] = C[j] + C[j-1];
   //   }
   //}
   //
   ////ret(C[k], 0);   

   return ret; 
}

// we don't need this here -- should be used in main before kernel invocation.               
__device__ void get_random_permutation( uint256_t perm,
                                        int mismatches,
                                        int subkey_length )
{


}

// we don't need this here -- should be used in main before kernel invocation.
__device__ void get_benchmark_permutation( uint256_t perm,
                                           int mismatches,
                                           int subkey_length )
{


}
   
#endif

                             
