// utility file for permutation delimination

#include "perm_util.h"

__device__ __host__ void decode_ordinal( uint256_t *perm, const uint64_t ordinal, int mismatches )
{
   uint64_t binom = 0;
   uint64_t wkg_ord = ordinal;

   for( uint8_t bit = UINT256_SIZE_IN_BITS-1; mismatches > 0; bit-- )
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

__device__ void assign_first_permutation( uint256_t *perm, int mismatches )
{
   for( int i=0; i<mismatches; ++i ) perm->set_bit( i );
}

__device__ void assign_last_permutation( uint256_t *perm, int mismatches )
{
   assign_first_permutation(perm, mismatches);

   *perm = *perm << (UINT256_SIZE_IN_BITS - mismatches);
}

__device__ void get_perm_pair( uint256_t *starting_perm, 
                               uint256_t *ending_perm,
                               const uint64_t tid,        
                               const uint64_t num_threads,
                               const uint8_t mismatches,           
                               const std::uint64_t keys_per_thread,
                               const std::uint32_t extra_keys
                             )
{
   uint64_t strt_ordinal   = 0;
   uint64_t ending_ordinal = 0;

    if( tid < extra_keys )
    {
        if( tid == 0 )
        {
           assign_first_permutation(starting_perm, mismatches);
        } 
        else
        {
           strt_ordinal = ( keys_per_thread + 1 ) * tid;

           decode_ordinal(starting_perm, strt_ordinal, mismatches);
        }

        ending_ordinal = strt_ordinal + keys_per_thread;
        
        decode_ordinal(ending_perm, ending_ordinal, mismatches);
    }
    else
    {
        strt_ordinal = ( keys_per_thread * tid ) + extra_keys;

        decode_ordinal(starting_perm, strt_ordinal, mismatches);

        if( tid == num_threads - 1 )
        {
           assign_last_permutation(ending_perm, mismatches);
        } 
        else
        {
           ending_ordinal = strt_ordinal + ( keys_per_thread - 1 );
        
           decode_ordinal(ending_perm, ending_ordinal, mismatches);
        }
    }
}

CUDA_CALLABLE_MEMBER uint64_t get_bin_coef(uint16_t n, uint16_t k)
{  
    uint64_t ret = 1;  
  
    // Since C(n, k) = C(n, n-k)  
    if( k > n-k )  
        k = n-k;  

    for( int i=0; i<k; ++i )  
    {  
        ret *= (n-i);  
        ret /= (i+1);  
    }

    return ret;  
}  


