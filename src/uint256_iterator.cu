#include "uint256_iterator.h"

CUDA_CALLABLE_MEMBER uint256_iter::uint256_iter( const unsigned char *key,
                                                 const uint256_t& first_perm,
                                                 const uint256_t& final_perm
                                               )
{
    curr_perm = first_perm;
    last_perm = final_perm;
    key_uint.from_string( key );
    corrupted_key = key_uint ^ curr_perm;
}