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

__device__ void uint256_iter::next()
{

    uint256_t add_tmp;
    curr_perm.add( add_tmp, UINT256_NEGATIVE_ONE );
    t = curr_perm | ( add_tmp );

    uint8_t shift = curr_perm.ctz() + 1;

    curr_perm = ~t;
    curr_perm.neg( tmp );
    tmp = curr_perm & tmp;

    tmp = tmp >> shift;

    overflow = t.add( t, UINT256_ONE );
    curr_perm = t | tmp;

    corrupted_key = key_uint ^ curr_perm;

}
