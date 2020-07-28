/*
Description: This is an iterator class for iterating keys of type uint256_t.
             The iteration process is specific to the RBC scheme. That is,
             given a hamming distance, h, and a starting key of size 256 bits, 
             this class can be used to iterate across all keys that are exactly 
             a hamming distance, h, away from the starting key.   
*/

#include "uint256_iterator.h"

uint256_iter::uint256_iter( const uint256_t& key,
                            const uint256_t& first_perm,
                            const uint256_t& final_perm
                          )
{
    curr_perm = first_perm;
    last_perm = final_perm;
    key_uint = key;
    corrupted_key = key_uint ^ curr_perm;
    overflow = 0;
}

uint256_iter::uint256_iter()
:
    curr_perm( 0x00 ), last_perm( 0x00 ), key_uint( 0x00 ), corrupted_key( 0x00 ), overflow( 0 ) {}

void uint256_iter::next()
{
    uint256_t aug;
    curr_perm.add(&aug, UINT256_NEGATIVE_ONE);
    uint256_t t = curr_perm | aug;

    uint8_t shift = curr_perm.ctz() + 1;

    uint256_t tmp1;
    overflow = t.add( &tmp1, UINT256_ONE );

    uint256_t tmp2;
    tmp2 = (~t) & -(~t);
    tmp2.add(&tmp2,UINT256_NEGATIVE_ONE);
    tmp2 = tmp2 >> shift;

    curr_perm = tmp1 | tmp2;

    corrupted_key = key_uint ^ curr_perm;
}

void uint256_iter::get( uint256_t& dest )
{
    dest = corrupted_key;
}

bool uint256_iter::end()
{
    return curr_perm.compare( last_perm ) > 0 || overflow;
}
