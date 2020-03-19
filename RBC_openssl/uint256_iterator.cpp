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
    uint256_t aug( 0 );
    curr_perm.add(&aug, UINT256_NEGATIVE_ONE);

    uint256_t t = curr_perm | aug;


    uint8_t shift = curr_perm.ctz() + 1;

    uint256_t tmp( 0 );

    overflow = t.add( &tmp, UINT256_ONE );

    uint256_t tmp2( 0 );
    uint256_t tmp3( 0 );
    uint256_t tmp4( 0 );

    tmp2 = (~t) & -(~t);
    tmp2.add(&tmp3,UINT256_NEGATIVE_ONE);
    tmp4 = tmp3 >> shift;
    curr_perm = tmp | tmp4;

    //curr_perm = (tmp) | ((((~t) & -(~t)) + UINT256_NEGATIVE_ONE ) >> shift); 
    corrupted_key = key_uint ^ curr_perm;
}

void uint256_iter::get( uint256_t& dest )
{
    dest = corrupted_key;
}

bool uint256_iter::end()
{
    //if (curr_perm.compare( last_perm ) > 0)
    //{
    //    curr_perm.dump();
    //    last_perm.dump();
    //}
    return curr_perm.compare( last_perm ) > 0 || overflow;
}
