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
    overflow = false;
}

uint256_iter::uint256_iter()
:
    curr_perm( 0x00 ), last_perm( 0x00 ), key_uint( 0x00 ), corrupted_key( 0x00 ), overflow( false ) {}

void uint256_iter::next()
{

    uint256_t t = curr_perm | ( curr_perm + UINT256_NEGATIVE_ONE );


    uint8_t shift = curr_perm.ctz() + 1;

    // add_tmp.set_all( 0x00 );
    uint256_t tmp;

    overflow = t.add( tmp, UINT256_ONE );

    curr_perm = (tmp) | ((((~t) & -(~t)) + UINT256_NEGATIVE_ONE ) >> shift ); 
    corrupted_key = key_uint ^ curr_perm;
}

void uint256_iter::get( uint256_t& dest )
{
    dest = corrupted_key;
}

bool uint256_iter::end()
{
    return curr_perm.compare( last_perm ) > 0 || overflow ;
}
