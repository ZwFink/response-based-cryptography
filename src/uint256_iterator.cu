#include "uint256_iterator.h"

CUDA_CALLABLE_MEMBER uint256_iter::uint256_iter( const uint256_t& key,
                                                 const uint256_t& first_perm,
                                                 const uint256_t& final_perm
                                               )
{
    curr_perm = first_perm;
    last_perm = final_perm;
    key_uint = key;
    corrupted_key = key_uint ^ curr_perm;
    overflow = false;
    tmp.set_all( 0x00 );
}

CUDA_CALLABLE_MEMBER uint256_iter::uint256_iter()
:
    curr_perm( 0x00 ), last_perm( 0x00 ), key_uint( 0x00 ), corrupted_key( 0x00 ), overflow( false ) {}

__device__ void uint256_iter::next()
{

    tmp.set_all( 0x00 );
    // t = curr | ( curr - 1 )
    uint256_t add_tmp( 0x00 );
    curr_perm.add( add_tmp, UINT256_NEGATIVE_ONE ); // curr - 1 

    t = curr_perm | ( add_tmp ); // curr | ( curr - 1 )


    uint8_t shift = curr_perm.ctz() + 1;

    add_tmp.set_all( 0x00 );


   // ( t + 1 ) | ( ( ( ~t & -~t ) - 1 ) >> ( ctz( perm ) + 1 ) )

    // ~t & -~t
    tmp = ~t; // ~t
    tmp.neg( add_tmp ); // -~t
    tmp = tmp & add_tmp; // ~t & -~t

    add_tmp.set_all( 0x00 );

    // - 1
    tmp.add( add_tmp, UINT256_NEGATIVE_ONE ); // ( ~t & -~t ) - 1

    // >> ctz( perm ) + 1
    add_tmp = add_tmp >> shift;

    tmp.set_all( 0x00 );

    // ( t + 1 )
    overflow = t.add( tmp, UINT256_ONE );

    curr_perm = tmp | add_tmp;

    corrupted_key = key_uint ^ curr_perm;
}

__device__ void uint256_iter::get( uint256_t& dest )
{
    dest = corrupted_key;
}

__device__ bool uint256_iter::end()
{
    return curr_perm.compare( last_perm ) > 0 || overflow;
}