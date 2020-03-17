#ifndef UINT256_ITERATOR_HH_INCLUDED
#define UINT256_ITERATOR_HH_INCLUDED

#include "uint256_t.h"

class uint256_iter
{
 public:

    uint256_iter( const uint256_t& key,
                  const uint256_t& first_perm,
                  const uint256_t& final_perm
                );
    uint256_iter();

    void get( uint256_t& dest );

    void next();
    bool end();

    uint256_t curr_perm;
    uint256_t last_perm;
    uint256_t key_uint;
    uint256_t corrupted_key;
    bool overflow;
};

#endif // UINT256_ITERATOR_HH_INCLUDED
