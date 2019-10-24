#ifndef UINT256_ITERATOR_HH_INCLUDED
#define UINT256_ITERATOR_HH_INCLUDED
#include "uint256_t.h"

class uint256_iter
{
 public:
    uint256_t curr_perm;
    uint256_t last_perm;
    uint256_t t;
    uint256_t tmp;
    uint256_t key;
    uint256_t corrupted_key;
    bool overflow;
};

#endif // UINT256_ITERATOR_HH_INCLUDED
