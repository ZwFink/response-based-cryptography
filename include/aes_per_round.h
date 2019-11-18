#ifndef AES_PER_ROUND_HH_INCLUDED
#define AES_PER_ROUND_HH_INCLUDED
#include <cstdint>

#include "uint256_t.h"
#include "cuda_defs.h"

#define ROTL8(x,shift) ((uint8_t) ((x) << (shift)) | ((x) >> (8 - (shift))))

namespace aes_per_round
{
    typedef struct key_128
    {
        std::uint8_t bits[ 16 ];

    } key_128;

    typedef struct message_128
    {
        std::uint8_t bits[ 16 ];

    } message_128;


    CUDA_CALLABLE_MEMBER void shift_rows( message_128 *message );
    


}; // namespace aes_per_round 

#endif // AES_PER_ROUND_HH_INCLUDED 
