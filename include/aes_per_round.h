#ifndef AES_PER_ROUND_HH_INCLUDED
#define AES_PER_ROUND_HH_INCLUDED
#include <cstdint>

#include "uint256_t.h"
#include "cuda_defs.h"
#include "aes_tables.h"
#include "sbox.h"
#define ROTL8(x,shift) ((uint8_t) ((x) << (shift)) | ((x) >> (8 - (shift))))
#define SBOX_SIZE_IN_BYTES 256

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

    typedef struct key_256
    {
        std::uint8_t bits[ 32 ];
    } key_256;


    CUDA_CALLABLE_MEMBER void shift_rows( message_128 *message );
    CUDA_CALLABLE_MEMBER void shift_mix( message_128 *message );

    CUDA_CALLABLE_MEMBER void xor_key( message_128 *message, const std::uint8_t *key );

    CUDA_CALLABLE_MEMBER uint8_t rcon( int in );

    CUDA_CALLABLE_MEMBER void rotate( uint8_t in[ 4 ] );

    CUDA_CALLABLE_MEMBER
        void schedule_core( uint8_t in[ 4 ], uint8_t i, const uint8_t sbox[ SBOX_SIZE_IN_BYTES ] );

    CUDA_CALLABLE_MEMBER void initialize_sbox( uint8_t sbox[ SBOX_SIZE_IN_BYTES ] );

    CUDA_CALLABLE_MEMBER
        void gmix_column( uint8_t r[ 4 ] );

    CUDA_CALLABLE_MEMBER
        void mix_columns( message_128 *message );

    CUDA_CALLABLE_MEMBER
        void sub_bytes( message_128 *message, uint8_t sbox[ SBOX_SIZE_IN_BYTES ] );

    DEVICE_ONLY
        void roundwise_encrypt( message_128 *dest,
                                const uint256_t *key,
                                const message_128 *message,
                                const uint8_t sbox[ SBOX_SIZE_IN_BYTES ]
                              );

    DEVICE_ONLY INLINE
        void do_aes_round( message_128 *message,
                           const std::uint8_t sbox[ SBOX_SIZE_IN_BYTES ],
                           const std::uint8_t *key
                         );

    DEVICE_ONLY
        void get_round_key( std::uint8_t keys[ 48 ],
                            const std::uint8_t sbox[ SBOX_SIZE_IN_BYTES ],
                            std::uint8_t *i,
                            int round_no
                          );


}; // namespace aes_per_round 


namespace aes_gpu
{
    DEVICE_ONLY void expand_key( uint *expanded_loc,
                                 uint *cipherKey
                               );

    DEVICE_ONLY void encrypt( const uint pt[4], uint ct[4], uint *rek,
                              const aes_tables *aes_tabs
                            );

};

#endif // AES_PER_ROUND_HH_INCLUDED 
