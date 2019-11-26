#include "aes_per_round.h"

#define KEY_128_SIZE_IN_BYTES 16

namespace aes_per_round
{

CUDA_CALLABLE_MEMBER void shift_rows( message_128 *message )
{
        uint8_t temp;

        //shift row 2 by 1
        temp = message->bits[1];
        message->bits[1] = message->bits[5];
        message->bits[5] = message->bits[9];
        message->bits[9] = message->bits[13];
        message->bits[13] = temp;

        //shift row 3 by 2
        temp = message->bits[2];
        message->bits[2] = message->bits[10];
        message->bits[10] = temp;
        temp = message->bits[6];
        message->bits[6] = message->bits[14];
        message->bits[14] = temp;

        //shift row 4 by 3
        temp = message->bits[3];
        message->bits[3] = message->bits[15];
        message->bits[15] = message->bits[11];
        message->bits[11] = message->bits[7];
        message->bits[7] = temp;

    }

    CUDA_CALLABLE_MEMBER void xor_key( message_128 *message, const std::uint8_t *key )
    {

        for( std::uint8_t idx = 0; idx < KEY_128_SIZE_IN_BYTES; ++idx )
            {
                message->bits[ idx ] ^= key[ idx ];
            }

    }

    CUDA_CALLABLE_MEMBER uint8_t rcon(int in)
    {
        uint8_t c = 1;
        if(in == 0) {return 0;}
    
        while(in != 1) {
            uint8_t b;
            b = c & 0x80;
            c <<= 1;
            if(b == 0x80) {
                c ^= 0x1b;
            }
            in--;
        }
        return c;
    }

    //from https://www.samiam.org/key-schedule.html
    CUDA_CALLABLE_MEMBER void rotate( uint8_t in[ 4 ] )
    {
        uint8_t a,c;
        a = in[0];
        for(c=0;c<3;c++) 
            in[c] = in[c + 1];
        in[3] = a;
        return;
    }

    /* This is the core key expansion, which, given a 4-byte value,
     * does some scrambling from https://www.samiam.org/key-schedule.html*/
    CUDA_CALLABLE_MEMBER
    void schedule_core( uint8_t in[ 4 ], uint8_t i, const uint8_t sbox[ 256 ] )
    {
        uint8_t a;
        /* Rotate the input 8 bits to the left */
        rotate(in);
        /* Apply Rijndael's s-box on all 4 bytes */
        for( a = 0; a < 4; a++ )
            {
                uint8_t less_nibble = in[a] & 0x0f;
                uint8_t more_nibble = (in[a] & 0xf0) >> 4;
                // in[a] = sbox[less_nibble + more_nibble*16];
                in[a] = sbox[less_nibble + more_nibble*16];
            } 
        /* On just the first byte, add 2^i to the byte */
        in[ 0 ] ^= rcon( i );
    }

    //Straight from Wikiepedia on Rijndael S-box
    CUDA_CALLABLE_MEMBER void initialize_sbox( uint8_t sbox[ 256 ] )
    {
        uint8_t p = 1, q = 1;
    
        /* loop invariant: p * q == 1 in the Galois field */
        do {
            /* multiply p by 3 */
            p = p ^ ( p << 1 ) ^ ( p & 0x80 ? 0x1B : 0 );

            /* divide q by 3 (equals multiplication by 0xf6) */
            q ^= q << 1;
            q ^= q << 2;
            q ^= q << 4;
            q ^= q & 0x80 ? 0x09 : 0;

            /* compute the affine transformation */
            uint8_t xformed = q ^ ROTL8(q, 1) ^ ROTL8(q, 2) ^ ROTL8(q, 3) ^ ROTL8(q, 4);

            sbox[ p ] = xformed ^ 0x63;
        } while ( p != 1 );

        /* 0 is a special case since it has no inverse */
        sbox[ 0 ] = 0x63;
    }

    //From Wikepedia Rijndael mix columns
    CUDA_CALLABLE_MEMBER
    void gmix_column( uint8_t r[ 4 ] )
    {
        uint8_t a[ 4 ];
        uint8_t b[ 4 ];
        uint8_t c;
        uint8_t h;
        /* The array 'a' is simply a copy of the input array 'r'
         * The array 'b' is each element of the array 'a' multiplied by 2
         * in Rijndael's Galois field
         * a[n] ^ b[n] is element n multiplied by 3 in Rijndael's Galois field */ 
        for ( c = 0; c < 4; c++ ) {
            a[ c ] = r[ c ];
            /* h is 0xff if the high bit of r[c] is set, 0 otherwise */
            h = (uint8_t)((signed char)r[ c ] >> 7); /* arithmetic right shift, thus shifting in either zeros or ones */
            b[ c ] = r[ c ] << 1; /* implicitly removes high bit because b[c] is an 8-bit char, so we xor by 0x1b and not 0x11b in the next line */
            b[ c ] ^= 0x1B & h; /* Rijndael's Galois field */
        }

        r[ 0 ] = b[ 0 ] ^ a[ 3 ] ^ a[ 2 ] ^ b[ 1 ] ^ a[ 1 ]; /* 2 * a0 + a3 + a2 + 3 * a1 */
        r[ 1 ] = b[ 1 ] ^ a[ 0 ] ^ a[ 3 ] ^ b[ 2 ] ^ a[ 2 ]; /* 2 * a1 + a0 + a3 + 3 * a2 */
        r[ 2 ] = b[ 2 ] ^ a[ 1 ] ^ a[ 0 ] ^ b[ 3 ] ^ a[ 3 ]; /* 2 * a2 + a1 + a0 + 3 * a3 */
        r[ 3 ] = b[ 3 ] ^ a[ 2 ] ^ a[ 1 ] ^ b[ 0 ] ^ a[ 0 ]; /* 2 * a3 + a2 + a1 + 3 * a0 */
    }

    CUDA_CALLABLE_MEMBER
    void mix_columns( message_128 *message )
    {
        //mix each set of 4 bytes
        message_128 temp = *message;
    
        uint8_t r[ 4 ];

        //first col
        r[ 0 ] = temp.bits[ 0 ];
        r[ 1 ] = temp.bits[ 1 ];
        r[ 2 ] = temp.bits[ 2 ];
        r[ 3 ] = temp.bits[ 3 ];

        gmix_column( r );

        message->bits[ 0 ] = r[ 0 ];
        message->bits[ 1 ] = r[ 1 ];
        message->bits[ 2 ] = r[ 2 ];
        message->bits[ 3 ] = r[ 3 ];

        //sec col
        r[ 0 ] = temp.bits[ 4 ];
        r[ 1 ] = temp.bits[ 5 ];
        r[ 2 ] = temp.bits[ 6 ];
        r[ 3 ] = temp.bits[ 7 ];

        gmix_column( r );

        message->bits[ 4 ] = r[ 0 ];
        message->bits[ 5 ] = r[ 1 ];
        message->bits[ 6 ] = r[ 2 ];
        message->bits[ 7 ] = r[ 3 ];

        //third col
        r[ 0 ] = temp.bits[ 8 ];
        r[ 1 ] = temp.bits[ 9 ];
        r[ 2 ] = temp.bits[ 10 ];
        r[ 3 ] = temp.bits[ 11 ];

        gmix_column( r );

        message->bits[ 8 ] = r[ 0 ];
        message->bits[ 9 ] = r[ 1 ];
        message->bits[ 10 ] = r[ 2 ];
        message->bits[ 11 ] = r[ 3 ];

        //fourth col
        r[ 0 ] = temp.bits[ 12 ];
        r[ 1 ] = temp.bits[ 13 ];
        r[ 2 ] = temp.bits[ 14 ];
        r[ 3 ] = temp.bits[ 15 ];

        gmix_column( r );

        message->bits[ 12 ] = r[ 0 ];
        message->bits[ 13 ] = r[ 1 ];
        message->bits[ 14 ] = r[ 2 ];
        message->bits[ 15 ] = r[ 3 ];
    }

    CUDA_CALLABLE_MEMBER
    void sub_bytes( message_128 *message, const uint8_t sbox[ 256 ] )
    {
        //take the bytes and seperate the nibbles out
        for ( int i = 0; i < 16; i++ ){
            uint8_t less_nibble  = message->bits[ i ] & 0x0f;
            uint8_t more_nibble = ( message->bits[ i ] & 0xf0 ) >> 4;
            message->bits[ i ] = sbox[ less_nibble + more_nibble*16 ];
        } 
    }

    DEVICE_ONLY
    void roundwise_encrypt( message_128 *dest,
                            const uint256_t *key,
                            const message_128 *message,
                            const std::uint8_t sbox[ SBOX_SIZE_IN_BYTES ]
                          )
    {

        std::uint8_t round_keys[ 48 ];

        std::uint8_t round_no = 0;
        std::uint8_t idx      = 0;
        std::uint8_t i        = 0;

        for( i = 0; i < 16; ++i )
            {
                dest->bits[ i ] = message->bits[ i ];
            }

        for( i = 0; i < 32; ++i )
            {
                round_keys[ i ] = (*key)[ i ];
            }

        xor_key( dest, round_keys );

        // do AES for the first round
        do_aes_round( dest,
                      sbox,
                      round_keys + 16
                    );

        // start at 2 because we have the first two keys
        for( round_no = 2; round_no < 14; ++round_no )
            {
                // get the key for the round
                get_round_key( round_keys,
                               sbox,
                               &i,
                               round_no
                             );

                // do aes for the round
                do_aes_round( dest,
                              sbox,
                              round_keys + 32
                            );

                idx = 0;
                // reset the round key
                for( idx = 0; idx < 16; ++idx )
                    {
                        // shift the bytes over by 16
                        round_keys[ idx ] = round_keys[ idx + 16 ];
                        round_keys[ idx + 16 ] = round_keys[ idx + 32 ];
                        round_keys[ idx + 32 ] = 0;
                    }
            }

        // roundno = 14
        get_round_key( round_keys, sbox, &i, round_no );

        sub_bytes( dest, sbox );
        shift_rows( dest );
        xor_key( dest, round_keys + 32 );

    }

    DEVICE_ONLY
    void get_round_key( std::uint8_t keys[ 48 ],
                        const std::uint8_t sbox[ SBOX_SIZE_IN_BYTES ],
                        std::uint8_t *i,
                        int round_no
                        )
    {
        std::uint8_t t[ 4 ];
        std::uint8_t c = 32;
        std::uint8_t a = 0;
        // first 16 bytes:  current key
        // second 16 bytes: previous key
        while( c < 48 )
            {

                for( a = 0; a < 4; ++a )
                    {
                        t[ a ] = keys[ a + c - 4 ];
                    }

                // NOTE: All threads in a warp will take the same path
                // for this branch
                if( !( round_no % 2 // even round number and first iteration
                       ||  c % 32
                     )
                  )
                    {
                        schedule_core( t, *i, sbox );
                        ++(*i);

                    }
                else if( round_no % 2 // odd round number and first iteration
                         && !( c % 32 )
                       )
                    {
                        for( a = 0; a < 4; ++a )
                            {
                                uint8_t less_nibble = t[ a ] & 0x0F;
                                uint8_t more_nibble = ( t[ a ] & 0xF0 ) >> 4;

                                t[ a ] = sbox[ less_nibble + more_nibble*16 ];

                            }
                    }

                for( a = 0; a < 4; ++a )
                    {
                        keys[ c ] = keys[ c - 32 ] ^ t[ a ];
                        ++c;
                    }
            }
    }

    DEVICE_ONLY INLINE void do_aes_round( message_128 *message,
                                          const std::uint8_t sbox[ SBOX_SIZE_IN_BYTES ],
                                          const std::uint8_t *key
                                        )
    {
        sub_bytes( message, sbox );
        shift_rows( message );
        mix_columns( message );
        xor_key( message, key );
    }

};
