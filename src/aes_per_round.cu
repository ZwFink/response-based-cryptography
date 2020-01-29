#include "aes_per_round.h"
#include "util.h"

#define GETWORD(pt) (((uint)(pt)[0] << 24) ^ ((uint)(pt)[1] << 16) ^ ((uint)(pt)[2] <<  8) ^ ((uint)(pt)[3]))
#define PUTWORD(ct, st) ((ct)[0] = (uchar)((st) >> 24), (ct)[1] = (uchar)((st) >> 16), (ct)[2] = (uchar)((st) >>  8), (ct)[3] = (uchar)(st), (st))
#define INTERPRET_UINT32(x)((std::uint32_t*)(x))

#define KEY_128_SIZE_IN_BYTES 16

namespace aes_per_round
{

    
    CUDA_CALLABLE_MEMBER INLINE void shift_mix( message_128 *message )
    {
        uint8_t a[4];
        uint8_t b[4];
        uint8_t c[4];
        uint8_t d[4];

        //first col

        a[0] = message->bits[0]; //0
        a[1] = message->bits[5]; //1
        a[2] = message->bits[10]; //2
        a[3] = message->bits[15]; //3

        //sec col
        b[0] = message->bits[4]; //4
        b[1] = message->bits[9]; //5
        b[2] = message->bits[14]; //6
        b[3] = message->bits[3]; //7

        //third col
        c[0] = message->bits[8]; //8
        c[1] = message->bits[13]; //9
        c[2] = message->bits[2]; //10
        c[3] = message->bits[7]; //11

        //fourth col
        d[0] = message->bits[12]; //12
        d[1] = message->bits[1]; //13
        d[2] = message->bits[6]; //14
        d[3] = message->bits[11]; //15

        gmix_column(a);
        gmix_column(b);
        gmix_column(c);
        gmix_column(d);

        message->bits[0] = a[0];
        message->bits[1] = a[1];
        message->bits[2] = a[2];
        message->bits[3] = a[3];

        message->bits[4] = b[0];
        message->bits[5] = b[1];
        message->bits[6] = b[2];
        message->bits[7] = b[3];

        message->bits[8] = c[0];
        message->bits[9] = c[1];
        message->bits[10] = c[2];
        message->bits[11] = c[3];

        message->bits[12] = d[0];
        message->bits[13] = d[1];
        message->bits[14] = d[2];
        message->bits[15] = d[3];


    }

CUDA_CALLABLE_MEMBER INLINE void shift_rows( message_128 *message )
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

    CUDA_CALLABLE_MEMBER INLINE uint8_t rcon(int in)
    {
        // uint8_t c = 1;
        // if(in == 0) {return 0;}
    
        // while(in != 1) {
        //     uint8_t b;
        //     b = c & 0x80;
        //     c <<= 1;
        //     if(b == 0x80) {
        //         c ^= 0x1b;
        //     }
        //     in--;
        // }
        // return c;
        return Rcon[ in ];
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
    CUDA_CALLABLE_MEMBER INLINE
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
    CUDA_CALLABLE_MEMBER INLINE
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

        // uint8_t temp[ 4 ];

        // for (std::uint8_t i = 0; i < 4; ++i)
        //     {
        //         temp[0] = (unsigned char) (mul2[r[0]] ^ mul_3[r[1]] ^ r[2] ^ r[3]);
        //         temp[1] = (unsigned char) (r[0] ^ mul2[r[1]] ^ mul_3[r[2]] ^ r[3]);
        //         temp[2] = (unsigned char) (r[0] ^ r[1] ^ mul2[r[2]] ^ mul_3[r[3]]);
        //         temp[3] = (unsigned char) (mul_3[r[0]] ^ r[1] ^ r[2] ^ mul2[r[3]]);
        //     }


        // r[ 0 ] = temp[ 0 ];
        // r[ 1 ] = temp[ 1 ];
        // r[ 2 ] = temp[ 2 ];
        // r[ 3 ] = temp[ 3 ];

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

    DEVICE_ONLY INLINE
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
        shift_mix( message );
        xor_key( message, key );
    }

};

namespace aes_gpu
{
    DEVICE_ONLY void expand_key( uint *expanded_loc,
                                 std::uint8_t i
                               )
    {
        uint temp;
        // expanded_loc[0] = GETWORD(cipherKey     );
        // expanded_loc[1] = GETWORD(cipherKey +  4);
        // expanded_loc[2] = GETWORD(cipherKey +  8);
        // expanded_loc[3] = GETWORD(cipherKey + 12);

        // expanded_loc[4] = GETWORD(cipherKey + 16);
        // expanded_loc[5] = GETWORD(cipherKey + 20);

        // expanded_loc[6] = GETWORD(cipherKey + 24);
        // expanded_loc[7] = GETWORD(cipherKey + 28);
        temp = expanded_loc[ 7 ];
        expanded_loc[ 8] = expanded_loc[ 0] ^
            (cTe4[(temp >> 16) & 0xff] & 0xff000000) ^
            (cTe4[(temp >>  8) & 0xff] & 0x00ff0000) ^
            (cTe4[(temp      ) & 0xff] & 0x0000ff00) ^
            (cTe4[(temp >> 24)       ] & 0x000000ff) ^
            Rcon[i];
        expanded_loc[ 9] = expanded_loc[ 1] ^ expanded_loc[ 8];
        expanded_loc[10] = expanded_loc[ 2] ^ expanded_loc[ 9];
        expanded_loc[11] = expanded_loc[ 3] ^ expanded_loc[10];
        // if (++i == 7) {
        //     return;
        // }
        temp = expanded_loc[11];
        expanded_loc[12] = expanded_loc[ 4] ^
            (cTe4[(temp >> 24)       ] & 0xff000000) ^
            (cTe4[(temp >> 16) & 0xff] & 0x00ff0000) ^
            (cTe4[(temp >>  8) & 0xff] & 0x0000ff00) ^
            (cTe4[(temp      ) & 0xff] & 0x000000ff);
        expanded_loc[13] = expanded_loc[ 5] ^ expanded_loc[12];
        expanded_loc[14] = expanded_loc[ 6] ^ expanded_loc[13];
        expanded_loc[15] = expanded_loc[ 7] ^ expanded_loc[14];
        // expanded_loc += 8;
    }

    DEVICE_ONLY void encrypt( const uint pt[4], uint ct[4], uint *rek,
                              const aes_tables *aes_tabs
                            )
    {
        #define Nr 14
        using namespace aes_per_round;

        std::uint8_t round_keys[ 64 ];
        std::uint8_t i        = 0;
        std::uint8_t idx = 0;

        // get the first two round keys as
        // given by the starting key
        for( i = 0; i < 8; ++i )
            {
                ( (uint32_t*)round_keys )[ i ]
                    = ( (uint32_t*) rek )[ i ];
            }


        i = 0;

        uint s0, s1, s2, s3, t0, t1, t2, t3;

        #define Te0 aes_tabs->Te0
        #define Te1 aes_tabs->Te1
        #define Te2 aes_tabs->Te2
        #define Te3 aes_tabs->Te3
        #define sbox aes_tabs->sbox

        /*
         * map byte array block to cipher state
         * and add initial round key:
         */
        s0 = pt[ 0] ^ INTERPRET_UINT32( round_keys )[0];
        s1 = pt[ 1] ^ INTERPRET_UINT32( round_keys )[1];
        s2 = pt[ 2] ^ INTERPRET_UINT32( round_keys )[2];
        s3 = pt[ 3] ^ INTERPRET_UINT32( round_keys )[3];

        /* round 1: */
        t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff]
             ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff]
             ^ INTERPRET_UINT32( round_keys )[4];
        t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff]
             ^ Te2[(s3 >>  8) & 0xff]
             ^ Te3[s0 & 0xff]
             ^ INTERPRET_UINT32( round_keys )[5];
        t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff]
             ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff]
             ^ INTERPRET_UINT32( round_keys )[ 6 ];
        t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff]
             ^ Te2[(s1 >>  8) & 0xff]
             ^ Te3[s2 & 0xff]
             ^ INTERPRET_UINT32( round_keys )[ 7 ];

        // get the key for the round

        expand_key( INTERPRET_UINT32( round_keys) , i );
        
        /* round 2: */
        s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 8 ];
        s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 9 ];
        s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 10 ];
        s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 11 ];

        /* round 3: */
        t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 12 ];
        t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 13 ];
        t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 14 ];
        t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 15 ];

        idx = 0;
        // reset the round key
        for( idx = 0; idx < 8; ++idx )
            {
                // shift the bytes over by 16
                INTERPRET_UINT32( round_keys )[ idx ]
                    = INTERPRET_UINT32( round_keys )[ idx + 8 ];
                INTERPRET_UINT32( round_keys )[ idx + 8 ]
                    = INTERPRET_UINT32( round_keys )[ idx + 16 ];

            }

        // get the key for the round
        expand_key( INTERPRET_UINT32( round_keys) , ++i );


        /* round 4: */
        s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 8 ];
        s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 9 ];
        s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 10 ];
        s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 11 ];

        /* round 5: */
        t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 12 ];
        t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 13 ];
        t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 14 ];
        t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 15 ];

        idx = 0;
        // reset the round key
        for( idx = 0; idx < 8; ++idx )
            {
                // shift the bytes over by 16
                INTERPRET_UINT32( round_keys )[ idx ]
                    = INTERPRET_UINT32( round_keys )[ idx + 8 ];
                INTERPRET_UINT32( round_keys )[ idx + 8 ]
                    = INTERPRET_UINT32( round_keys )[ idx + 16 ];

            }

        // get the key for the round
        expand_key( INTERPRET_UINT32( round_keys) , ++i );


        /* round 6: */
        s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 8 ];
        s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 9 ];
        s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 10 ];
        s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 11 ];

        /* round 7: */
        t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 12 ];
        t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 13 ];
        t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 14 ];
        t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 15 ];

        idx = 0;
        // reset the round key
        for( idx = 0; idx < 8; ++idx )
            {
                // shift the bytes over by 16
                INTERPRET_UINT32( round_keys )[ idx ]
                    = INTERPRET_UINT32( round_keys )[ idx + 8 ];
                INTERPRET_UINT32( round_keys )[ idx + 8 ]
                    = INTERPRET_UINT32( round_keys )[ idx + 16 ];

            }

        /* round 8: */
        s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 8 ];
        s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 9 ];
        s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 10 ];
        s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 11 ];

        /* round 9: */
        t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 12 ];
        t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 13 ];
        t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 14 ];
        t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff]
            ^ INTERPRET_UINT32( round_keys )[ 15 ];
        if (Nr > 10) {

        idx = 0;
        // reset the round key
        for( idx = 0; idx < 8; ++idx )
            {
                // shift the bytes over by 16
                INTERPRET_UINT32( round_keys )[ idx ]
                    = INTERPRET_UINT32( round_keys )[ idx + 8 ];
                INTERPRET_UINT32( round_keys )[ idx + 8 ]
                    = INTERPRET_UINT32( round_keys )[ idx + 16 ];

            }

            /* round 10: */
            s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff]
                ^ INTERPRET_UINT32( round_keys )[ 8 ];
            s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff]
                ^ INTERPRET_UINT32( round_keys )[ 9 ];
            s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff]
                ^ INTERPRET_UINT32( round_keys )[ 10 ];
            s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff]
                ^ INTERPRET_UINT32( round_keys )[ 11 ];

            /* round 11: */
            t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff]
                ^ INTERPRET_UINT32( round_keys )[ 12 ];
            t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff]
                ^ INTERPRET_UINT32( round_keys )[ 13 ];
            t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff]
                ^ INTERPRET_UINT32( round_keys )[ 14 ];
            t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff]
                ^ INTERPRET_UINT32( round_keys )[ 15 ];
            if (Nr > 12) {

        idx = 0;
        // reset the round key
        for( idx = 0; idx < 8; ++idx )
            {
                // shift the bytes over by 16
                INTERPRET_UINT32( round_keys )[ idx ]
                    = INTERPRET_UINT32( round_keys )[ idx + 8 ];
                INTERPRET_UINT32( round_keys )[ idx + 8 ]
                    = INTERPRET_UINT32( round_keys )[ idx + 16 ];

            }

        // get the key for the round
        expand_key( INTERPRET_UINT32( round_keys) , ++i );


                /* round 12: */
                s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff]
                    ^ INTERPRET_UINT32( round_keys )[ 8 ];
                s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff]
                    ^ INTERPRET_UINT32( round_keys )[ 9 ];
                s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff]
                    ^ INTERPRET_UINT32( round_keys )[ 10 ];
                s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff]
                    ^ INTERPRET_UINT32( round_keys )[ 11 ];


                /* round 13: */
                t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff]
                    ^ INTERPRET_UINT32( round_keys )[ 12 ];
                t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff]
                    ^ INTERPRET_UINT32( round_keys )[ 13 ];
                t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff]
                    ^ INTERPRET_UINT32( round_keys )[ 14 ];
                t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff]
                    ^ INTERPRET_UINT32( round_keys )[ 15 ];
            }
        }

        idx = 0;
        // reset the round key
        for( idx = 0; idx < 8; ++idx )
            {
                // shift the bytes over by 16
                INTERPRET_UINT32( round_keys )[ idx ]
                    = INTERPRET_UINT32( round_keys )[ idx + 8 ];
                INTERPRET_UINT32( round_keys )[ idx + 8 ]
                    = INTERPRET_UINT32( round_keys )[ idx + 16 ];

            }

        // get the key for the round
        expand_key( INTERPRET_UINT32( round_keys) , ++i );


        /*
         * apply last round and
         * map cipher state to byte array block:
         */
        ct[ 0 ] =
            (Te2[(t0 >> 24)       ] & 0xff000000) ^
            (Te3[(t1 >> 16) & 0xff] & 0x00ff0000) ^
            (Te0[(t2 >>  8) & 0xff] & 0x0000ff00) ^
            (Te1[(t3      ) & 0xff] & 0x000000ff) ^
            INTERPRET_UINT32( round_keys )[ 12 ];
        ct[ 1] =
            (Te2[(t1 >> 24)       ] & 0xff000000) ^
            (Te3[(t2 >> 16) & 0xff] & 0x00ff0000) ^
            (Te0[(t3 >>  8) & 0xff] & 0x0000ff00) ^
            (Te1[(t0      ) & 0xff] & 0x000000ff) ^
            INTERPRET_UINT32( round_keys )[ 13 ];
        ct[ 2] =
            (Te2[(t2 >> 24)       ] & 0xff000000) ^
            (Te3[(t3 >> 16) & 0xff] & 0x00ff0000) ^
            (Te0[(t0 >>  8) & 0xff] & 0x0000ff00) ^
            (Te1[(t1      ) & 0xff] & 0x000000ff) ^
            INTERPRET_UINT32( round_keys )[ 14 ];
        ct[ 3] =
            (Te2[(t3 >> 24)       ] & 0xff000000) ^
            (Te3[(t0 >> 16) & 0xff] & 0x00ff0000) ^
            (Te0[(t1 >>  8) & 0xff] & 0x0000ff00) ^
            (Te1[(t2      ) & 0xff] & 0x000000ff) ^
            INTERPRET_UINT32( round_keys )[ 15 ];

    }


};