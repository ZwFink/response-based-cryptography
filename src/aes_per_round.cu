#include "aes_per_round.h"

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

};