// test file

#include "test.h"

int main(int argc, char **argv)
{

    uint256_t num1;
    num1.set_bit(0);
    uint256_t num2( 0 );
    num2.set_bit(127);
    uint256_t result( 0 );
    unsigned char carry = 0;

    num1.dump();
    num2.dump();

    //result = num1 + num2;
    //carry = num1.add(&result,num2);
    //fprintf(stderr,"\n\nCarry = %d\n\n",carry);
    //result = -num1;

    uint256_t test;
    num1.add(&test, UINT256_NEGATIVE_ONE);

    //test.dump();

    uint256_t t = num1 | ( num1 + UINT256_NEGATIVE_ONE );
    uint256_t t2= num1 | test;

    t.dump();
    t2.dump();


    uint8_t shift = num1.ctz() + 1;
    fprintf(stderr,"\nshift = %d\n",shift);

    // add_tmp.set_all( 0x00 );
    uint256_t tmp;

    unsigned char overflow = t.add( &tmp, UINT256_ONE );
    fprintf(stderr,"\noverflow = %d\n",overflow);

    //t = (~t) & -(~t);
    //t.add(&t,UINT256_NEGATIVE_ONE);
    //t = t >> shift;
    //num1 = tmp | t;

    //num1 = (tmp) | ((((~t) & -(~t)) + UINT256_NEGATIVE_ONE ) >> shift ); 
    

    //uint8_t shift = num1.ctz() + 1;

    //fprintf(stderr,"\nShift = %d\n",shift);

    //result.dump();

}

// end test file    
