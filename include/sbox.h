#ifndef __SBOX_TAB_HH
#define __SBOX_TAB_HH
using uchar = unsigned char;

#ifdef USE_CONSTANT
__constant__
#endif
__device__
extern uint Rcon[10];

#ifdef USE_CONSTANT
__constant__
#endif
__device__
extern uchar Tsbox_256[256];

#ifdef USE_CONSTANT
__constant__
#endif
__device__
extern uchar Tsbox_128[256];

#ifdef USE_CONSTANT
__constant__
#endif
__device__
extern uchar Tsbox_64[256];

#ifdef USE_CONSTANT
__constant__
#endif
__device__
extern uchar mul2[256];

#ifdef USE_CONSTANT
__constant__
#endif
__device__
extern uchar mul_3[256];

#ifdef USE_CONSTANT
__constant__
#endif
__device__
extern uchar mul_9[256];

#ifdef USE_CONSTANT
__constant__
#endif
__device__
extern uchar mul_11[256];

#ifdef USE_CONSTANT
__constant__
#endif
__device__
extern uchar mul_13[256];

#ifdef USE_CONSTANT
__constant__
#endif
__device__
extern uchar mul_14[256];

#ifdef USE_CONSTANT
__constant__
#endif
__device__
extern uint cTe0[256];

#ifdef USE_CONSTANT
__constant__
#endif
__device__
extern uint cTe1[256];

#ifdef USE_CONSTANT
__constant__
#endif
__device__
extern uint cTe2[256];

#ifdef USE_CONSTANT
__constant__
#endif
__device__
extern uint cTe3[256];

#ifdef USE_CONSTANT
__constant__
#endif
__device__
extern uint cTe4[256];


#endif
