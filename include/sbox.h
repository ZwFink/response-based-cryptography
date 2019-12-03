#ifndef __SBOX_TAB_HH
#define __SBOX_TAB_HH
using uchar = unsigned char;

extern uchar Rcon[255];

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




#endif
