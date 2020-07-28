#ifndef UINT256_T_HH_INCLUDED
#define UINT256_T_HH_INCLUDED

#define UINT256_SIZE_IN_BYTES 32
#define UINT256_SIZE_IN_BITS 256
#define UINT256_LIMB_SIZE 8
#define UINT256_MAX_INT uint256_t( 0xFF )
#define UINT256_ZERO uint256_t( 0x00 )
#define UINT256_ONE uint256_t( 0x01, 0 )
#define UINT256_NEGATIVE_ONE UINT256_MAX_INT

#include <cstddef>
#include <string>
#include <string.h>
#include <x86intrin.h>
#include <cstdint>

#include <iostream>
#include <iomanip>
#include <bits/stdc++.h>

using namespace std;


typedef std::uint8_t uint256_data_t[ UINT256_SIZE_IN_BYTES ];

namespace uint256_ctz_table
{
    extern std::uint8_t lookup[ 37 ];

    int ctz( const std::uint32_t loc );
}

class uint256_t
{

 public:

     /* Constructors */

     uint256_t();
     uint256_t( std::uint8_t set );
     uint256_t( std::uint8_t set, std::uint8_t index );
     void copy( const uint256_t& copied );

     /* Methods */

       // manipulate data
     void set_all( std::uint8_t value );
     void set( std::uint8_t set, std::uint8_t index );
     void set_bit( std::uint8_t bit_idx );
     void from_string( const unsigned char *string );
     uint256_t operator~() const;
     uint256_t operator&( const uint256_t& comp ) const;
     uint256_t operator^( const uint256_t& comp ) const;
     uint256_t operator|( const uint256_t& comp ) const;
     void operator=( const uint256_t& set );
     uint256_t operator<<( int shift ) const;
     uint256_t operator>>( int shift ) const;
     unsigned char add( uint256_t *rop, uint256_t op2 );
     uint256_t operator+( const uint256_t& other ) const;
     void neg( uint256_t& dest ) const;
     uint256_t operator-() const;

       // access data
     std::uint8_t at( int loc );
     std::uint8_t& operator[]( std::uint8_t idx );
     const std::uint8_t& operator[]( std::uint8_t idx ) const;
     uint256_data_t& get_data();
     std::uint8_t *get_data_ptr();

       // compare data
     bool operator==( const uint256_t& comp ) const;
     bool operator!=( const uint256_t& comp ) const;
     bool operator<( const uint256_t& comp ) const;
     bool operator>( const uint256_t& comp ) const;
     int compare( const uint256_t& comp ) const;

       // print data
     void dump_hex();
     void dump();

       // get data information
     int ctz();
     int popc();



     std::uint8_t data[ UINT256_SIZE_IN_BYTES ];
};

#endif // UINT256_T_HH_INCLUDED
