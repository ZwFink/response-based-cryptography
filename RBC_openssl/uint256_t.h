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

using namespace std;


typedef std::uint8_t uint256_data_t[ UINT256_SIZE_IN_BYTES ];

//namespace uint256_ctz_table
//{
//    extern std::uint8_t lookup[ 37 ];
//
//    int ctz( const std::uint32_t loc );
//}

class uint256_t
{
 public:
     uint256_t();
     //~uint256_t() = default;
     void set_all( std::uint8_t value );
     std::uint8_t at( int loc );
     uint256_t( std::uint8_t set );
     uint256_data_t& get_data();
     std::uint8_t *get_data_ptr();
     // set the member at index to set
     uint256_t( std::uint8_t set, std::uint8_t index );
     void set( std::uint8_t set, std::uint8_t index );

     void copy( const uint256_t& copied );
    // copy 64 bit integer into member starting at index
     void copy_64( uint64_t ref, uint8_t index );

     void from_string( const unsigned char *string );

     uint256_t operator&( const uint256_t& comp ) const;
     uint256_t operator|( const uint256_t& comp ) const;
     uint256_t operator^( const uint256_t& comp ) const;

     uint256_t operator<<( int shift ) const;
     uint256_t operator>>( int shift ) const;

     uint256_t operator~() const;

     std::uint8_t& operator[]( std::uint8_t idx );
     const std::uint8_t& operator[]( std::uint8_t idx ) const;

     bool operator==( const uint256_t& comp ) const;
     bool operator!=( const uint256_t& comp ) const;

     int compare( const uint256_t& comp ) const;
     bool operator<( const uint256_t& comp ) const;
     bool operator>( const uint256_t& comp ) const;



     void operator=( const uint256_t& set );
     void dump();
     void dump_hex();

     int ctz();
     int popc();

     void to_32_bit_arr( std::uint32_t* dest );

     void set_bit( std::uint8_t bit_idx );

     //bool add( uint256_t& dest, const uint256_t augend ) const;

     unsigned char add( uint256_t rop, 
                        uint256_t op2 );

     uint256_t operator+( const uint256_t& other ) const;

    // this must be device-only because uint256_t::add is used
     void neg( uint256_t& dest ) const;
     uint256_t operator-() const;


     std::uint8_t data[ UINT256_SIZE_IN_BYTES ];

};

#endif // UINT256_T_HH_INCLUDED
