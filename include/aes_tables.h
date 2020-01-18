#ifndef AES_TABLES_HH_INCLUDED
#define AES_TABLES_HH_INCLUDED
#include <cstdint>

struct aes_tables
{
    uint *Te0, *Te1, *Te2, *Te3;
    std::uint8_t *sbox;
};

#endif // AES_TABLES_HH_INCLUDED
