#ifndef UTIL_CU_
#define UTIL_CU_

// GTX 1080: smem 40x1024, gmem 80x512
// GTX 980: 32x1024
// K40: 30x1024
// TRX 2080: smem 92x1024
#ifndef NUM_BLOCKS
#define NUM_BLOCKS 30
#endif

#ifdef _MSC_VER
#define SWAP(x) (_lrotl(x, 8) & 0x00ff00ff | _lrotr(x, 8) & 0xff00ff00)
#define GETWORD(p) SWAP(*((uint *)(p)))
#define PUTWORD(ct, st) (*((uint *)(ct)) = SWAP((st)))
#else
#define GETWORD(pt) (((uint)(pt)[0] << 24) ^ ((uint)(pt)[1] << 16) ^ ((uint)(pt)[2] <<  8) ^ ((uint)(pt)[3]))
#define PUTWORD(ct, st) ((ct)[0] = (uchar)((st) >> 24), (ct)[1] = (uchar)((st) >> 16), (ct)[2] = (uchar)((st) >>  8), (ct)[3] = (uchar)(st), (st))
#endif

#define NUM_THREADS 1024

#define MAX_RK_SIZE 182
__device__ void expand_key( const uchar *cipherKey,
                            uint *e_sched,
                            uint Nr,
                            uint keyBits
                          )
{
    uint *rek = e_sched;
    uint i = 0;
    uint temp;
    rek[0] = GETWORD(cipherKey     );
    rek[1] = GETWORD(cipherKey +  4);
    rek[2] = GETWORD(cipherKey +  8);
    rek[3] = GETWORD(cipherKey + 12);
    if (keyBits == 128) {
        for (;;) {
            temp  = rek[3];
            rek[4] = rek[0] ^
                (cTe4[(temp >> 16) & 0xff] & 0xff000000) ^
                (cTe4[(temp >>  8) & 0xff] & 0x00ff0000) ^
                (cTe4[(temp      ) & 0xff] & 0x0000ff00) ^
                (cTe4[(temp >> 24)       ] & 0x000000ff) ^
                crcon[i];
            rek[5] = rek[1] ^ rek[4];
            rek[6] = rek[2] ^ rek[5];
            rek[7] = rek[3] ^ rek[6];
            if (++i == 10) {
                Nr = 10;
                return;
            }
            rek += 4;
        }
    }
    rek[4] = GETWORD(cipherKey + 16);
    rek[5] = GETWORD(cipherKey + 20);
    if (keyBits == 192) {
        for (;;) {
            temp = rek[ 5];
            rek[ 6] = rek[ 0] ^
                (cTe4[(temp >> 16) & 0xff] & 0xff000000) ^
                (cTe4[(temp >>  8) & 0xff] & 0x00ff0000) ^
                (cTe4[(temp      ) & 0xff] & 0x0000ff00) ^
                (cTe4[(temp >> 24)       ] & 0x000000ff) ^
                crcon[i];
            rek[ 7] = rek[ 1] ^ rek[ 6];
            rek[ 8] = rek[ 2] ^ rek[ 7];
            rek[ 9] = rek[ 3] ^ rek[ 8];
            if (++i == 8) {
                Nr = 12;
                return;
            }
            rek[10] = rek[ 4] ^ rek[ 9];
            rek[11] = rek[ 5] ^ rek[10];
            rek += 6;
        }
    }
    rek[6] = GETWORD(cipherKey + 24);
    rek[7] = GETWORD(cipherKey + 28);
    if (keyBits == 256) {
        for (;;) {
            temp = rek[ 7];
            rek[ 8] = rek[ 0] ^
                (cTe4[(temp >> 16) & 0xff] & 0xff000000) ^
                (cTe4[(temp >>  8) & 0xff] & 0x00ff0000) ^
                (cTe4[(temp      ) & 0xff] & 0x0000ff00) ^
                (cTe4[(temp >> 24)       ] & 0x000000ff) ^
                crcon[i];
            rek[ 9] = rek[ 1] ^ rek[ 8];
            rek[10] = rek[ 2] ^ rek[ 9];
            rek[11] = rek[ 3] ^ rek[10];
            if (++i == 7) {
                Nr = 14;
                return;
            }
            temp = rek[11];
            rek[12] = rek[ 4] ^
                (cTe4[(temp >> 24)       ] & 0xff000000) ^
                (cTe4[(temp >> 16) & 0xff] & 0x00ff0000) ^
                (cTe4[(temp >>  8) & 0xff] & 0x0000ff00) ^
                (cTe4[(temp      ) & 0xff] & 0x000000ff);
            rek[13] = rek[ 5] ^ rek[12];
            rek[14] = rek[ 6] ^ rek[13];
            rek[15] = rek[ 7] ^ rek[14];
            rek += 8;
        }
    }
    Nr = 0; // this should never happen
}


__device__ void load_roundkey(uint *s, uint *rk) {
    int tid = threadIdx.x;
    if (tid < MAX_RK_SIZE)
	s[tid] = rk[tid];
    __syncthreads();
}

__device__ void load_smem_sbox(uchar *smem, uchar *gmem) {
    int tid = threadIdx.x;
    if (tid < 256)
	smem[tid] = gmem[tid];
    __syncthreads();
}

#if NUM_THREADS == 1024
__device__ void load_smem(uchar *st0, uchar *gt0, uchar *st1, uchar *gt1, uchar *st2, uchar *gt2, uchar *st3, uchar *gt3) {
    int tid = threadIdx.x;
    uint *s, *g;
    if (tid < 256) {
	s = (uint *)st0; g = (uint *)gt0;
	s[tid] = g[tid];
    } else if (tid < 512) {
	tid -= 256;
	s = (uint *)st1; g = (uint *)gt1;
	s[tid] = g[tid];
    } else if (tid < 768) {
	tid -= 512;
	s = (uint *)st2; g = (uint *)gt2;
	s[tid] = g[tid];
    } else {
	tid -= 768;
	s = (uint *)st3; g = (uint *)gt3;
	s[tid] = g[tid];
    }
    
    __syncthreads();
}
__device__ void load_smem(uint *st0, uint *gt0, uint *st1, uint *gt1, uint *st2, uint *gt2, uint *st3, uint *gt3) {
    int tid = threadIdx.x;
    if (tid < 256) {
	st0[tid] = gt0[tid];
    } else if (tid < 512) {
	tid -= 256;
	st1[tid] = gt1[tid];
    } else if (tid < 768) {
	tid -= 512;
	st2[tid] = gt2[tid];
    } else {
	tid -= 768;
	st3[tid] = gt3[tid];
    }
    
    __syncthreads();
}
#elif NUM_THREADS == 512
__device__ void load_smem(uchar *st0, uchar *gt0, uchar *st1, uchar *gt1, uchar *st2, uchar *gt2, uchar *st3, uchar *gt3) {
    int tid = threadIdx.x;
    uint *s, *g;
    if (tid < 256) {
	s = (uint *)st0; g = (uint *)gt0;
	s[tid] = g[tid];
	s = (uint *)st2; g = (uint *)gt2;
	s[tid] = g[tid];
    } else {
	tid -= 256;
	s = (uint *)st1; g = (uint *)gt1;
	s[tid] = g[tid];
	s = (uint *)st3; g = (uint *)gt3;
	s[tid] = g[tid];
    }
    
    __syncthreads();
}
__device__ void load_smem(uint *st0, uint *gt0, uint *st1, uint *gt1, uint *st2, uint *gt2, uint *st3, uint *gt3) {
    int tid = threadIdx.x;
    if (tid < 256) {
	st0[tid] = gt0[tid];
	st2[tid] = gt2[tid];
    } else {
	tid -= 256;
	st1[tid] = gt1[tid];
	st3[tid] = gt3[tid];
    }
    
    __syncthreads();
}

#elif NUM_THREADS == 256
__device__ void load_smem(uchar *st0, uchar *gt0, uchar *st1, uchar *gt1, uchar *st2, uchar *gt2, uchar *st3, uchar *gt3) {
    int tid = threadIdx.x;
    uint *s, *g;
    s = (uint *)st0; g = (uint *)gt0;
    s[tid] = g[tid];
    s = (uint *)st1; g = (uint *)gt1;
    s[tid] = g[tid];
    s = (uint *)st2; g = (uint *)gt2;
    s[tid] = g[tid];
    s = (uint *)st3; g = (uint *)gt3;
    s[tid] = g[tid];
    
    __syncthreads();
}
__device__ void load_smem(uint *st0, uint *gt0, uint *st1, uint *gt1, uint *st2, uint *gt2, uint *st3, uint *gt3) {
    int tid = threadIdx.x;
    st0[tid] = gt0[tid];
    st1[tid] = gt1[tid];
    st2[tid] = gt2[tid];
    st3[tid] = gt3[tid];
    
    __syncthreads();
}
#else 
#error NUM_THREADS must be 256, 512 or 1024
#endif // NUM_THREADS

#endif

