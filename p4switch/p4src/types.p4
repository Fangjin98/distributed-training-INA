#ifndef _TYPES_
#define _TYPES_

#include "config.p4"

typedef bit<9>  egressSpec_t;
typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;

typedef bit<32> data_t; // 16bit float -> 32bit integer -> int(signed)
typedef bit<32> index_t;
struct data_single_t{
    data_t first;
}
#endif /* _TYPES_ */