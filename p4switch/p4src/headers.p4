#ifndef HEADERS_P4
#define HEADERS_P4

#include "types.p4"

header ethernet_t {
    macAddr_t dstAddr;
    macAddr_t srcAddr;
    bit<16>   etherType;
}

header ipv4_t {
    bit<4>    version;
    bit<4>    ihl;
    bit<8>    diffserv;
    bit<16>   totalLen;
    bit<16>   identification;
    bit<3>    flags;
    bit<13>   fragOffset;
    bit<8>    ttl;
    bit<8>    protocol;
    bit<16>   hdrChecksum;
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
}

header atp_t {
    bit<32> workerMap;
    bit<8>  aggregationDegree;
    bit<1> overflow;           
    bit<1> isAck;               
    bit<1> ecn;                
    bit<1> resend;            
    bit<4> timestamp;         
    bit<32> aggIndex;
    bit<8> switchId;            
    bit<32> sequenceId;         
}

header entry_t { // TODO: rename
    data_t value00;
    data_t value01;
    data_t value02;
    data_t value03;
    data_t value04;
    data_t value05;
    data_t value06;
    data_t value07;
    data_t value08;
    data_t value09;
    data_t value10;
    data_t value11;
    data_t value12;
    data_t value13;
    data_t value14;
    data_t value15;
    data_t value16;
    data_t value17;
    data_t value18;
    data_t value19;
    data_t value20;
    data_t value21;
    data_t value22;
    data_t value23;
    data_t value24;
    data_t value25;
    data_t value26;
    data_t value27;
    data_t value28;
    data_t value29;
    data_t value30;
    data_t value31;
}

struct header_t {
    ethernet_t   ethernet;
    ipv4_t       ipv4;
    atp_t        atp;
    entry_t      atp_data;
}

struct empty_metadata_t {}


/*************************************************************************
 ***********************  M E T A D A T A  *******************************
 *************************************************************************/

struct metadata_t {
    bit<1> tobe_agg;
    bit<1> isAck;
    bit<1> clear;
    bit<8> aggDegree;
    bit<8> count_value;
    bit<32> aggIndex;
}


#endif /* HEADERS_P4 */