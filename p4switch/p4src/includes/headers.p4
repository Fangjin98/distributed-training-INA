#ifndef HEADERS_P4
#define HEADERS_P4

typedef bit<9>  egressSpec_t;
typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;

typedef int<32> data_t; // 16bit float -> 32bit integer -> int(signed)

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

header atp_t {      // TODO: 硬编码
    bit<32> workerMap;
    bit<5>  aggregationDegree;
    bit<1> overflow;            // TODO: 
    bit<1> isAck;               // TODO: 
    bit<1> ecn;                 // TODO: 
    bit<1> resend;              // TODO: 
    bit<5> aggIndex;
    bit<5> timestamp;           // TODO: 
    bit<5> switchId;            // TODO: 
    bit<32> sequenceId;         // TODO: 
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

struct headers {
    ethernet_t   ethernet;
    ipv4_t       ipv4;
    atp_t        atp;
    entry_t      atp_data;
}


/*************************************************************************
 ***********************  M E T A D A T A  *******************************
 *************************************************************************/

struct metadata { // FIXME: 每次都会清空对吧，这和直接当场定义有什么区别呢
    bit<1> tobe_agg;
    bit<5> aggIndex;  // int<8> -x-> bit<32>
    // count 的中间量
    bit<5> count_value; // 最多不超过 aggregationDegree 的大小，类型与之对应 bit<5>
    // aggregtor value 的中间量
    data_t aggre_value00;
    data_t aggre_value01;
    data_t aggre_value02;
    data_t aggre_value03;
    data_t aggre_value04;
    data_t aggre_value05;
    data_t aggre_value06;
    data_t aggre_value07;
    data_t aggre_value08;
    data_t aggre_value09;
    data_t aggre_value10;
    data_t aggre_value11;
    data_t aggre_value12;
    data_t aggre_value13;
    data_t aggre_value14;
    data_t aggre_value15;
    data_t aggre_value16;
    data_t aggre_value17;
    data_t aggre_value18;
    data_t aggre_value19;
    data_t aggre_value20;
    data_t aggre_value21;
    data_t aggre_value22;
    data_t aggre_value23;
    data_t aggre_value24;
    data_t aggre_value25;
    data_t aggre_value26;
    data_t aggre_value27;
    data_t aggre_value28;
    data_t aggre_value29;
    data_t aggre_value30;
    data_t aggre_value31;
}

#endif /* HEADERS_P4 */