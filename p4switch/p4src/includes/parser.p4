#ifndef PARSER_P4
#define PARSER_P4

#include "headers.p4"

const bit<16> TYPE_IPV4 = 0x800;
const bit<8>  TYPE_ATP = 0x12;

parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {

    state start {
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            TYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            TYPE_ATP: parse_value;
            default: accept;
        }
    }

    state parse_value {
        packet.extract(hdr.atp);
        packet.extract(hdr.atp_data);
        transition accept;
    }
}

#endif /* PARSER_P4 */