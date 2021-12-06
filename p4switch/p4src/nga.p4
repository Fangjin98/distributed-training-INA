/* -*- P4_16 -*- */
#include <core.p4>
#include <tna.p4>
#include "parser.p4"
#include "headers.p4"
#include "processor.p4"
#include "types.p4"
#include "config.p4"
// #include "counter.p4"

/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control Ingress(
        inout header_t hdr,
        inout metadata_t ig_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {
    
    action set_agg() {
        ig_md.tobe_agg = 1;
    }

    action unset_agg() {
        ig_md.tobe_agg = 0;
    }
    
    table switch_check {
        key = {
            hdr.atp.switchId: exact;
        }
        actions = {
            set_agg;
            unset_agg;
        }
        size = 1024;
        default_action = unset_agg;
    }
   
    action drop() {
        ig_dprsr_md.drop_ctl = 1;
    }
    
    action ipv4_forward(macAddr_t dstMacAddr, egressSpec_t port) {
        ig_tm_md.ucast_egress_port = port;
        // hdr.ethernet.srcAddr = hdr.ethernet.dstAddr;
        // hdr.ethernet.dstAddr = dstMacAddr;
        // hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
    }
    
    table ipv4_lpm {
        key = {
            hdr.ipv4.dstAddr: exact;
        }
        actions = {
            ipv4_forward;
            drop;
            NoAction;
        }
        size = 1<<8;
        default_action = drop;
    }


    Register<bit<8>,index_t>(register_size, 0) count_reg; 

    RegisterAction<bit<8>, index_t, bit<8>>(count_reg) count_register_action = {
        void apply(inout bit<8> value, out bit<8> read_value) {
            if(value == hdr.atp.aggregationDegree){
                value=1;
            }
            else{
                value = value+1;
            }
            read_value = value;
        }
    };

    action count_action(){
        ig_md.count_value=count_register_action.execute(ig_md.aggIndex);
    }
    
    Processor() value00;
    Processor() value01;
    Processor() value02;
    Processor() value03;
    Processor() value04;
    Processor() value05;
    Processor() value06;
    Processor() value07;
    Processor() value08;
    Processor() value09;
    Processor() value10;
    Processor() value11;
    Processor() value12;
    Processor() value13;
    Processor() value14;
    Processor() value15;
    Processor() value16;
    Processor() value17;
    Processor() value18;
    Processor() value19;
    Processor() value20;
    Processor() value21;
    Processor() value22;
    Processor() value23;
    Processor() value24;
    Processor() value25;
    Processor() value26;
    Processor() value27;
    Processor() value28;
    Processor() value29;
    Processor() value30;
    Processor() value31;

    apply {
        if(hdr.atp.isValid()) {
            switch_check.apply();
            if(ig_md.tobe_agg == 1 ){
                if( hdr.atp.isAck==1){ //send back to PS
                    ipv4_lpm.apply();
                }
                else{
                    ig_md.aggIndex = hdr.atp.aggIndex;
                    ig_md.aggDegree = hdr.atp.aggregationDegree;
                    count_action(); 
                    
                    value00.apply(hdr.atp_data.value00,hdr.atp_data.value00,ig_md);
                    value01.apply(hdr.atp_data.value01,hdr.atp_data.value01,ig_md);
                    value02.apply(hdr.atp_data.value02,hdr.atp_data.value02,ig_md);
                    value03.apply(hdr.atp_data.value03,hdr.atp_data.value03,ig_md);
                    value04.apply(hdr.atp_data.value04,hdr.atp_data.value04,ig_md);
                    value05.apply(hdr.atp_data.value05,hdr.atp_data.value05,ig_md);
                    value06.apply(hdr.atp_data.value06,hdr.atp_data.value06,ig_md);
                    value07.apply(hdr.atp_data.value07,hdr.atp_data.value07,ig_md);
                    value08.apply(hdr.atp_data.value08,hdr.atp_data.value08,ig_md);
                    value09.apply(hdr.atp_data.value09,hdr.atp_data.value09,ig_md);
                    value10.apply(hdr.atp_data.value10,hdr.atp_data.value10,ig_md);
                    value11.apply(hdr.atp_data.value11,hdr.atp_data.value11,ig_md);
                    value12.apply(hdr.atp_data.value12,hdr.atp_data.value12,ig_md);
                    value13.apply(hdr.atp_data.value13,hdr.atp_data.value13,ig_md);
                    value14.apply(hdr.atp_data.value14,hdr.atp_data.value14,ig_md);
                    value15.apply(hdr.atp_data.value15,hdr.atp_data.value15,ig_md);
                    value16.apply(hdr.atp_data.value16,hdr.atp_data.value16,ig_md);
                    value17.apply(hdr.atp_data.value17,hdr.atp_data.value17,ig_md);
                    value18.apply(hdr.atp_data.value18,hdr.atp_data.value18,ig_md);
                    value19.apply(hdr.atp_data.value19,hdr.atp_data.value19,ig_md);
                    value20.apply(hdr.atp_data.value20,hdr.atp_data.value20,ig_md);
                    value21.apply(hdr.atp_data.value21,hdr.atp_data.value21,ig_md);
                    value22.apply(hdr.atp_data.value22,hdr.atp_data.value22,ig_md);
                    value23.apply(hdr.atp_data.value23,hdr.atp_data.value23,ig_md);
                    value24.apply(hdr.atp_data.value24,hdr.atp_data.value24,ig_md);
                    value25.apply(hdr.atp_data.value25,hdr.atp_data.value25,ig_md);
                    value26.apply(hdr.atp_data.value26,hdr.atp_data.value26,ig_md);
                    value27.apply(hdr.atp_data.value27,hdr.atp_data.value27,ig_md);
                    value28.apply(hdr.atp_data.value28,hdr.atp_data.value28,ig_md);
                    value29.apply(hdr.atp_data.value29,hdr.atp_data.value29,ig_md);
                    value30.apply(hdr.atp_data.value30,hdr.atp_data.value30,ig_md);
                    value31.apply(hdr.atp_data.value31,hdr.atp_data.value31,ig_md);
                
                    if (ig_md.count_value == hdr.atp.aggregationDegree){
                        ipv4_lpm.apply();
                    }
                    else{
                        drop();
                    }
                }
            }
            else{
                drop();
            }
        } 
        else {     
            if (hdr.ipv4.isValid()) {
                ipv4_lpm.apply();
            }
        }

        
    }
}

/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

Pipeline(IngressParser(),
         Ingress(),
         IngressDeparser(),
         EgressParser(),
         Egress(),
         EgressDeparser()) pipe;

Switch(pipe) main;