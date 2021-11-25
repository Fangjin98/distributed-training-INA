/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>
#include "includes/headers.p4"
#include "includes/parser.p4"

#define JOB_NUM 512                // 支持的聚合 Job 数量


/*************************************************************************
************   C H E C K S U M    V E R I F I C A T I O N   *************
*************************************************************************/

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {   
    apply {  }
}


/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {
    
    // NOTE: switchID check
    action set_agg() {
        meta.tobe_agg = 1;
    }

    action unset_agg() {
        meta.tobe_agg = 0;
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
        default_action = unset_agg();
    }
    
    action drop() {
        mark_to_drop(standard_metadata);
    }
    
    // NOTE: IPv4
    action ipv4_forward(macAddr_t dstAddr, egressSpec_t port) {
        standard_metadata.egress_spec = port;
        hdr.ethernet.srcAddr = hdr.ethernet.dstAddr;
        hdr.ethernet.dstAddr = dstAddr;
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
    }
    
    table ipv4_lpm {
        key = {
            hdr.ipv4.dstAddr: lpm;
        }
        actions = {
            ipv4_forward;
            drop;
            NoAction;
        }
        size = 1024;
        default_action = drop();
    }

    // NOTE: aggregation_foward，实现上看起来和 ipv4_forward 一样，但其实如果和控制平面交互，会不一样
    action aggregation_foward(macAddr_t dstAddr, egressSpec_t port) {
        standard_metadata.egress_spec = port;
        hdr.ethernet.srcAddr = hdr.ethernet.dstAddr;
        hdr.ethernet.dstAddr = dstAddr;
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
        meta.tobe_agg = 1;
    }

    table aggregate_link_lpm {
        key = {
            hdr.ipv4.srcAddr: lpm;
            hdr.ipv4.dstAddr: exact;
            standard_metadata.ingress_port: exact;
        }
        actions = {
            aggregation_foward;
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }


    // NOTE: counter
    register<bit<5>>(JOB_NUM) count_reg;       // 和 aggregationDegree 同类型

    action count_read() {
        count_reg.read(meta.count_value, (bit<32>)meta.aggIndex);
    }

    action count_add() { 
        count_reg.read(meta.count_value, (bit<32>)meta.aggIndex);
        meta.count_value = meta.count_value + 1;
        count_reg.write((bit<32>)meta.aggIndex, meta.count_value);
    }
    
    action count_clean() {
        count_reg.write((bit<32>)meta.aggIndex, 0);
    }

    // NOTE: aggregators
    register<data_t>(JOB_NUM) aggrvalue_vector00; // NOTE: 暂时不考虑溢出问题。 TODO: int 没有试过，到时候发包解析试试，但是寄存器会遇到问题
    register<data_t>(JOB_NUM) aggrvalue_vector01;
    register<data_t>(JOB_NUM) aggrvalue_vector02;
    register<data_t>(JOB_NUM) aggrvalue_vector03;
    register<data_t>(JOB_NUM) aggrvalue_vector04;
    register<data_t>(JOB_NUM) aggrvalue_vector05;
    register<data_t>(JOB_NUM) aggrvalue_vector06;
    register<data_t>(JOB_NUM) aggrvalue_vector07;
    register<data_t>(JOB_NUM) aggrvalue_vector08;
    register<data_t>(JOB_NUM) aggrvalue_vector09;
    register<data_t>(JOB_NUM) aggrvalue_vector10;
    register<data_t>(JOB_NUM) aggrvalue_vector11;
    register<data_t>(JOB_NUM) aggrvalue_vector12;
    register<data_t>(JOB_NUM) aggrvalue_vector13;
    register<data_t>(JOB_NUM) aggrvalue_vector14;
    register<data_t>(JOB_NUM) aggrvalue_vector15;
    register<data_t>(JOB_NUM) aggrvalue_vector16;
    register<data_t>(JOB_NUM) aggrvalue_vector17;
    register<data_t>(JOB_NUM) aggrvalue_vector18;
    register<data_t>(JOB_NUM) aggrvalue_vector19;
    register<data_t>(JOB_NUM) aggrvalue_vector20;
    register<data_t>(JOB_NUM) aggrvalue_vector21;
    register<data_t>(JOB_NUM) aggrvalue_vector22;
    register<data_t>(JOB_NUM) aggrvalue_vector23;
    register<data_t>(JOB_NUM) aggrvalue_vector24;
    register<data_t>(JOB_NUM) aggrvalue_vector25;
    register<data_t>(JOB_NUM) aggrvalue_vector26;
    register<data_t>(JOB_NUM) aggrvalue_vector27;
    register<data_t>(JOB_NUM) aggrvalue_vector28;
    register<data_t>(JOB_NUM) aggrvalue_vector29;
    register<data_t>(JOB_NUM) aggrvalue_vector30;
    register<data_t>(JOB_NUM) aggrvalue_vector31;

    // NOTE: 在 SwitchML 里面都是这样的逻辑，但是至少在P4_16中可以像函数一样直接调用 action.

    // vectorX_add
    action vector00_add() {
        aggrvalue_vector00.read(meta.aggre_value00, (bit<32>)meta.aggIndex);
        meta.aggre_value00 = meta.aggre_value00 + hdr.atp_data.value00;
        aggrvalue_vector00.write((bit<32>)meta.aggIndex, meta.aggre_value00);
    }

    action vector01_add() {
        aggrvalue_vector01.read(meta.aggre_value01, (bit<32>)meta.aggIndex);
        meta.aggre_value01 = meta.aggre_value01 + hdr.atp_data.value01;
        aggrvalue_vector01.write((bit<32>)meta.aggIndex, meta.aggre_value01);
    }

    action vector02_add() {
        aggrvalue_vector02.read(meta.aggre_value02, (bit<32>)meta.aggIndex);
        meta.aggre_value02 = meta.aggre_value02 + hdr.atp_data.value02;
        aggrvalue_vector02.write((bit<32>)meta.aggIndex, meta.aggre_value02);
    }

    action vector03_add() {
        aggrvalue_vector03.read(meta.aggre_value03, (bit<32>)meta.aggIndex);
        meta.aggre_value03 = meta.aggre_value03 + hdr.atp_data.value03;
        aggrvalue_vector03.write((bit<32>)meta.aggIndex, meta.aggre_value03);
    }

    action vector04_add() {
        aggrvalue_vector04.read(meta.aggre_value04, (bit<32>)meta.aggIndex);
        meta.aggre_value04 = meta.aggre_value04 + hdr.atp_data.value04;
        aggrvalue_vector04.write((bit<32>)meta.aggIndex, meta.aggre_value04);
    }

    action vector05_add() {
        aggrvalue_vector05.read(meta.aggre_value05, (bit<32>)meta.aggIndex);
        meta.aggre_value05 = meta.aggre_value05 + hdr.atp_data.value05;
        aggrvalue_vector05.write((bit<32>)meta.aggIndex, meta.aggre_value05);
    }

    action vector06_add() {
        aggrvalue_vector06.read(meta.aggre_value06, (bit<32>)meta.aggIndex);
        meta.aggre_value06 = meta.aggre_value06 + hdr.atp_data.value06;
        aggrvalue_vector06.write((bit<32>)meta.aggIndex, meta.aggre_value06);
    }

    action vector07_add() {
        aggrvalue_vector07.read(meta.aggre_value07, (bit<32>)meta.aggIndex);
        meta.aggre_value07 = meta.aggre_value07 + hdr.atp_data.value07;
        aggrvalue_vector07.write((bit<32>)meta.aggIndex, meta.aggre_value07);
    }

    action vector08_add() {
        aggrvalue_vector08.read(meta.aggre_value08, (bit<32>)meta.aggIndex);
        meta.aggre_value08 = meta.aggre_value08 + hdr.atp_data.value08;
        aggrvalue_vector08.write((bit<32>)meta.aggIndex, meta.aggre_value08);
    }

    action vector09_add() {
        aggrvalue_vector09.read(meta.aggre_value09, (bit<32>)meta.aggIndex);
        meta.aggre_value09 = meta.aggre_value09 + hdr.atp_data.value09;
        aggrvalue_vector09.write((bit<32>)meta.aggIndex, meta.aggre_value09);
    }

    action vector10_add() {
        aggrvalue_vector10.read(meta.aggre_value10, (bit<32>)meta.aggIndex);
        meta.aggre_value10 = meta.aggre_value10 + hdr.atp_data.value10;
        aggrvalue_vector10.write((bit<32>)meta.aggIndex, meta.aggre_value10);
    }

    action vector11_add() {
        aggrvalue_vector11.read(meta.aggre_value11, (bit<32>)meta.aggIndex);
        meta.aggre_value11 = meta.aggre_value11 + hdr.atp_data.value11;
        aggrvalue_vector11.write((bit<32>)meta.aggIndex, meta.aggre_value11);
    }

    action vector12_add() {
        aggrvalue_vector12.read(meta.aggre_value12, (bit<32>)meta.aggIndex);
        meta.aggre_value12 = meta.aggre_value12 + hdr.atp_data.value12;
        aggrvalue_vector12.write((bit<32>)meta.aggIndex, meta.aggre_value12);
    }

    action vector13_add() {
        aggrvalue_vector13.read(meta.aggre_value13, (bit<32>)meta.aggIndex);
        meta.aggre_value13 = meta.aggre_value13 + hdr.atp_data.value13;
        aggrvalue_vector13.write((bit<32>)meta.aggIndex, meta.aggre_value13);
    }

    action vector14_add() {
        aggrvalue_vector14.read(meta.aggre_value14, (bit<32>)meta.aggIndex);
        meta.aggre_value14 = meta.aggre_value14 + hdr.atp_data.value14;
        aggrvalue_vector14.write((bit<32>)meta.aggIndex, meta.aggre_value14);
    }

    action vector15_add() {
        aggrvalue_vector15.read(meta.aggre_value15, (bit<32>)meta.aggIndex);
        meta.aggre_value15 = meta.aggre_value15 + hdr.atp_data.value15;
        aggrvalue_vector15.write((bit<32>)meta.aggIndex, meta.aggre_value15);
    }

    action vector16_add() {
        aggrvalue_vector16.read(meta.aggre_value16, (bit<32>)meta.aggIndex);
        meta.aggre_value16 = meta.aggre_value16 + hdr.atp_data.value16;
        aggrvalue_vector16.write((bit<32>)meta.aggIndex, meta.aggre_value16);
    }

    action vector17_add() {
        aggrvalue_vector17.read(meta.aggre_value17, (bit<32>)meta.aggIndex);
        meta.aggre_value17 = meta.aggre_value17 + hdr.atp_data.value17;
        aggrvalue_vector17.write((bit<32>)meta.aggIndex, meta.aggre_value17);
    }

    action vector18_add() {
        aggrvalue_vector18.read(meta.aggre_value18, (bit<32>)meta.aggIndex);
        meta.aggre_value18 = meta.aggre_value18 + hdr.atp_data.value18;
        aggrvalue_vector18.write((bit<32>)meta.aggIndex, meta.aggre_value18);
    }

    action vector19_add() {
        aggrvalue_vector19.read(meta.aggre_value19, (bit<32>)meta.aggIndex);
        meta.aggre_value19 = meta.aggre_value19 + hdr.atp_data.value19;
        aggrvalue_vector19.write((bit<32>)meta.aggIndex, meta.aggre_value19);
    }

    action vector20_add() {
        aggrvalue_vector20.read(meta.aggre_value20, (bit<32>)meta.aggIndex);
        meta.aggre_value20 = meta.aggre_value20 + hdr.atp_data.value20;
        aggrvalue_vector20.write((bit<32>)meta.aggIndex, meta.aggre_value20);
    }

    action vector21_add() {
        aggrvalue_vector21.read(meta.aggre_value21, (bit<32>)meta.aggIndex);
        meta.aggre_value21 = meta.aggre_value21 + hdr.atp_data.value21;
        aggrvalue_vector21.write((bit<32>)meta.aggIndex, meta.aggre_value21);
    }

    action vector22_add() {
        aggrvalue_vector22.read(meta.aggre_value22, (bit<32>)meta.aggIndex);
        meta.aggre_value22 = meta.aggre_value22 + hdr.atp_data.value22;
        aggrvalue_vector22.write((bit<32>)meta.aggIndex, meta.aggre_value22);
    }

    action vector23_add() {
        aggrvalue_vector23.read(meta.aggre_value23, (bit<32>)meta.aggIndex);
        meta.aggre_value23 = meta.aggre_value23 + hdr.atp_data.value23;
        aggrvalue_vector23.write((bit<32>)meta.aggIndex, meta.aggre_value23);
    }

    action vector24_add() {
        aggrvalue_vector24.read(meta.aggre_value24, (bit<32>)meta.aggIndex);
        meta.aggre_value24 = meta.aggre_value24 + hdr.atp_data.value24;
        aggrvalue_vector24.write((bit<32>)meta.aggIndex, meta.aggre_value24);
    }

    action vector25_add() {
        aggrvalue_vector25.read(meta.aggre_value25, (bit<32>)meta.aggIndex);
        meta.aggre_value25 = meta.aggre_value25 + hdr.atp_data.value25;
        aggrvalue_vector25.write((bit<32>)meta.aggIndex, meta.aggre_value25);
    }

    action vector26_add() {
        aggrvalue_vector26.read(meta.aggre_value26, (bit<32>)meta.aggIndex);
        meta.aggre_value26 = meta.aggre_value26 + hdr.atp_data.value26;
        aggrvalue_vector26.write((bit<32>)meta.aggIndex, meta.aggre_value26);
    }

    action vector27_add() {
        aggrvalue_vector27.read(meta.aggre_value27, (bit<32>)meta.aggIndex);
        meta.aggre_value27 = meta.aggre_value27 + hdr.atp_data.value27;
        aggrvalue_vector27.write((bit<32>)meta.aggIndex, meta.aggre_value27);
    }

    action vector28_add() {
        aggrvalue_vector28.read(meta.aggre_value28, (bit<32>)meta.aggIndex);
        meta.aggre_value28 = meta.aggre_value28 + hdr.atp_data.value28;
        aggrvalue_vector28.write((bit<32>)meta.aggIndex, meta.aggre_value28);
    }

    action vector29_add() {
        aggrvalue_vector29.read(meta.aggre_value29, (bit<32>)meta.aggIndex);
        meta.aggre_value29 = meta.aggre_value29 + hdr.atp_data.value29;
        aggrvalue_vector29.write((bit<32>)meta.aggIndex, meta.aggre_value29);
    }

    action vector30_add() {
        aggrvalue_vector30.read(meta.aggre_value30, (bit<32>)meta.aggIndex);
        meta.aggre_value30 = meta.aggre_value30 + hdr.atp_data.value30;
        aggrvalue_vector30.write((bit<32>)meta.aggIndex, meta.aggre_value30);
    }

    action vector31_add() {
        aggrvalue_vector31.read(meta.aggre_value31, (bit<32>)meta.aggIndex);
        meta.aggre_value31 = meta.aggre_value31 + hdr.atp_data.value31;
        aggrvalue_vector31.write((bit<32>)meta.aggIndex, meta.aggre_value31);
    }

    // vectorX_read
    action vector00_read() {
        aggrvalue_vector00.read(hdr.atp_data.value00, (bit<32>)meta.aggIndex);
    }

    action vector01_read() {
        aggrvalue_vector01.read(hdr.atp_data.value01, (bit<32>)meta.aggIndex);
    }

    action vector02_read() {
        aggrvalue_vector02.read(hdr.atp_data.value02, (bit<32>)meta.aggIndex);
    }

    action vector03_read() {
        aggrvalue_vector03.read(hdr.atp_data.value03, (bit<32>)meta.aggIndex);
    }

    action vector04_read() {
        aggrvalue_vector04.read(hdr.atp_data.value04, (bit<32>)meta.aggIndex);
    }

    action vector05_read() {
        aggrvalue_vector05.read(hdr.atp_data.value05, (bit<32>)meta.aggIndex);
    }

    action vector06_read() {
        aggrvalue_vector06.read(hdr.atp_data.value06, (bit<32>)meta.aggIndex);
    }

    action vector07_read() {
        aggrvalue_vector07.read(hdr.atp_data.value07, (bit<32>)meta.aggIndex);
    }

    action vector08_read() {
        aggrvalue_vector08.read(hdr.atp_data.value08, (bit<32>)meta.aggIndex);
    }

    action vector09_read() {
        aggrvalue_vector09.read(hdr.atp_data.value09, (bit<32>)meta.aggIndex);
    }

    action vector10_read() {
        aggrvalue_vector10.read(hdr.atp_data.value10, (bit<32>)meta.aggIndex);
    }

    action vector11_read() {
        aggrvalue_vector11.read(hdr.atp_data.value11, (bit<32>)meta.aggIndex);
    }

    action vector12_read() {
        aggrvalue_vector12.read(hdr.atp_data.value12, (bit<32>)meta.aggIndex);
    }

    action vector13_read() {
        aggrvalue_vector13.read(hdr.atp_data.value13, (bit<32>)meta.aggIndex);
    }

    action vector14_read() {
        aggrvalue_vector14.read(hdr.atp_data.value14, (bit<32>)meta.aggIndex);
    }

    action vector15_read() {
        aggrvalue_vector15.read(hdr.atp_data.value15, (bit<32>)meta.aggIndex);
    }

    action vector16_read() {
        aggrvalue_vector16.read(hdr.atp_data.value16, (bit<32>)meta.aggIndex);
    }

    action vector17_read() {
        aggrvalue_vector17.read(hdr.atp_data.value17, (bit<32>)meta.aggIndex);
    }

    action vector18_read() {
        aggrvalue_vector18.read(hdr.atp_data.value18, (bit<32>)meta.aggIndex);
    }

    action vector19_read() {
        aggrvalue_vector19.read(hdr.atp_data.value19, (bit<32>)meta.aggIndex);
    }

    action vector20_read() {
        aggrvalue_vector20.read(hdr.atp_data.value20, (bit<32>)meta.aggIndex);
    }

    action vector21_read() {
        aggrvalue_vector21.read(hdr.atp_data.value21, (bit<32>)meta.aggIndex);
    }

    action vector22_read() {
        aggrvalue_vector22.read(hdr.atp_data.value22, (bit<32>)meta.aggIndex);
    }

    action vector23_read() {
        aggrvalue_vector23.read(hdr.atp_data.value23, (bit<32>)meta.aggIndex);
    }

    action vector24_read() {
        aggrvalue_vector24.read(hdr.atp_data.value24, (bit<32>)meta.aggIndex);
    }

    action vector25_read() {
        aggrvalue_vector25.read(hdr.atp_data.value25, (bit<32>)meta.aggIndex);
    }

    action vector26_read() {
        aggrvalue_vector26.read(hdr.atp_data.value26, (bit<32>)meta.aggIndex);
    }

    action vector27_read() {
        aggrvalue_vector27.read(hdr.atp_data.value27, (bit<32>)meta.aggIndex);
    }

    action vector28_read() {
        aggrvalue_vector28.read(hdr.atp_data.value28, (bit<32>)meta.aggIndex);
    }

    action vector29_read() {
        aggrvalue_vector29.read(hdr.atp_data.value29, (bit<32>)meta.aggIndex);
    }

    action vector30_read() {
        aggrvalue_vector30.read(hdr.atp_data.value30, (bit<32>)meta.aggIndex);
    }

    action vector31_read() {
        aggrvalue_vector31.read(hdr.atp_data.value31, (bit<32>)meta.aggIndex);
    }

    // vectorX_clean
    action vector00_clean() {
        aggrvalue_vector00.write((bit<32>)meta.aggIndex, 0);
    }

    action vector01_clean() {
        aggrvalue_vector01.write((bit<32>)meta.aggIndex, 0);
    }

    action vector02_clean() {
        aggrvalue_vector02.write((bit<32>)meta.aggIndex, 0);
    }

    action vector03_clean() {
        aggrvalue_vector03.write((bit<32>)meta.aggIndex, 0);
    }

    action vector04_clean() {
        aggrvalue_vector04.write((bit<32>)meta.aggIndex, 0);
    }

    action vector05_clean() {
        aggrvalue_vector05.write((bit<32>)meta.aggIndex, 0);
    }

    action vector06_clean() {
        aggrvalue_vector06.write((bit<32>)meta.aggIndex, 0);
    }

    action vector07_clean() {
        aggrvalue_vector07.write((bit<32>)meta.aggIndex, 0);
    }

    action vector08_clean() {
        aggrvalue_vector08.write((bit<32>)meta.aggIndex, 0);
    }

    action vector09_clean() {
        aggrvalue_vector09.write((bit<32>)meta.aggIndex, 0);
    }

    action vector10_clean() {
        aggrvalue_vector10.write((bit<32>)meta.aggIndex, 0);
    }

    action vector11_clean() {
        aggrvalue_vector11.write((bit<32>)meta.aggIndex, 0);
    }

    action vector12_clean() {
        aggrvalue_vector12.write((bit<32>)meta.aggIndex, 0);
    }

    action vector13_clean() {
        aggrvalue_vector13.write((bit<32>)meta.aggIndex, 0);
    }

    action vector14_clean() {
        aggrvalue_vector14.write((bit<32>)meta.aggIndex, 0);
    }

    action vector15_clean() {
        aggrvalue_vector15.write((bit<32>)meta.aggIndex, 0);
    }

    action vector16_clean() {
        aggrvalue_vector16.write((bit<32>)meta.aggIndex, 0);
    }

    action vector17_clean() {
        aggrvalue_vector17.write((bit<32>)meta.aggIndex, 0);
    }

    action vector18_clean() {
        aggrvalue_vector18.write((bit<32>)meta.aggIndex, 0);
    }

    action vector19_clean() {
        aggrvalue_vector19.write((bit<32>)meta.aggIndex, 0);
    }

    action vector20_clean() {
        aggrvalue_vector20.write((bit<32>)meta.aggIndex, 0);
    }

    action vector21_clean() {
        aggrvalue_vector21.write((bit<32>)meta.aggIndex, 0);
    }

    action vector22_clean() {
        aggrvalue_vector22.write((bit<32>)meta.aggIndex, 0);
    }

    action vector23_clean() {
        aggrvalue_vector23.write((bit<32>)meta.aggIndex, 0);
    }

    action vector24_clean() {
        aggrvalue_vector24.write((bit<32>)meta.aggIndex, 0);
    }

    action vector25_clean() {
        aggrvalue_vector25.write((bit<32>)meta.aggIndex, 0);
    }

    action vector26_clean() {
        aggrvalue_vector26.write((bit<32>)meta.aggIndex, 0);
    }

    action vector27_clean() {
        aggrvalue_vector27.write((bit<32>)meta.aggIndex, 0);
    }

    action vector28_clean() {
        aggrvalue_vector28.write((bit<32>)meta.aggIndex, 0);
    }

    action vector29_clean() {
        aggrvalue_vector29.write((bit<32>)meta.aggIndex, 0);
    }

    action vector30_clean() {
        aggrvalue_vector30.write((bit<32>)meta.aggIndex, 0);
    }

    action vector31_clean() {
        aggrvalue_vector31.write((bit<32>)meta.aggIndex, 0);
    }


    // NOTE: apply
    apply {
        switch_check.apply();
        if(meta.tobe_agg == 1 && hdr.atp.isValid()) {        // NOTE: 在此交换机上聚合的场景
            meta.aggIndex = hdr.atp.aggIndex;
            count_read();

            vector00_add(); vector01_add(); vector02_add(); vector03_add(); vector04_add(); vector05_add(); vector06_add(); vector07_add(); vector08_add(); vector09_add();
            vector10_add(); vector11_add(); vector12_add(); vector13_add(); vector14_add(); vector15_add(); vector16_add(); vector17_add(); vector18_add(); vector19_add();
            vector20_add(); vector21_add(); vector22_add(); vector23_add(); vector24_add(); vector25_add(); vector26_add(); vector27_add(); vector28_add(); vector29_add();
            vector30_add(); vector31_add();

            count_add();

            if((bit<5>)meta.count_value == hdr.atp.aggregationDegree) {
                count_clean();

                vector00_read(); vector01_read(); vector02_read(); vector03_read(); vector04_read(); vector05_read(); vector06_read(); vector07_read(); vector08_read(); vector09_read();
                vector10_read(); vector11_read(); vector12_read(); vector13_read(); vector14_read(); vector15_read(); vector16_read(); vector17_read(); vector18_read(); vector19_read();
                vector20_read(); vector21_read(); vector22_read(); vector23_read(); vector24_read(); vector25_read(); vector26_read(); vector27_read(); vector28_read(); vector29_read();
                vector30_read(); vector31_read();

                vector00_clean(); vector01_clean(); vector02_clean(); vector03_clean(); vector04_clean(); vector05_clean(); vector06_clean(); vector07_clean(); vector08_clean(); vector09_clean();
                vector10_clean(); vector11_clean(); vector12_clean(); vector13_clean(); vector14_clean(); vector15_clean(); vector16_clean(); vector17_clean(); vector18_clean(); vector19_clean();
                vector20_clean(); vector21_clean(); vector22_clean(); vector23_clean(); vector24_clean(); vector25_clean(); vector26_clean(); vector27_clean(); vector28_clean(); vector29_clean();
                vector30_clean(); vector31_clean();
                ipv4_lpm.apply();
            } else {
                drop();
            }
        } else {                            // IPv4
            if (hdr.atp.isValid() && hdr.ipv4.isValid()) {   // NOTE: 将要/已经在其他交换机上聚合进行转发
                aggregate_link_lpm.apply(); // TODO: 匹配将要转发的会 meta.tobe_agg = 1
            }
            if (hdr.ipv4.isValid() && meta.tobe_agg == 0) { // NOTE: 其他IP报文
                ipv4_lpm.apply();
            }
        }
    }
}

/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
    apply {  }
}

/*************************************************************************
*************   C H E C K S U M    C O M P U T A T I O N   **************
*************************************************************************/

control MyComputeChecksum(inout headers  hdr, inout metadata meta) {
     apply {
	update_checksum(
	    hdr.ipv4.isValid(),
            { hdr.ipv4.version,
	      hdr.ipv4.ihl,
              hdr.ipv4.diffserv,
              hdr.ipv4.totalLen,
              hdr.ipv4.identification,
              hdr.ipv4.flags,
              hdr.ipv4.fragOffset,
              hdr.ipv4.ttl,
              hdr.ipv4.protocol,
              hdr.ipv4.srcAddr,
              hdr.ipv4.dstAddr },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16);
    }
}

/*************************************************************************
***********************  D E P A R S E R  *******************************
*************************************************************************/

control MyDeparser(packet_out packet, in headers hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.atp);
        packet.emit(hdr.atp_data);
    }
}

/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

V1Switch(
MyParser(),
MyVerifyChecksum(),
MyIngress(),
MyEgress(),
MyComputeChecksum(),
MyDeparser()
) main;