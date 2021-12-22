#ifndef _PROCESSOR_
#define _PROCESSOR_
#include "types.p4"
#include "headers.p4"


control Processor(
    in data_t value_in,
    out data_t value_out,
    in metadata_t meta) {

    Register<data_single_t,index_t>(NUM_REGISTER) value; 

    RegisterAction<data_single_t, index_t, data_t>(value) sum_read_value = {
        void apply(inout data_single_t value, out data_t out_value) {
            if(meta.count == 1){ //first packet
                value.first=value_in;
            }
            else{
                value.first  = value.first + value_in;
            }
            out_value = value.first;
        }
    };
    
    action sum_read_action() {
        value_out = sum_read_value.execute(meta.index);
    }


    // table sum {
    //     key = {
    //        meta.is_aggregation: exact;
    //     }
    //     actions = {
    //         sum_read_action;
    //         NoAction;
    //     }
    //     size = 1;
    //     const entries={
    //         (1):sum_read_action();
    //     }
    //     const default_action = NoAction;
    // }

    apply {
        sum_read_action();
    }
}

#endif 