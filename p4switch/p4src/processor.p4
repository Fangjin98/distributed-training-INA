#ifndef _PROCESSOR_
#define _PROCESSOR_
#include "types.p4"
#include "headers.p4"


control Processor(
    in data_t value_in,
    out data_t value_out,
    in metadata_t meta) {

    Register<data_single_t,index_t>(register_size) values; 

    // Compute sum of both values and read first one
    RegisterAction<data_single_t, index_t, data_t>(values) sum_read_register_action = {
        void apply(inout data_single_t value, out data_t read_value) {
            if(meta.count_value == 1){ //the first value
                value.first=value_in;
            }
            else{
                value.first  = value.first + value_in;
            }
            read_value = value.first;
        }
    };

    action sum_read_action() {
        value_out = sum_read_register_action.execute(meta.aggIndex);
    }

    // Read first sum register
    RegisterAction<data_single_t, index_t, data_t>(values) read_register_action = {
        void apply(inout data_single_t value, out data_t read_value) {
            read_value = value.first;
        }
    };

    action read_action() {
        value_out = read_register_action.execute(meta.aggIndex);
    }

    table sum {
        key = {
           meta.tobe_agg: exact;
        }
        actions = {
            sum_read_action;
            read_action;
            NoAction;
        }
        size = 1;
        const entries={
            (1):sum_read_action();
        }
        const default_action = NoAction;
    }

    apply {
        sum.apply();
    }
}

#endif 