#ifndef _FRAGCHECK_
#define _FRAGCHECK_
#include "types.p4"
#include "headers.p4"


control Fragcheck(
    in b32_t frag_id_in,
    out b32_t frag_id_out,
    in metadata_t meta) {

    Register<b32_t,index_t>(NUM_REGISTER) frag_id; 

    RegisterAction<b32_t, index_t, b32_t>(frag_id) write_read_id = {
        void apply(inout b32_t value, out b32_t out_value) {
            if(value==0){ //unused
                value=frag_id_in;
                out_value=value;
            }
            else{
                out_value=value;
            }
        }
    };

    RegisterAction<b32_t, index_t, b32_t>(frag_id) reset_id = {
        void apply(inout b32_t value, out b32_t out_value) {
            value=0;
            out_value=frag_id_in;
        }
    };
    
    action check_action() {
        frag_id_out=write_read_id.execute(meta.index);
    }

    action reset_action(){
        frag_id_out=reset_id.execute(meta.index);
    }

    table check {
        key = {
           meta.is_aggregation : exact;
           meta.is_ack : ternary;
        }
        actions = {
            check_action;
            reset_action;
            NoAction;
        }
        size = 2;
        const entries={
            (1,1):reset_action();
            (1,0):check_action();
        }
        const default_action = NoAction;
    }

    apply {
        check.apply();
    }
}

#endif 