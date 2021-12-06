#ifndef _COUNTER_
#define _COUNTER_
#include "types.p4"
#include "headers.p4"


control My_Counter(
    in metadata_t ig_md,
    in bit<8> aggregationDegree,
    out bit<8> value_out
    ) {

    
    table switch_count{
        actions = {
            count_action;
        }
        key = {
            ig_md.is_count: exact;
        }
        size = 2;
        default_action = count_action;
    }

    apply {
        switch_count.apply();
    }
}

#endif 