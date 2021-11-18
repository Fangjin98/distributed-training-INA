from scapy.all import *
from scapy.layers.inet import IP
from config import *


class NGA(Packet):
    name = "NGA"
    fields_desc = [
        BitField("worker_map", 0, WORKERMAPBIT),
        BitField("aggregation_degree", 0, DEGREEBIT),
        BitField("overflow", 0, OVERFLOWBIT),
        BitField("is_ack", 0, ISACKBIT),
        BitField("ecn", 0, ECNBIT),
        BitField("resend", 0, RESENDBIT),
        BitField("agg_index", 0, INDEXBIT),
        BitField("timestamp", 0, TIMEBIT),
        BitField("switch_id", 0, SWITCHIDBIT),
        BitField("sequence_id", 0, SEQUENCEBIT),
    ]


class NGAData(Packet):
    name = "NGA_data"
    fields_desc = [
        IntField("d00", 0),
        IntField("d01", 0),
        IntField("d02", 0),
        IntField("d03", 0),
        IntField("d04", 0),
        IntField("d05", 0),
        IntField("d06", 0),
        IntField("d07", 0),
        IntField("d08", 0),
        IntField("d09", 0),
        IntField("d10", 0),
        IntField("d11", 0),
        IntField("d12", 0),
        IntField("d13", 0),
        IntField("d14", 0),
        IntField("d15", 0),
        IntField("d16", 0),
        IntField("d17", 0),
        IntField("d18", 0),
        IntField("d19", 0),
        IntField("d20", 0),
        IntField("d21", 0),
        IntField("d22", 0),
        IntField("d23", 0),
        IntField("d24", 0),
        IntField("d25", 0),
        IntField("d26", 0),
        IntField("d27", 0),
        IntField("d28", 0),
        IntField("d29", 0),
        IntField("d30", 0),
        IntField("d31", 0),
    ]


bind_layers(IP, NGA, proto=TYPE_NGA)
bind_layers(NGA, NGAData)
