import struct
from ctypes import Structure, c_ubyte, c_ushort, c_uint, c_byte, c_int
from scapy.all import *
from header_config import *
from utils.comm_utils import int_to_float


class NGA(Packet):
    name = "NGA"
    fields_desc = [
        BitField("worker_map", 0, WORKERMAPBIT),
        BitField("aggregation_degree", 0, DEGREEBIT),
        BitField("overflow", 0, OVERFLOWBIT),
        BitField("is_ack", 0, ISACKBIT),
        BitField("ecn", 0, ECNBIT),
        BitField("resend", 0, RESENDBIT),
        BitField("timestamp", 0, TIMEBIT),
        BitField("agg_index", 0, INDEXBIT),
        BitField("switch_id", 0, SWITCHIDBIT),
        BitField("sequence_id", 0, SEQUENCEBIT)
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


class NGAHeader(Structure):
    _fields_ = [
        ("version_ihl", c_ubyte),
        ("tos", c_ubyte),
        ("len", c_ushort),
        ("id", c_ushort),
        ("offset", c_ushort),
        ("ttl", c_ubyte),
        ("protocol_num", c_ubyte),
        ("sum", c_ushort),
        ("src", c_uint),
        ("dst", c_uint),
        ("worker_map", c_byte * 4),
        ("degree", c_ubyte),
        ("overflow_isack_ecn_resend_time", c_ubyte),
        ("index", c_byte * 4),
        ("switch_id", c_ubyte),
        ("sequence_id", c_byte * 4)
    ]

    def __new__(self, socket_buffer=None):
        return self.from_buffer_copy(socket_buffer)

    def __init__(self, socket_buffer=None):
        super().__init__()
        self.protocol_map = {1: "ICMP", 6: "TCP", 17: "UDP", 0x12: "NGA_TYPE"}

        # readable ip address
        self.src_address = socket.inet_ntoa(struct.pack("<I", self.src))
        self.dst_address = socket.inet_ntoa(struct.pack("<I", self.dst))
        # type of protocol
        try:
            self.protocol = self.protocol_map[self.protocol_num]
        except Exception as e:
            self.protocol = str(self.protocol_num)

        self.workermap = struct.unpack(">I", self.worker_map)[0]
        self.sequenceid = struct.unpack(">I", self.sequence_id)[0]
        self.switchid = self.switch_id
        self.aggregationdegree = self.degree
        self.aggindex = struct.unpack(">I", self.index)[0]


class NGAPayload(Structure):
    _fields_ = [
        ("payload", c_int * DATA_NUM)
    ]

    def __new__(self, socket_buffer=None):
        return self.from_buffer_copy(socket_buffer)

    def __init__(self, socket_buffer=None):
        super().__init__()
        self.data = []
        for i in range(DATA_NUM):
            self.data.append(self.payload[i])
        self.data = int_to_float(self.data)


class NGATotal(Structure):
    _fields_ = [
        ("version_ihl", c_ubyte),
        ("tos", c_ubyte),
        ("len", c_ushort),
        ("id", c_ushort),
        ("offset", c_ushort),
        ("ttl", c_ubyte),
        ("protocol_num", c_ubyte),
        ("sum", c_ushort),
        ("src", c_uint),
        ("dst", c_uint),
        ("worker_map", c_byte * 4),
        ("degree", c_ubyte),
        ("overflow_isack_ecn_resend_time", c_ubyte),
        ("index", c_byte * 4),
        ("switch_id", c_ubyte),
        ("sequence_id", c_byte * 4),
        ("payload", c_int * DATA_NUM)
    ]

    def __new__(self, socket_buffer=None):
        return self.from_buffer_copy(socket_buffer)
