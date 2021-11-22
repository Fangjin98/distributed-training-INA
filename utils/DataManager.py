import struct
import time

from scapy.all import *
from scapy.layers.inet import IP
from scapy.layers.l2 import Ether
from utils.NGAPacket import *
from utils.comm_utils import int_to_float, float_to_int
from impacket import ImpactDecoder, ImpactPacket


def my_get_if(interface="eth0"):
    if_list = get_if_list()
    for iface in if_list:
        if iface == interface:
            return interface

    print("Cannot find interface {}".format(interface))
    print("Probably try {}".format([iface for iface in if_list]))
    exit(1)


def check_pkt(pkt):
    if (NGA in pkt) or (IP in pkt):
        print("got a packet:")
        pkt.show()
        hexdump(pkt)
        print("len(pkt) = ", len(pkt))
        sys.stdout.flush()


class DataManager:
    def __init__(self, src_ip, dst_ip, data, interface='eth0'):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.data = float_to_int(data)
        self.iface = my_get_if(interface)
        print(self.data[0])
        print("datamanager init done.")

    def get_data(self, fliter=None):
        return sniff(iface=self.iface, filter=fliter, prn=lambda x: check_pkt(x))

    def send_data(self, worker_id, switch_id, degree):
        print("send to nic...")
        print("Len: {}".format(len(self.data)))
        # args = ["d00", "d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09", "d10", "d11", "d12", "d13",
        #         "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        #         "d28", "d29", "d30", "d31"]
        # mac = get_if_hwaddr(self.iface)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_RAW, NGA_TYPE)
        except OSError as e:
            print(e)
            sys.exit(1)
        else:
            start_time=time.time()
            for i, index in enumerate(range(0, len(self.data), DATA_NUM)):
                left = index
                right = index + DATA_NUM if (index + DATA_NUM <= len(self.data)) else len(self.data)
                nga = struct.pack(
                    'IBBBBI', worker_id, degree, 0, 0, switch_id, i
                )
                for d in self.data[left:right]:
                    nga += d
                if right-left < DATA_NUM:
                    for j in range(DATA_NUM-right+left):
                        nga += struct.pack('L', 0)
                s.sendto(nga, (self.dst_ip, 0))
            total_time=time.time()-start_time

        print("END: send to nic.")
        print("Total time: {}.".format(total_time))

    def update_data(self, new_data):
        self.data = new_data

    # def _partition_data(self, worker_id, switch_id, degree):
    #     packet_list = []
    #     for i, index in enumerate(range(0, len(self.data), DATA_NUM)):
    #         left = index
    #         right = index + DATA_NUM if (index + DATA_NUM <= len(self.data)) else len(self.data)
    #
    #         args = ["d00", "d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09", "d10", "d11", "d12", "d13",
    #                 "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
    #                 "d28", "d29", "d30", "d31"]
    #
    #         packet_list.append(
    #             Ether(src=get_if_hwaddr(self.iface), dst='ff:ff:ff:ff:ff:ff') /
    #             IP(src=self.src_ip, dst=self.dst_ip, proto=NGA_TYPE) /
    #             NGA(worker_map=worker_id,
    #                 aggregation_degree=degree,
    #                 timestamp=i,
    #                 agg_index=i,
    #                 sequence_id=i,
    #                 switch_id=switch_id) /
    #             NGAData(**dict(zip(args, self.data[left:right])))
    #         )
    #     return packet_list
