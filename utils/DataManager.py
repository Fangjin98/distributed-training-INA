import struct
import time
from operator import mod

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
    def __init__(self, src_ip, dst_ip, data=None, interface='eth0'):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.data = float_to_int(data)
        self.iface = my_get_if(interface)

    def get_data(self, fliter=None):
        return sniff(iface=self.iface, filter=fliter, prn=lambda x: check_pkt(x))

    def send_data(self, worker_id, switch_id, degree):
        print("send to nic...")

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_RAW, NGA_TYPE)
        except OSError as e:
            print(e)
            sys.exit(1)
        else:
            start_time = time.time()
            data_num = len(self.data)
            for i in range(DATA_NUM - mod(data_num, DATA_NUM)):
                self.data.append(int(0).to_bytes(4, byteorder='little', signed=True))
            for i, index in enumerate(range(0, len(self.data), DATA_NUM)):
                left = index
                nga = struct.pack(
                    'IBBBBI', worker_id, degree, 0, 0, switch_id, i
                )
                for d in self.data[left:left + DATA_NUM]:
                    nga += d
                s.sendto(nga, (self.dst_ip, 0))
            nga_end = struct.pack(
                'IBBBBi', worker_id, degree, 0, 0, switch_id, -1
            )
            # for d in range(DATA_NUM):
            #     nga_end += int(0).to_bytes(4, byteorder='little', signed=True)
            s.sendto(nga_end, (self.dst_ip, 0))
            total_time = time.time() - start_time

        print("END: send to nic.")
        print("Total time: {}.".format(total_time))

    def update_data(self, new_data):
        self.data = float_to_int(new_data)
