import struct
import time
from operator import mod

from scapy.all import *
from scapy.layers.inet import IP
from scapy.layers.l2 import Ether
from utils.NGAPacket import *
from utils.comm_utils import int_to_float, float_to_int
from concurrent.futures import ThreadPoolExecutor


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
    def __init__(self, src_ip, dst_ip, data=None, interface='eth0', thread_num=4):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.data = float_to_int(data)
        self.iface = my_get_if(interface)
        self.thread_num = thread_num
        self.thread_pool = ThreadPoolExecutor(max_workers=self.thread_num)

    def fast_send_data(self, worker_id, switch_id, degree, send_step=10000):
        """
        Using multi-threads to send packets
        Returns: None
        """
        print("send to nic...")
        total_len = len(self.data)
        offset = 0
        sequence_num = 0
        step = send_step
        tail = mod(total_len, self.thread_num)
        thread_res = []
        start_time = time.time()

        if total_len <= DATA_NUM * step:
            self.send_data(worker_id, switch_id, degree)
        else:  # each thread sends DATA_NUM*step number of packets.
            for i in range(0, total_len, DATA_NUM * step):
                if offset + DATA_NUM * step > total_len:
                    break
                res = self.thread_pool.submit(self._send_data,
                                              worker_id,
                                              switch_id,
                                              degree,
                                              offset,
                                              DATA_NUM * step,
                                              sequence=sequence_num)
                offset += DATA_NUM * step
                sequence_num += step
                # thread_res.append(res)
            res = self.thread_pool.submit(self._send_data,
                                          worker_id,
                                          switch_id,
                                          degree,
                                          offset,
                                          tail,
                                          sequence_num,
                                          True)
            # thread_res.append(res)

        # while True:
        #     done = True
        #     for t in thread_res:
        #         # print(t.exception())
        #         done = t.done()
        #     if done:
        #         break
        total_time = time.time() - start_time
        print("END: send to nic.")
        print("Total time: {}.".format(total_time))

    def send_data(self, worker_id, switch_id, degree):
        print("send to nic...")
        start_time = time.time()
        self._send_data(worker_id, switch_id, degree, 0, len(self.data), True)
        total_time = time.time() - start_time
        print("END: send to nic.")
        print("Total time: {}.".format(total_time))

    def _send_data(self, worker_id, switch_id, degree, offset, step, sequence=0, end=False):
        # print("{} is sending data...".format(threading.current_thread().name))
        s = socket.socket(socket.AF_INET, socket.SOCK_RAW, NGA_TYPE)
        left = 0
        pkt_id = 0
        for i, index in enumerate(range(offset, offset + step, DATA_NUM)):
            left = index
            pkt_id = i
            if left + DATA_NUM > offset + step:
                break
            nga = struct.pack(
                'IBBBBi', worker_id, degree, 0, 0, switch_id, sequence + pkt_id
            )
            for d in self.data[left:left + DATA_NUM]:
                nga += d
            s.sendto(nga, (self.dst_ip, 0))

        if mod(step, DATA_NUM) != 0:  # tail packets
            nga = struct.pack(
                'IBBBBi', worker_id, degree, 0, 0, switch_id, sequence + pkt_id
            )
            for d in self.data[left:]:
                nga += d
            for i in range(DATA_NUM - mod(step, DATA_NUM)):
                nga += int(0).to_bytes(4, byteorder='little', signed=True)
            s.sendto(nga, (self.dst_ip, 0))

        if end is True:
            nga_end = struct.pack(
                'IBBBBi', worker_id, degree, 0, 0, switch_id, -1
            )
            s.sendto(nga_end, (self.dst_ip, 0))
        # print("{} is done".format(threading.current_thread().name))

    def update_data(self, new_data):
        self.data = float_to_int(new_data)
