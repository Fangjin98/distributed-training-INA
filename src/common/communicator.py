from ctypes import *
from multiprocessing import Pool
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import threading

TENSOR_NUM_PER_PACKET = 128
AGGREGATOR_SIZE = 199665
PARA_LEN = 25557032

dst_ip_str = "172.16.210.33"

_send = cdll.LoadLibrary("./send.so") 

_send.send_gradients.argtypes = [
    POINTER(c_uint32), 
    c_int, 
    c_uint32, 
    c_int, 
    c_uint32,
    c_int
]


def ip2int(ip):
    ip_list = ip.strip().split('.')
    ip_int = int(ip_list[0])*256**3+int(ip_list[1])*256**2+int(ip_list[2])*256**1+int(ip_list[3])*256**0
    return ip_int

def c_send_wrapper(gradient: "numpy.array", packet_num, dst_ip: int, worker_id, aggregator_index, tensor_index: int):
    c_pointer_gradient=gradient.ctypes.data_as(POINTER(c_uint32))
    c_packet_num=c_int(packet_num)
    c_dst_ip=c_uint32(dst_ip)
    c_worker_id=c_int(worker_id)
    c_aggregator_index=c_uint32(aggregator_index)
    c_tensor_index=c_int(tensor_index)
    _send.send_gradients(c_pointer_gradient, c_packet_num, c_dst_ip, c_worker_id, c_aggregator_index, c_tensor_index)

def single_process_send(data):
    c_send_wrapper(data, int(len(data) / TENSOR_NUM_PER_PACKET), ip2int(dst_ip_str),0,0,0)
    
def multi_process_send(process_num, data):
    start=time.time()

    process_pool = Pool(process_num)
    total_packet = int(len(data) / TENSOR_NUM_PER_PACKET)
    packet_num_per_process= int(total_packet / process_num)
    remained_packets= int(total_packet % process_num)
    offset=0

    for i in range(process_num):
        if i != process_num-1:
            process_pool.apply_async(c_send_wrapper, (data[offset: offset+packet_num_per_process * TENSOR_NUM_PER_PACKET],  packet_num_per_process, ip2int(dst_ip_str), 0, 0,offset))
        else:
            process_pool.apply_async(c_send_wrapper, (data[offset : ], packet_num_per_process + remained_packets, ip2int(dst_ip_str), 0, 0, offset))
        
        offset+=packet_num_per_process * TENSOR_NUM_PER_PACKET
    
    process_pool.close()
    process_pool.join()

    end=time.time()
    print("{} processes cost: {} sec; Throuthput {} GBps".format(str(process_num), str(end-start), str(data_size/(end-start))))

# multi process send through concurrent.futures.ThreadPoolExecutor
def multi_thread_send_futures(process_num, data):
    start=time.time()

    executor=ThreadPoolExecutor()
    total_packet = int(len(data) / TENSOR_NUM_PER_PACKET)
    packet_num_per_process= int(total_packet / process_num)
    remained_packets= int(total_packet % process_num)
    offset=0
    f = []

    for i in range(process_num):
        if i != process_num-1:
            f.append(executor.submit(c_send_wrapper, data[offset: offset+packet_num_per_process * TENSOR_NUM_PER_PACKET],  packet_num_per_process, ip2int(dst_ip_str), 0, 0,offset))
        else:
            f.append(executor.submit(c_send_wrapper, data[offset : ], packet_num_per_process + remained_packets, ip2int(dst_ip_str), 0, 0, offset))
        
        offset+=packet_num_per_process * TENSOR_NUM_PER_PACKET
    
    
    executor.shutdown(wait=True)
    # for i in range(process_num):
    #     print(f'task{i}是否完成: {f[i].done()}')
    end=time.time()
    print("{} processes cost: {} sec; Throuthput {} GBps".format(str(process_num), str(end-start), str(data_size/(end-start))))

# multi process send through concurrent.features.ProcessPoolExecutor
def multi_process_send_futures_P(process_num, data):
    start=time.time()

    executor=ProcessPoolExecutor()
    total_packet = int(len(data) / TENSOR_NUM_PER_PACKET)
    packet_num_per_process= int(total_packet / process_num)
    remained_packets= int(total_packet % process_num)
    offset=0
    f = []

    for i in range(process_num):
        if i != process_num-1:
            f.append(executor.submit(c_send_wrapper, data[offset: offset+packet_num_per_process * TENSOR_NUM_PER_PACKET],  packet_num_per_process, ip2int(dst_ip_str), 0, 0,offset))
        else:
            f.append(executor.submit(c_send_wrapper, data[offset : ], packet_num_per_process + remained_packets, ip2int(dst_ip_str), 0, 0, offset))
        
        offset+=packet_num_per_process * TENSOR_NUM_PER_PACKET
    
    
    executor.shutdown(wait=True)
    # for i in range(process_num):
    #     print(f'task{i}是否完成: {f[i].done()}')
    end=time.time()
    print("{} processes cost: {} sec; Throuthput {} GBps".format(str(process_num), str(end-start), str(data_size/(end-start))))

# multi process send through threading
class myThread(threading.Thread):
    def __init__(self, threadID, gradient, packet_num, dst_ip, worker_id, aggregator_index, tensor_index):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.gradient = gradient
        self.packet_num = packet_num
        self.dst_ip = dst_ip
        self.worker_id = worker_id
        self.aggregator_index = aggregator_index
        self.tensor_index = tensor_index
    def run(self):
        c_send_wrapper(self.gradient, self.packet_num, self.dst_ip, self.worker_id, self.aggregator_index, self.tensor_index)

def multi_thread_send_threading(process_num, data):
    start=time.time()

    total_packet = int(len(data) / TENSOR_NUM_PER_PACKET)
    packet_num_per_process= int(total_packet / process_num)
    remained_packets= int(total_packet % process_num)
    offset=0
    f = []

    for i in range(process_num):
        if i != process_num-1:
            f.append(myThread(i, data[offset: offset+packet_num_per_process * TENSOR_NUM_PER_PACKET],  packet_num_per_process, ip2int(dst_ip_str), 0, 0,offset))
            f[i].start()
        else:
            f.append(myThread(i, data[offset : ], packet_num_per_process + remained_packets, ip2int(dst_ip_str), 0, 0, offset))
            f[i].start()
        
        offset+=packet_num_per_process * TENSOR_NUM_PER_PACKET
    
    
    for i in range(process_num):
        f[i].join()
    
    end=time.time()
    print("{} processes cost: {} sec; Throuthput {} GBps".format(str(process_num), str(end-start), str(data_size/(end-start))))

