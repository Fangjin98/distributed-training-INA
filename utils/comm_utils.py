import os
import sys
import struct
import socket
import pickle
import threading
from time import sleep
import time

from header_config import DATA_NUM, DATA_BYTE


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def killport(port):
    command = '''kill -9 $(netstat -nlp | grep :''' + str(
        port) + ''' | awk '{print $7}' | awk -F"/" '{ print $1 }')'''
    os.system(command)


def connect_send_socket(dst_ip, dst_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    while s.connect_ex((dst_ip, dst_port)) != 0:
        sleep(0.5)

    return s


def connect_get_socket(listen_ip, listen_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    start_time = time.time()
    while True:
        try:
            s.bind((listen_ip, listen_port))
            break
        except OSError as e:
            print(e)
            print("**OSError**", listen_ip, listen_port)
            sleep(0.7)
            killport(listen_port)
            if time.time() - start_time > 30:
                sys.exit(0)
    s.listen()

    conn, _ = s.accept()

    return conn


def send_data_socket(data, s):
    data = pickle.dumps(data)
    s.sendall(struct.pack(">I", len(data)))
    s.sendall(data)


def get_data_socket(conn):
    data_len = struct.unpack(">I", conn.recv(4))[0]
    data = conn.recv(data_len, socket.MSG_WAITALL)
    recv_data = pickle.loads(data)
    return recv_data


def get_data_from_nic(s, buffer, nic_ip=None):
    if nic_ip:
        print("Get data from nic {}...".format(nic_ip))
    while True:
        raw_data = s.recvfrom(DATA_NUM + DATA_BYTE)[0]
        buffer.append(raw_data)


def float_to_int(num_list):
    if num_list is None:
        return
    scale_factor = 10000
    res = []
    for num in num_list:
        res.append(int(num * scale_factor).to_bytes(4, byteorder='little', signed=True))
    return res


def int_to_float(num_list):
    scale_factor = 10000.0
    res = []
    for num in num_list:
        res.append(float(num / scale_factor))
    return res


class RecvThread(threading.Thread):
    def __init__(self, func, args=()):
        super(RecvThread, self).__init__()
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        threading.Thread.join(self)
        return self.result
