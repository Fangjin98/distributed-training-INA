
import os
import struct
import socket
import pickle
from time import sleep
import time


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def killport(port):
    command = (
        """kill -9 $(netstat -nlp | grep :"""
        + str(port)
        + """ | awk '{print $7}' | awk -F"/" '{ print $1 }')"""
    )
    os.system(command)


def bind_port(listen_ip, listen_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    start_time = time.time()
    while True:
        try:
            s.bind((listen_ip, listen_port))
            break
        except OSError as e:
            print("**OSError**", listen_ip, listen_port)
            sleep(0.7)
            killport(listen_port)
            if time.time() - start_time > 30:
                exit(1)
    s.listen()
    conn, _ = s.accept()
    return conn


def send_timestamp_data(s, timestamp, data):
    s.sendall(struct.pack(">d", timestamp))
    ser_data = pickle.dumps(data)
    s.sendall(struct.pack(">I", len(ser_data)))
    s.sendall(ser_data)


def get_data(s):
    data_len = struct.unpack(">I", s.recv(4))[0]
    data = s.recv(data_len, socket.MSG_WAITALL)
    recv_data = pickle.loads(data)
    return recv_data


def get_data_from_nic(s, buffer):
    while True:
        raw_data = s.recvfrom(HEADER_BYTE + DATA_BYTE)[0]
        buffer.append(raw_data)
