import argparse
import socket
import sys

import torch

from header_config import *
from utils.NGAPacket import *

sys.path.append("..")

from utils.comm_utils import *

ETH_P_ALL = 0x3

parser = argparse.ArgumentParser(description='Packet Sender')
parser.add_argument('--ip', type=str, default='172.16.170.3')

args = parser.parse_args()

if __name__ == "__main__":
    listen_ip = args.ip
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_RAW, NGA_TYPE)
    except OSError as e:
        print(e)
        sys.exit(1)
    else:
        s.bind((listen_ip, 0))
    print("Get data...")
    while True:
        raw_data = s.recvfrom(HEADER_BYTE + DATA_BYTE)[0]
        nga_header = NGAHeader(raw_data[:HEADER_BYTE])
        print("Workerid and sequenceid: {} {}".format(nga_header.workermap, nga_header.sequenceid))
        if nga_header.sequenceid == -1:
            break
        nga_payload = NGAPayload(raw_data[HEADER_BYTE:])
        # print("Protocol: {} {}->{}".format(nga_header.protocol, nga_header.src_address, nga_header.dst_address))
        print("Payload:")
        for index, d in enumerate(nga_payload.data):
            print(index, d)
        # tensor = torch.Tensor(nga_payload.data)
        # print(tensor)
