import argparse
import sys

sys.path.append("..")

from utils.DataManager import DataManager

parser = argparse.ArgumentParser(description='Packet Sender')
parser.add_argument('--src-ip', type=str, default='172.16.170.3')
parser.add_argument('--dst-ip', type=str, default='172.16.170.2')
parser.add_argument('--iface', type=str, default='eno6')

args = parser.parse_args()

if __name__ == "__main__":
    dst_ip = args.dst_ip
    src_ip = args.src_ip
    iface = args.iface
    print("send src_ip: {}, dst_ip: {}, iface: {}.".format(src_ip, dst_ip, iface))
    data = [0.01 for i in range(10000)]
    datamanger = DataManager(src_ip, dst_ip, data, iface)
    print("Sending packets...")
    datamanger.fast_send_data(1, 2, 3)
