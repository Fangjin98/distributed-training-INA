import argparse
import sys
import os
import time

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT_PATH = os.path.join(BASE_DIR, '../')
print(PROJECT_ROOT_PATH)

sys.path.append(PROJECT_ROOT_PATH)
sys.path.append("..")

from utils.DataManager import DataManager

parser = argparse.ArgumentParser(description='Gradient Sender')
parser.add_argument('--file', type=str)
parser.add_argument('--src_ip', type=str)
parser.add_argument('--dst_ip', type=str)
parser.add_argument('--worker_id', type=int)
parser.add_argument('--switch_id', type=int)
parser.add_argument('--degree', type=int)

args = parser.parse_args()
iface = 'eno6'

if __name__ == "__main__":
    print("file name: {}".format(args.file))
    print("send src_ip: {}, dst_ip: {} ...".format(args.src_ip, args.dst_ip))
    file_path = PROJECT_ROOT_PATH + 'data/log/tensor/' + args.file
    data = []
    with open(file_path, 'r') as gradient_file:
        data.append(float(gradient_file.readline()))
    datamanger = DataManager(args.src_ip, args.dst_ip, data, iface)
    print("Worker {} is sending packets...".format(args.worker_id))
    start_time = time.time()
    datamanger.fast_send_data(worker_id=args.worker_id, switch_id=args.switch_id, degree=args.degree)
    print("Finish, send time is {}".format(str(time.time() - start_time)))
