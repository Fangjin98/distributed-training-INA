import argparse
import asyncio
import concurrent.futures
import json
import random
from functools import singledispatch
from re import L
import numpy as np
import torch
from config import *
from header_config import NGA_TYPE
from utils.comm_utils import *
from utils import datasets, models
from utils.training_utils import test
from utils.NGAPacket import *
from utils.comm_utils import *

# init parameters
parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_pattern', type=int, default=0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--algorithm', type=str, default='proposed')
parser.add_argument('--ratio', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--decay_rate', type=float, default=0.98)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--action_num', type=int, default=9)
parser.add_argument('--worker_num', type=int, default=8)
parser.add_argument('--use_cuda', action="store_false", default=True)
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--nic_ip', type=str, default='127.0.0.1')
parser.add_argument('--write_to_file',  default=False)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")


def main():
    offset = random.randint(0, 20) * 20
    print(offset)

    common_config = CommonConfig('CIFAR10',
                                 args.model,
                                 args.epoch,
                                 args.batch_size,
                                 args.lr,
                                 args.decay_rate,
                                 args.step_size,
                                 args.ratio,
                                 args.algorithm,
                                 write_to_file=args.write_to_file,
                                 use_cuda=args.use_cuda,
                                 master_listen_port_base=53300+offset
                                 )

    with open("worker_config.json") as json_file:
        workers_config = json.load(json_file)

    # global_model = models.create_model_instance(common_config.dataset, common_config.model)
    global_model = models.get_model(common_config.model)
    init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
    para_nums = torch.nn.utils.parameters_to_vector(global_model.parameters()).nelement()
    model_size = init_para.nelement() * 4 / 1024 / 1024
    print("Model name: {}".format(common_config.model))
    print("Model Size: {} MB".format(model_size))

    worker_num = min(args.worker_num, len(workers_config['worker_config_list']))

    worker_list = []
    for i in range(worker_num):
        worker_config = workers_config['worker_config_list'][i]
        custom = dict()
        custom["computation"] = worker_config["computation"]
        custom["dynamics"] = worker_config["dynamics"]
        worker_list.append(
            Worker(config=ClientConfig(idx=i,
                                       client_host=worker_config["host"],
                                       client_ip=worker_config["ip"],
                                       client_nic_ip=worker_config["nic_ip"],
                                       ssh_port=worker_config["ssh_port"],
                                       client_user=worker_config["user"],
                                       client_pwd=worker_config['pwd'],
                                       master_ip=args.ip,
                                       master_port=common_config.master_listen_port_base + i,
                                       master_nic_ip=args.nic_ip,
                                       custom=custom),
                   common_config=common_config,
                   user_name=worker_config['name'],
                   para_nums=para_nums
                   )
        )

    train_data_partition, test_data_partition = partition_data(common_config.dataset, args.data_pattern, worker_num)

    # nic_socket = socket.socket(socket.AF_INET, socket.SOCK_RAW, NGA_TYPE)
    # nic_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10240000)
    # print("Recv buff: {}".format(nic_socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)))
    # nic_socket.bind((args.nic_ip, 0))
    # updated_para = []
    # recv_thread = RecvThread(func=get_data_from_nic, args=(nic_socket, updated_para, args.nic_ip))
    # recv_thread.start()

    for worker_idx, worker in enumerate(worker_list):
        worker.config.para = init_para
        worker.config.custom["train_data_idxes"] = train_data_partition.use(worker_idx)
        worker.config.custom["test_data_idxes"] = test_data_partition.use(worker_idx)
    print("Try to connect socket and send init config.")
    try:
        communication_parallel(worker_list, action="init")
    except Exception as e:
        for worker in worker_list:
            worker.socket.shutdown(2)
        return
    else:
        print("SUCCESSFUL: inition done.")

    global_model.to(device)
    train_dataset, test_dataset = datasets.load_datasets(common_config.dataset)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, shuffle=False)

    total_time = 0.0

    for epoch_idx in range(1, 1 + common_config.epoch):
        print("get begin")
        communication_parallel(worker_list, action="get_model")
        print("get end")
        global_para = torch.nn.utils.parameters_to_vector(global_model.parameters()).clone().detach()
        # tmp_para = aggregate_model_from_nic(global_para, updated_para, args.step_size, worker_num)
        # updated_para.clear()
        global_para = aggregate_model(global_para, worker_list, args.step_size)

        print("send begin")
        communication_parallel(worker_list, action="send_model", data=global_para)
        # communication_parallel(worker_list, action="send_model", data=tmp_para)
        print("send end")

        torch.nn.utils.vector_to_parameters(global_para, global_model.parameters())
        test_loss, acc = test(global_model, test_loader, device, model_type=args.model)
        common_config.recoder.add_scalar('Accuracy/average', acc, epoch_idx)
        common_config.recoder.add_scalar('Test_loss/average', test_loss, epoch_idx)
        print("Epoch: {}, accuracy: {}, test_loss: {}\n".format(epoch_idx, acc, test_loss))
        common_config.recoder.add_scalar('Accuracy/average_time', acc, total_time)
        common_config.recoder.add_scalar('Test_loss/average_time', test_loss, total_time)

    for worker in worker_list:
        worker.socket.shutdown(2)


def aggregate_model(local_para, worker_list, step_size):
    with torch.no_grad():
        para_delta = torch.zeros_like(local_para)
        average_weight = 1.0 / (len(worker_list) + 1)
        for worker in worker_list:
            model_delta = (worker.config.neighbor_paras - local_para)
            para_delta += step_size * average_weight * model_delta

        local_para += para_delta

    return local_para


def get_nic_data(recv_data):
    sequence_payload = []
    count = 0
    for raw_data in recv_data:
        nga_header = NGAHeader(raw_data[:HEADER_BYTE])
        if nga_header.sequenceid == -1:
            continue
        nga_payload = NGAPayload(raw_data[HEADER_BYTE:])
        count += len(nga_payload.data)
        sequence_payload.append((nga_header.sequenceid, nga_payload.data))
    print("Len of data from nic: {}.".format(count))
    return sequence_payload


def aggregate_model_from_nic(local_para, recv_data, step_size, worker_num):
    header_payload = get_nic_data(recv_data)
    updated_para = [0.0 for i in range(len(local_para))]
    count=0
    for hp in header_payload:
        for i in range(DATA_NUM):
            if hp[0] * DATA_NUM + i == len(local_para):
                break
            updated_para[hp[0] * DATA_NUM + i] += hp[1][i]
            count+=1
    updated_para = torch.Tensor(updated_para)
    print(count)

    with torch.no_grad():
        delta = (updated_para - local_para)
        local_para += ((step_size * delta) / (worker_num + 1))

    return local_para


def communication_parallel(worker_list, action, data=None):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_list), )
        tasks = []
        for worker in worker_list:
            if action == "init":
                tasks.append(loop.run_in_executor(executor, worker.send_init_config))
            elif action == "get_model":
                tasks.append(loop.run_in_executor(executor, get_compressed_model_top, worker.config, worker.socket,
                                                  worker.para_nums))
            elif action == "send_model":
                tasks.append(loop.run_in_executor(executor, worker.send_data, data))
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
    except:
        sys.exit(0)


def get_compressed_model_top(config, socket, nelement):
    print('Get compressed model')
    try:
        received_para = get_data_socket(socket)
    except Exception as e:
        print("FAILED: get model error.")
        print(e)
    else:
        received_para.to(device)
        config.neighbor_paras = received_para


def non_iid_partition(ratio, worker_num=10):
    partition_sizes = np.ones((10, worker_num)) * ((1 - ratio) / (worker_num - 1))

    for worker_idx in range(worker_num):
        partition_sizes[worker_idx][worker_idx] = ratio

    return partition_sizes


def partition_data(dataset_type, data_pattern, worker_num):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)

    if dataset_type == "CIFAR100":
        test_partition_sizes = np.ones((100, worker_num)) * (1 / worker_num)
        partition_sizes = np.ones((100, worker_num)) * (1 / (worker_num - data_pattern))
        for worker_idx in range(worker_num):
            tmp_idx = worker_idx
            for _ in range(data_pattern):
                partition_sizes[tmp_idx * worker_num:(tmp_idx + 1) * worker_num, worker_idx] = 0
                tmp_idx = (tmp_idx + 1) % 10
    elif dataset_type == "CIFAR10":
        test_partition_sizes = np.ones((10, worker_num)) * (1 / worker_num)
        if data_pattern == 0:
            partition_sizes = np.ones((10, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            partition_sizes = [
                [0.0, 0.0, 0.0, 0.1482, 0.1482, 0.1482, 0.148, 0.1482, 0.1482, 0.111],
                [0.0, 0.0, 0.0, 0.1482, 0.1482, 0.1482, 0.1482, 0.148, 0.1482, 0.111],
                [0.0, 0.0, 0.0, 0.1482, 0.1482, 0.1482, 0.1482, 0.1482, 0.148, 0.111],
                [0.148, 0.1482, 0.1482, 0.0, 0.0, 0.0, 0.1482, 0.1482, 0.1482, 0.111],
                [0.1482, 0.148, 0.1482, 0.0, 0.0, 0.0, 0.1482, 0.1482, 0.1482, 0.111],
                [0.1482, 0.1482, 0.148, 0.0, 0.0, 0.0, 0.1482, 0.1482, 0.1472, 0.112],
                [0.1482, 0.1482, 0.1482, 0.148, 0.1482, 0.1482, 0.0, 0.0, 0.0, 0.111],
                [0.1482, 0.1482, 0.1482, 0.1482, 0.148, 0.1482, 0.0, 0.0, 0.0, 0.111],
                [0.1482, 0.1482, 0.1482, 0.1482, 0.1482, 0.148, 0.0, 0.0, 0.0, 0.111],
                [0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.112, 0.0],
            ]
        elif data_pattern == 2:
            partition_sizes = [
                [0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                [0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                [0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                [0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                [0.125, 0.125, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125],
                [0.125, 0.125, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125],
                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125],
                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125],
                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0],
                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0],
            ]
        elif data_pattern == 3:
            partition_sizes = [[0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0, 0.0, 0.0],
                               [0.0, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0, 0.0],
                               [0.0, 0.0, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0],
                               [0.0, 0.0, 0.0, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432],
                               [0.1432, 0.0, 0.0, 0.0, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428],
                               [0.1428, 0.1432, 0.0, 0.0, 0.0, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428],
                               [0.1428, 0.1428, 0.1432, 0.0, 0.0, 0.0, 0.1428, 0.1428, 0.1428, 0.1428],
                               [0.1428, 0.1428, 0.1428, 0.1432, 0.0, 0.0, 0.0, 0.1428, 0.1428, 0.1428],
                               [0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0, 0.0, 0.0, 0.1428, 0.1428],
                               [0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0, 0.0, 0.0, 0.1428],
                               ]
        elif data_pattern == 4:
            partition_sizes = [[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0],
                               [0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0],
                               [0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                               [0.125, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                               [0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                               [0.125, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125],
                               [0.125, 0.125, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125],
                               [0.125, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125],
                               [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125],
                               [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0, 0.125],
                               ]
        elif data_pattern == 5:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 6:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 7:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 8:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 9:
            non_iid_ratio = 0.9
            partition_sizes = non_iid_partition(non_iid_ratio)
        # elif data_pattern == 10:
        #     non_iid_ratio = 0.5
        #     partition_sizes = non_iid_partition(non_iid_ratio)

    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=test_partition_sizes)

    return train_data_partition, test_data_partition


if __name__ == "__main__":
    print(np.random.rand(10) * 10)
    print(np.random.randint(1, 10, 10))
    print(np.random.randint(1, 10, 10))
    print(np.random.normal(loc=1, scale=np.sqrt(2), size=(1, 10)))
    main()
