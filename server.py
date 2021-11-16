import argparse
import asyncio
import concurrent.futures
import json
import random
from re import L
import numpy as np
import torch
from config import *
from utils.comm_utils import *
from utils import datasets, models
from utils.training_utils import test

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
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--decay_rate', type=float, default=0.993)
parser.add_argument('--epoch', type=int, default=2)
parser.add_argument('--action_num', type=int, default=9)
parser.add_argument('--worker_num', type=int, default=5)
parser.add_argument('--use_cuda', action="store_false", default=True)
parser.add_argument('--ip',type=str,default='127.0.0.1')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

global_is_end = 0


def main():
    offset = random.randint(0, 20) * 20
    print(offset)

    server_ip=args.ip

    common_config = CommonConfig('CIFAR10',
                                 args.model,
                                 args.epoch,
                                 args.batch_size,
                                 args.lr,
                                 args.decay_rate,
                                 args.step_size,
                                 args.ratio,
                                 args.algorithm,
                                 use_cuda=args.use_cuda,
                                 master_listen_port_base=53622 #53300+offset
                                 )

    with open("worker_config.json") as json_file:
        workers_config = json.load(json_file)

    # global_model = models.create_model_instance(common_config.dataset, common_config.model)
    global_model=models.get_model(common_config.model)
    init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
    para_nums = torch.nn.utils.parameters_to_vector(global_model.parameters()).nelement()
    model_size = init_para.nelement() * 4 / 1024 / 1024
    print("Model name: {}".format(common_config.model))
    print("Model Size: {} MB".format(model_size))

    worker_num=min(args.worker_num,len(workers_config['worker_config_list']))

    worker_list = []
    for i in range(worker_num):
        worker_config=workers_config['worker_config_list'][i]
        custom = dict()
        custom["computation"] = worker_config["computation"]
        custom["dynamics"] = worker_config["dynamics"]
        worker_list.append(
            Worker(config=ClientConfig(idx=i,
                                    client_host=worker_config["host"],
                                    client_ip=worker_config["ip"],
                                    client_port=worker_config["port"],
                                    client_user=worker_config["user"],
                                    client_pwd=worker_config['pwd'],
                                    master_ip=server_ip,
                                    master_port=common_config.master_listen_port_base + i,
                                    custom=custom),
                   common_config=common_config,
                   user_name=worker_config['name'],
                   para_nums=para_nums
                   )
        )

    train_data_partition, test_data_partition = partition_data(common_config.dataset, args.data_pattern, worker_num)

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
    # local_steps_list=[50,40,50,30,50,40,30,50,40,30]
    # compre_ratio_list=[0.8,0.7,0.8,0.6,0.8,0.7,0.6,0.8,0.7,0.6]
    local_steps_list = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
    compre_ratio_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    computation_resource = [9, 3, 5, 1, 4, 6, 4, 1, 4, 7]
    total_resource = 0.0
    bandwith_resource = [8, 7, 7, 6, 9, 5, 3, 3, 5, 4]
    total_bandwith = 0.0
    # computation_resource,bandwith_resource=random_RC(10)

    # local_steps,compre_ratio=40,0.5
    for epoch_idx in range(1, 1 + common_config.epoch):

        print("get begin")
        try:
            communication_parallel(worker_list, action="get_model")
            #  communication_parallel(worker_list, action="get_time")
        except Exception as e:
            for worker in worker_list:
                worker.socket.shutdown(2)
            return
        else:
            print("get end")

        global_para = torch.nn.utils.parameters_to_vector(global_model.parameters()).clone().detach()
        global_para = aggregate_model(global_para, worker_list, args.step_size)

        # with open('data/para_epoch' + str(epoch_idx), 'w') as f:
        #     for para in global_para.data:
        #         f.write(str(float(para)))
        #         f.write('\n')

        print("send begin")
        try:
            communication_parallel(worker_list, action="send_model", data=global_para)
        except Exception as e:
            for worker in worker_list:
                worker.socket.shutdown(2)
            return
        else:
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
            # elif action == "get_time":
            #    tasks.append(loop.run_in_executor(executor, get_time, worker.config, worker.socket))
            elif action == "send_model":
                tasks.append(loop.run_in_executor(executor, worker.send_data, data))
            # elif action == "send_para":
            #    data = (worker.config.local_steps, worker.config.compre_ratio)
            #    tasks.append(loop.run_in_executor(executor, worker.send_data, data))
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
    except:
        sys.exit(0)


def get_time(config, socket):
    train_time, send_time = get_data_socket(socket)
    config.train_time = train_time
    config.send_time = send_time
    print(config.idx, " train time: ", train_time, " send time: ", send_time)


def get_compressed_model_top(config, socket, nelement):
    # start_time = time.time()
    print('Get compressed model')
    try:
        received_para = get_data_socket(socket)
    except Exception as e:
        print("FAILED: get model error.")
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
