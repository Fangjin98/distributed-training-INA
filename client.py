import argparse

import torch
import torch.optim as optim
# from pulp import *
from torch.utils.tensorboard import SummaryWriter
from threading import Thread

from config import ClientConfig
from utils import models, datasets
from utils.DataManager import DataManager
from utils.comm_utils import *
from utils.file_utils import write_tensor_to_file
from utils.training_utils import train, test

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--idx', type=str, default="0",
                    help='index of worker')
parser.add_argument('--master_ip', type=str, default="127.0.0.1",
                    help='IP address for controller or ps')
parser.add_argument('--master_port', type=int, default=58000, metavar='N',
                    help='')
parser.add_argument('--master_nic_ip',type=str, default="127.0.0.1")
parser.add_argument('--client_ip', type=str, default='127.0.0.1')
parser.add_argument('--client_nic_ip', type=str, default='127.0.0.1')
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--model', type=str, default='LR')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--min_lr', type=float, default=0.001)
parser.add_argument('--ratio', type=float, default=0.2)
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--decay_rate', type=float, default=0.97)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--local_iters', type=int, default=-1)
parser.add_argument('--use_cuda', action="store_false", default=True)
parser.add_argument('--adaptive', action="store_false", default=False)
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--algorithm', type=str, default='proposed')
parser.add_argument('--write_to_file', type=bool, default=False)

args = parser.parse_args()

if args.visible_cuda == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = str((int(args.idx)) % 2 + 2)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_cuda
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")


def write_tensor(filename, tensor):
    t = Thread(target=write_tensor_to_file, args=(filename, tensor))
    t.start()


def main():
    client_config = ClientConfig()
    recorder = SummaryWriter("log_" + str(client_config.idx))
    # receive config
    print(str(args.client_ip), str(args.master_port))
    master_socket = connect_get_socket(args.client_ip, args.master_port)
    config_received = get_data_socket(master_socket)

    print(config_received)

    for k, v in config_received.__dict__.items():
        setattr(client_config, k, v)
    computation = client_config.custom["computation"]
    dynamics = client_config.custom["dynamics"]

    # init config
    args.local_ip = client_config.client_ip

    for arg in vars(args):
        print(arg, ":", getattr(args, arg))

    print('Create local model.')

    local_model = models.get_model(args.model)
    torch.nn.utils.vector_to_parameters(client_config.para, local_model.parameters())
    local_model.to(device)
    para_nums = torch.nn.utils.parameters_to_vector(local_model.parameters()).nelement()

    print(args.batch_size, args.ratio)
    # create dataset
    print(len(client_config.custom["train_data_idxes"]))
    train_dataset, test_dataset = datasets.load_datasets(args.dataset)
    train_loader = datasets.create_dataloaders(train_dataset, batch_size=args.batch_size,
                                               selected_idxs=client_config.custom["train_data_idxes"])

    test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, shuffle=False)

    local_model.to(device)
    epoch_lr = args.lr
    local_steps, compre_ratio = 50, 1

    data_manager = DataManager(src_ip=args.client_nic_ip,
                               dst_ip=args.master_nic_ip,
                               interface='eno5')

    for epoch in range(1, 1 + args.epoch):
        if epoch > 1 and epoch % 1 == 0:
            epoch_lr = max((args.decay_rate * epoch_lr, args.min_lr))
        print("model-{}-epoch-{} lr: {}, ratio: {} ".
              format(args.model, epoch, epoch_lr, args.ratio))
        start_time = time.time()
        optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
        train_loss = train(local_model, train_loader, optimizer, local_iters=local_steps, device=device,
                           model_type=args.model)
        local_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).clone().detach()

        if args.write_to_file:
            write_tensor("data/log/tensor/model_{}_epoch_{}_worker_{}".
                         format(args.model, epoch, args.idx), local_para)

        train_time = time.time() - start_time
        train_time = train_time / local_steps
        print("train time: ", train_time)
        print(train_time / computation)
        test_loss, acc = test(local_model, test_loader, device, model_type=args.model)
        recorder.add_scalar('acc_worker-' + str(args.idx), acc, epoch)
        recorder.add_scalar('test_loss_worker-' + str(args.idx), test_loss, epoch)
        recorder.add_scalar('train_loss_worker-' + str(args.idx), train_loss, epoch)
        print("after aggregation, epoch: {}, train loss: {}, test loss: {}, test accuracy: {}".format(epoch, train_loss,
                                                                                                      test_loss, acc))
        print("send para")

        start_time = time.time()
        send_data_socket(local_para, master_socket)
        send_time = time.time() - start_time
        print("Socket send time: ", send_time)

        data_manager.update_data(local_para.detach().tolist())
        t1 = Thread(target=data_manager.send_data, args=(int(args.idx), 1, 2))
        t1.start()

        print("get begin")
        local_para = get_data_socket(master_socket)
        print("get end")
        local_para.to(device)
        torch.nn.utils.vector_to_parameters(local_para, local_model.parameters())

    master_socket.shutdown(2)
    master_socket.close()


if __name__ == '__main__':
    main()
