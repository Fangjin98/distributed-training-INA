from typing import List
from torch.utils.tensorboard import SummaryWriter
from utils.comm_utils import *


class Worker:
    def __init__(self,
                 config,
                 common_config,
                 user_name,
                 para_nums
                 ):
        self.config = config
        self.common_config = common_config
        self.user_name = user_name
        self.idx = config.idx
        self.worker_time = 0.0
        self.socket = None
        self.train_info = None
        self.para_nums = para_nums
        self.socket = None
        self.__start_local_worker_process()

    def __start_local_worker_process(self):
        cmd = 'cd ' + os.getcwd() + ';nohup ' + 'python3' + ' -u client.py --master_ip ' \
              + self.config.master_ip + ' --master_port ' + \
              str(self.config.master_port) + ' --idx ' + str(self.idx) \
              + ' --dataset ' + str(self.common_config.dataset) + ' --model ' + str(
            self.common_config.model) \
              + ' --epoch ' + str(self.common_config.epoch) + ' --batch_size ' + str(self.common_config.batch_size) \
              + ' --ratio ' + str(self.common_config.ratio) + ' --lr ' + str(self.common_config.lr) + \
              ' --decay_rate ' + str(self.common_config.decay_rate) \
              + ' --algorithm ' + self.common_config.algorithm + ' --step_size ' + str(self.common_config.step_size) \
              + ' > data/log/client_' + str(self.idx) + '_log.txt 2>&1 &'

        print(cmd)
        os.system(cmd)

        print("start process at ", self.user_name)

    def send_data(self, data):
        send_data_socket(data, self.socket)

    def send_init_config(self):
        self.socket = connect_send_socket(self.config.master_ip, self.config.master_port)
        send_data_socket(self.config, self.socket)

    def get_config(self):
        self.train_info = get_data_socket(self.socket)


class CommonConfig:
    def __init__(self,
                 dataset,
                 model,
                 epoch,
                 batch_size,
                 lr,
                 decay_rate,
                 step_size,
                 ratio,
                 algorithm,
                 epoch_start=0,
                 train_mode='local',
                 use_cuda=True,
                 master_listen_port_base=53300,
                 summary_writer=SummaryWriter()
                 ):
        self.dataset = dataset
        self.model = model
        self.use_cuda = use_cuda
        self.train_mode = train_mode

        self.epoch_start = epoch_start
        self.epoch = epoch

        self.batch_size = batch_size
        self.test_batch_size = self.batch_size

        self.lr = lr
        self.decay_rate = decay_rate
        self.step_size = step_size
        self.ratio = ratio
        self.algorithm = algorithm

        self.master_listen_port_base = master_listen_port_base
        self.recoder = summary_writer


class ClientConfig:
    def __init__(self,
                 idx: int = 0,
                 client_ip: str =None,
                 master_ip: str =None,
                 master_port: int=0,
                 custom: dict = dict()
                 ):
        self.idx = idx
        self.client_ip = client_ip
        self.master_ip = master_ip
        self.master_port = master_port
        self.neighbor_paras = None
        self.neighbor_indices = None
        self.train_time = 0
        self.send_time = 0
        self.local_steps = 50
        self.compre_ratio = 1
        self.average_weight = 0
        self.custom = custom
        self.epoch_num: int = 1
        self.model = list()
        self.para = dict()
        self.resource = {"CPU": "1"}
        self.acc: float = 0
        self.loss: float = 1
        self.running_time: int = 0
