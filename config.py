from threading import Thread

from utils.comm_utils import *

script_path_of_host = {
    "edge401": '/home/jfang/.conda/envs/fj/bin/python3',
    "edge404": '/data/yxu/software/Anaconda/envs/fj/bin/python3',
    "server05": '/usr/bin/python3',
    "server01": '/home/sdn/anaconda3/envs/fj/bin/python3',
    "server02": '/home/sdn/anaconda3/envs/fj/bin/python3',
    "server03": '/usr/bin/python3',
    "server08": '/home/sdn/anaconda3/envs/fj/bin/python3'
}

work_dir_of_host = {
    "edge401": '/home/jfang/distributed_PS_ML',
    "edge404": '/home/jfang/distributed_PS_ML',
    "server05": '/home/sdn/fj/distributed_PS_ML',
    "server01": '/home/sdn/fj/distributed_PS_ML',
    "server02": '/home/sdn/fj/distributed_PS_ML',
    "server03": '/home/sdn/fj/distributed_PS_ML',
    "server08": '/home/sdn/fj/distributed_PS_ML'
}


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

        cmd = ' cd ' + work_dir_of_host[self.config.client_host] + '; sudo ' + script_path_of_host[
            self.config.client_host] + ' -u client.py ' + \
              ' --master_ip ' + str(self.config.master_ip) + \
              ' --master_port ' + str(self.config.master_port) + \
              ' --master_nic_ip ' + str(self.config.master_nic_ip) + \
              ' --client_ip ' + str(self.config.client_ip) + \
              ' --client_nic_ip ' + str(self.config.client_nic_ip) + \
              ' --idx ' + str(self.idx) + \
              ' --dataset ' + str(self.common_config.dataset) + \
              ' --model ' + str(self.common_config.model) + \
              ' --epoch ' + str(self.common_config.epoch) + \
              ' --batch_size ' + str(self.common_config.batch_size) + \
              ' --ratio ' + str(self.common_config.ratio) + \
              ' --lr ' + str(self.common_config.lr) + \
              ' --decay_rate ' + str(self.common_config.decay_rate) + \
              ' --algorithm ' + self.common_config.algorithm + \
              ' --step_size ' + str(self.common_config.step_size) + \
              ' --write_to_file ' + str(self.common_config.write_to_file) + \
              ' > data/log/client_' + str(self.idx) + '_model_' + str(self.config.model) + '_log.txt 2>&1'

        if self.config.client_ip == '127.0.0.1':
            t = Thread(target=self.__start_local_worker_process, args=(cmd,))
            t.start()
        else:
            t = Thread(target=self.__start_remote_worker_process, args=(cmd,))
            t.start()

    def __start_local_worker_process(self, cmd):
        print(cmd)
        os.system(cmd)

    def __start_remote_worker_process(self, cmd):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(hostname=self.config.client_ip, port=int(self.config.ssh_port),
                        username=self.config.client_user, password=self.config.client_pwd)
        except Exception as e:
            print("SSH FAILED: {}".format(self.config.client_ip))
            print(e)
            ssh.close()
        else:
            cmd = ' cd ' + work_dir_of_host[self.config.client_host] + '; sudo ' + script_path_of_host[
                self.config.client_host] + ' -u client.py ' + \
                  ' --master_ip ' + str(self.config.master_ip) + \
                  ' --master_port ' + str(self.config.master_port) + \
                  ' --master_nic_ip ' + str(self.config.master_nic_ip) + \
                  ' --client_ip ' + str(self.config.client_ip) + \
                  ' --client_nic_ip ' + str(self.config.client_nic_ip) + \
                  ' --idx ' + str(self.idx) + \
                  ' --dataset ' + str(self.common_config.dataset) + \
                  ' --model ' + str(self.common_config.model) + \
                  ' --epoch ' + str(self.common_config.epoch) + \
                  ' --batch_size ' + str(self.common_config.batch_size) + \
                  ' --ratio ' + str(self.common_config.ratio) + \
                  ' --lr ' + str(self.common_config.lr) + \
                  ' --decay_rate ' + str(self.common_config.decay_rate) + \
                  ' --algorithm ' + self.common_config.algorithm + \
                  ' --step_size ' + str(self.common_config.step_size) + \
                  ' --write_to_file ' + str(self.common_config.write_to_file) + \
                  ' > data/log/client_' + str(self.idx) + '_log.txt 2>&1'
            print("Execute cmd.")
            print(cmd)
            stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
            stdin.write(str(self.config.client_pwd) + '\n')
            output = []
            out = stdout.read()
            error = stderr.read()
            if out:
                print('[%s] OUT:\n%s' % (self.config.client_ip, out.decode('utf8')))
                output.append(out.decode('utf-8'))
            if error:
                print('ERROR:[%s]\n%s' % (self.config.client_ip, error.decode('utf8')))
                output.append(str(self.config.client_ip) + ' ' + error.decode('utf-8'))
            print(output)

    def send_data(self, data):
        send_data_socket(data, self.socket)

    def send_init_config(self):
        self.socket = connect_send_socket(self.config.client_ip, int(self.config.master_port))
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
                 write_to_file=False,
                 epoch_start=0,
                 train_mode='local',
                 use_cuda=True,
                 master_listen_port_base=53300,
                 project_dir="distributed_PS_ML"
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
        self.project_dir = project_dir
        self.write_to_file = write_to_file


class ClientConfig:
    def __init__(self,
                 idx: int = 0,
                 client_host: str = None,
                 client_ip: str = '127.0.0.1',
                 client_nic_ip: str = '127.0.0.1',
                 ssh_port: str = None,
                 client_user: str = None,
                 client_pwd: str = None,
                 master_ip: str = '127.0.0.1',
                 master_port: int = 0,
                 master_nic_ip: str = '127.0.0.1',
                 custom: dict = dict()
                 ):
        self.idx = idx
        self.client_host = client_host
        self.client_ip = client_ip
        self.client_nic_ip = client_nic_ip
        self.ssh_port = ssh_port
        self.client_user = client_user
        self.client_pwd = client_pwd
        self.master_ip = master_ip
        self.master_port = master_port
        self.master_nic_ip = master_nic_ip
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
