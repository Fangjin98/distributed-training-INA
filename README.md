# Distributed ML Training with In-Network Aggregation

A distributed PS training architecture with P4 programmable switches accelerating.

## Dependency

pytorch needed

  ```bash
  sudo apt install libjpeg-dev zlib1g-dev libssl-dev libffi-dev python-dev build-essential libxml2-dev libxslt1-dev
  ```

python dependency  

```bash
  pip3 install pulp numpy tensorboard
  ```

cpu only pytorch

```bash
pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

## Usage

We ignore the config files for security. You need to create `config\workers.json` for distributed training.

```json
[
    {
        "host_ip": "id of worker 1",
        "ssh_port": "port for ssh",
        "ssh_usr" : "user account to ssh",
        "ssh_psw" : "password",
        "work_dir": "path of files"
    },
    {
        "host_ip": "id of worker 2",
        "ssh_port": "port for ssh",
        "ssh_usr" : "user account to ssh",
        "ssh_psw" : "password",
        "work_dir": "path of files"
    },
]
```

Run `./deploy.sh` to sync codes among all the machines: make sure you have created the `<repo>` directory.

```bash
# deploy.sh

scp -r current_path ssh_usr@machine_ip:dest_path
```

Run `./test.sh $WORKER_NUM` to start training. The scripts will run `python3 launch.py --master True xxx` to launch the PS, which will launch workers via ssh according to the IP list in `config/workers.json`

```bash
# test.sh

WORKER_NUM=$1

sudo python3 src/launch.py --master 1 --ip machine_ip --worker_num $WORKER_NUM --config_file config/workers.json --dataset CIFAR100 --model resnet50
```
