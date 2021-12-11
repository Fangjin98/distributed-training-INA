from matplotlib import pyplot as plt
import matplotlib
import os
import sys
import seaborn as sns
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, '../')
print(PROJECT_DIR)
sys.path.append(PROJECT_DIR)

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

fontLegend = {
    # 'family': 'Times New Roman',
    'weight': 'normal',
    'size': 20,
}
fontLabel = {
    # 'family': 'Times New Roman',
    'weight': 'normal',
    'size': 24,
}


def loss_timestamp():
    time_stamp = []
    loss = []
    with open(PROJECT_DIR + 'data/epoch_time_worker_8_model_resnet50', 'r') as f:
        time_0 = list(map(float, f.readline().strip().split(' ')))
        for i in range(1, len(time_0)):
            time_0[i] += time_0[i - 1]
    time_stamp.append(time_0)
    with open(PROJECT_DIR + 'data/timestamp_1', 'r') as f:
        time_1 = []
        for line in f:
            time_1.append(float(line))
    time_stamp.append(time_1)

    with open(PROJECT_DIR + 'data/client_0_log.txt', 'r') as f:
        loss_0 = []
        while True:
            f.readline()
            line = f.readline()
            f.readline()
            if not line:
                break
            loss_0.append(float(line.strip().split(' ')[-1]))
        loss.append(loss_0)
    with open(PROJECT_DIR + 'data/client_1_log.txt', 'r') as f:
        loss_1 = []
        while True:
            f.readline()
            line = f.readline()
            f.readline()
            if not line:
                break
            loss_1.append(float(line.strip().split(' ')[-1]))
        loss.append(loss_1)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(time_stamp[0], loss[0],
            marker='x', markersize=6,
            linestyle='-', linewidth=2,
            label='NCCL')
    ax.plot(time_stamp[1], loss[1],
            marker='o', markersize=4,
            linestyle='-', linewidth=2,
            label='NGAA')

    legend = ax.legend(frameon=True, prop=fontLegend)
    # frame = legend.get_frame()
    # frame.set_alpha(1)
    # frame.set_facecolor('none')

    ax.tick_params(labelsize=24)
    ax.set_xlabel('Time (s)', fontLabel)
    ax.set_ylabel('Acc', fontLabel)
    ax.set_xlim(0, 8000)
    ax.set_ylim(0.5, 0.9)
    ax.set_xticks([i * 1000 for i in range(0, 9)], [str(i * 1000) for i in range(0, 9)])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9], ['0.5', '0.6', '0.7', '0.8', '0.9'])
    # plt.show()
    plt.savefig(PROJECT_DIR + 'figures/lossvstime.pdf', format='pdf')


def control_vs_dataplane():
    schedule_time = [0.007996559143066406, 0.00700068473815918,
                     0.00796961784362793, 0.009997367858886719]
    deploy_time = []
    epoch_time = []
    for index, num in enumerate([2, 4, 6, 8]):
        with open(PROJECT_DIR + 'data/epoch_time_worker_{}'.format(str(num)), 'r') as f:
            epoch_time.append(list(map(float, f.readline().strip().split(' '))))
        with open(PROJECT_DIR + 'data/transfer_time_worker_{}'.format(str(num)), 'r') as f:
            tmp = list(map(float, f.readline().strip().split(' ')))
            deploy_time.append([tmp[i] + schedule_time[index] for i in range(10)])
    my_index = [
        [1.75, 3.75, 5.75, 7.75],
        [2.25, 4.25, 6.25, 8.25]
    ]
    eva_result = pd.DataFrame(deploy_time + epoch_time)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bar1 = ax.bar(my_index[0], width=0.5, edgecolor="k", height=eva_result.mean(axis=1)[:4],
                  label="Update Time")
    for bar in bar1:
        bar.set_hatch('oo')
    bar2 = ax.bar(my_index[1], width=0.5, edgecolor="k", height=eva_result.mean(axis=1)[4:],
                  label="Train Time")
    for bar in bar2:
        bar.set_hatch('xx')
    legend = ax.legend(frameon=False, prop=fontLegend, loc=2)
    frame = legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')
    ax.tick_params(labelsize=24)
    ax.set_xlabel('No. of Workers', fontLabel)
    ax.set_ylabel('Time (s)', fontLabel)
    ax.set_xlim([1, 9])
    ax.set_xticks([2, 4, 6, 8], ['2', '4', '6', '8'])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100], ['0', '20', '40', '60', '80', '100'])
    # plt.show()
    plt.savefig(PROJECT_DIR + 'figures/deployvstrain.pdf', format='pdf')


if __name__ == '__main__':
    # control_vs_dataplane()
    loss_timestamp()
