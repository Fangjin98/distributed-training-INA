from matplotlib import pyplot as plt
import matplotlib
import os
import sys
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, '../')
print(PROJECT_DIR)
sys.path.append(PROJECT_DIR)

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'Times New Roman'

fontLegend = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 20,
}
fontLabel = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 24,
}

ours='RRIAR'
benchmark1='ATP'
benchmark2='SwitchMl'

throughput_resnet50={
    ours: [1920,2304,2592,2976],
    benchmark1: [1920,2016,2112,2208],
    benchmark2: [1920,2304,2688,3072]
}

def worker_num_vs_throughput(throughput, worker_num=['20','24','28','32']):

    x_index = [
        [1.75, 3, 4.25, 5.5],
        [2.25, 3.5, 4.75, 6],
        [2.75,4,5.25,6.5]
    ]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bar1 = ax.bar(x_index[0], width=0.5, edgecolor="k", height=throughput[ours],
                  label=ours)
    for bar in bar1:
        bar.set_hatch('oo')
    bar2 = ax.bar(x_index[1], width=0.5, edgecolor="k", height=throughput[benchmark1],
                  label=benchmark1)
    for bar in bar2:
        bar.set_hatch('xx')
    bar3 = ax.bar(x_index[2], width=0.5, edgecolor="k", height=throughput[benchmark2],
                  label=benchmark2)
    for bar in bar3:
        bar.set_hatch('\\')
    
    legend = ax.legend(frameon=False, prop=fontLegend, loc=2)
    frame = legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')
    
    ax.tick_params(labelsize=24)
    ax.set_xlabel('No. of Workers', fontLabel)
    ax.set_ylabel('Throughput (Mbps)', fontLabel)
    ax.set_xlim([1.25, 6.75])
    ax.set_xticks([2, 3.25, 4.5, 5.75], worker_num)
    ax.set_ylim(0, 100)
    ax.set_yticks([1800, 2400, 2800, 3200], ['0', '20', '40', '60'])
    # plt.show()
    plt.savefig(PROJECT_DIR + 'figures/deployvstrain.pdf', format='pdf')


if __name__=="__main__":
    worker_num_vs_throughput(throughput=throughput_resnet50)