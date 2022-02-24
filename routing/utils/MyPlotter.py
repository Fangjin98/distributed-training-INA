from matplotlib import pyplot as plt

class BasicPlotter:
    def __init__(self, algs, x_label, xlim, x_infos,
                 bar_pos=None, bar_color=None, bar_hatch=None,
                 marker=None, linestyle=None, color=None,
                 label_font=None, legend_font=None):
        self.algs = algs

        self.label_font = label_font
        self.legend_font = legend_font

        self.bar_pos = bar_pos
        self.bar_color = bar_color
        self.bar_hatch = bar_hatch

        self.marker = marker
        self.linestyle = linestyle
        self.color = color

        self.x_label = x_label
        self.xlim = xlim
        self.x_infos = x_infos

    def plot_diagram(self,
                     y_values, y_infos, y_label,
                     bar_width=0.25,
                     fig_size=(8, 5),
                     save_file=False, file_name=None,
                     x_pos=None, x_label=None, xlim=None, x_infos=None):

        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

        bar_pos = self.bar_pos
        if x_pos is not None:
            bar_pos = x_pos

        for i, alg in enumerate(self.algs):
            bars = ax.bar(bar_pos[i], width=bar_width,
                          color=self.bar_color[alg], edgecolor="k",
                          height=y_values[alg], label=alg)
            for bar in bars:
                bar.set_hatch(self.bar_hatch[alg])

        legend = ax.legend(frameon=False, prop=self.legend_font,
                           loc=2, labelspacing=0.1, borderpad=0.1)
        frame = legend.get_frame()
        frame.set_alpha(1)
        frame.set_facecolor('none')

        ax.tick_params(labelsize=30)

        if x_label is None:
            ax.set_xlabel(self.x_label, self.label_font)
        else:
            ax.set_xlabel(x_label, self.label_font)

        if xlim is None:
            ax.set_xlim(self.xlim)
        else:
            ax.set_xlim(xlim)

        if x_infos is None:
            ax.set_xticks(self.x_infos[0], self.x_infos[1])
        else:
            ax.set_xticks(x_infos[0], x_infos[1])

        ax.set_ylabel(y_label, self.label_font)
        ax.set_yticks(y_infos[0], y_infos[1])
        ax.set_ylim(y_infos[0][0], y_infos[0][-1])

        if save_file:
            try:
                plt.savefig(file_name, format='pdf')
            except Exception as e:
                print(e)
                plt.savefig('unnamed_figure.pdf', format='pdf')
        else:
            plt.show()

    def plot_linegraph(self, y_values,y_infos,y_label,
                       x_edges, x_label=None, x_infos=None,
                       fig_size=(8, 5),
                       save_file=False, file_name=None):
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

        for alg in self.algs:
            ax.plot(x_edges, y_values[alg], self.color[alg], label=alg,
                    marker=self.marker[alg], markersize=20, markerfacecolor='none',
                    linestyle=self.linestyle[alg], linewidth=3)

        legend = ax.legend(frameon=False, prop=self.legend_font,
                           labelspacing=0.1, borderpad=0.1)
        frame = legend.get_frame()
        frame.set_alpha(1)
        frame.set_facecolor('none')

        ax.tick_params(labelsize=30)

        if x_label is None:
            ax.set_xlabel(self.x_label, self.label_font)
        else:
            ax.set_xlabel(x_label, self.label_font)

        if x_infos is None:
            ax.set_xlim((self.x_infos[0][0], self.x_infos[0][-1]))
            ax.set_xticks(self.x_infos[0], self.x_infos[1])
        else:
            ax.set_xlim((x_infos[0][0], x_infos[0][-1]))
            ax.set_xticks(x_infos[0], x_infos[1])

        ax.set_ylabel(y_label, self.label_font)
        ax.set_yticks(y_infos[0], y_infos[1])
        ax.set_ylim(y_infos[0][0],y_infos[0][-1])

        if save_file:
            try:
                plt.savefig(file_name, format='pdf')
            except Exception as e:
                print(e)
                plt.savefig('unnamed_figure.pdf', format='pdf')
        else:
            plt.show()
