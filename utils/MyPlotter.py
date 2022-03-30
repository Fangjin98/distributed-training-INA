from matplotlib import pyplot as plt


class BasicPlotter:
    def __init__(self, algs, x_label, xlim, x_infos,
                 bar_pos=None, bar_color=None, bar_hatch=None,
                 marker=None, linestyle=None, color=None,
                 label_font=None, legend_font=None,
                 y_label=None, y_infos=None):

        self.algs = algs

        self.label_font = label_font
        self.legend_font = legend_font
        self.legend = None

        self.bar_pos = bar_pos
        self.bar_color = bar_color
        self.bar_hatch = bar_hatch

        self.marker = marker
        self.linestyle = linestyle
        self.color = color

        self.x_label = x_label
        self.xlim = xlim
        self.x_infos = x_infos

        self.y_label = y_label
        self.y_infos = y_infos

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

        self.legend = ax.legend(frameon=False, prop=self.legend_font,
                                labelspacing=0.1, borderpad=0.1, loc=2)
        if x_label:
            self.x_label = x_label

        if x_infos:
            self.x_infos = x_infos

        if xlim:
            self.xlim = xlim

        if y_label:
            self.y_label = y_label

        if y_infos:
            self.y_infos = y_infos

        self._plot(plt, ax, save_file, file_name)

    def plot_linegraph(self, y_values, y_infos, y_label,
                       x_edges, x_label=None, x_infos=None,
                       fig_size=(8, 5),
                       save_file=False, file_name=None):
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

        for alg in self.algs:
            ax.plot(x_edges, y_values[alg], self.color[alg], label=alg,
                    marker=self.marker[alg], markersize=20, markerfacecolor='none',
                    linestyle=self.linestyle[alg], linewidth=3)

        self.legend = ax.legend(frameon=False, prop=self.legend_font,
                                labelspacing=0.1, borderpad=0.1)

        if x_label:
            self.x_label = x_label

        if x_infos:
            self.x_infos = x_infos

        self.xlim = (self.x_infos[0][0], self.x_infos[0][-1])

        if y_label:
            self.y_label = y_label

        if y_infos:
            self.y_infos = y_infos

        self._plot(plt, ax, save_file, file_name)

    def plot_diagram_with_error(self,
                                y_max_values, y_mean_values, y_min_values, y_infos, y_label,
                                bar_width=0.25,
                                fig_size=(8, 5),
                                save_file=False, file_name=None,
                                x_pos=None, x_label=None, xlim=None, x_infos=None

                                ):
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

        bar_pos = self.bar_pos
        if x_pos is not None:
            bar_pos = x_pos

        for i, alg in enumerate(self.algs):
            bars = ax.bar(x=bar_pos[i], height=y_mean_values[alg], width=bar_width,
                          color=self.bar_color[alg], edgecolor="k",
                          yerr=[
                              [y_max - y_mean for y_max, y_mean in zip(y_max_values[alg], y_mean_values[alg])],
                              [y_mean - y_min for y_mean, y_min in zip(y_mean_values[alg], y_min_values[alg])]
                          ],
                          error_kw={
                              'elinewidth': 3,
                              'capsize': 6,
                            'capthick': 3},
                          align='edge',
                          label=alg)
            for bar in bars:
                bar.set_hatch(self.bar_hatch[alg])

        self.legend = ax.legend(labelspacing=0.1, borderpad=0.1, frameon=False, prop=self.legend_font, ncol=2, loc=2)

        if x_label:
            self.x_label = x_label

        if x_infos:
            self.x_infos = x_infos

        if xlim:
            self.xlim = xlim

        if y_label:
            self.y_label = y_label

        if y_infos:
            self.y_infos = y_infos

        self._plot(plt, ax, save_file, file_name)

    def _plot(self, plt, ax, save_file, file_name):
        legend = self.legend

        frame = legend.get_frame()
        frame.set_alpha(1)
        frame.set_facecolor('none')

        ax.tick_params(labelsize=30)

        ax.set_xlabel(self.x_label, self.label_font)
        ax.set_xlim(self.xlim)
        ax.set_xticks(self.x_infos[0], self.x_infos[1])

        ax.set_ylabel(self.y_label, self.label_font)
        ax.set_yticks(self.y_infos[0], self.y_infos[1])
        ax.set_ylim(self.y_infos[0][0], self.y_infos[0][-1])

        if save_file:
            try:
                plt.savefig(file_name, format='pdf')
            except Exception as e:
                print(e)
                plt.savefig('unnamed_figure.pdf', format='pdf')
        else:
            plt.show()


class AccPlotter(BasicPlotter):
    def __init__(self, algs, x_label, xlim, x_infos,
                 bar_pos=None, bar_color=None, bar_hatch=None,
                 marker=None, linestyle=None, color=None,
                 label_font=None, legend_font=None,
                 y_label=None, y_infos=None
                 ):
        super().__init__(algs, x_label, xlim, x_infos,
                         bar_pos, bar_color, bar_hatch,
                         marker, linestyle, color,
                         label_font, legend_font,
                         y_label, y_infos
                         )

    def plot_linegraph(self, y_values, y_infos, y_label,
                       x_edges, x_label=None, x_infos=None,
                       fig_size=(8, 5),
                       save_file=False, file_name=None):
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

        for alg in self.algs:
            ax.plot(x_edges[alg], y_values[alg], self.color[alg], label=alg,
                    marker=self.marker[alg], markersize=10, markerfacecolor='none',
                    linestyle=self.linestyle[alg], linewidth=3)

        self.legend = ax.legend(frameon=False, prop=self.legend_font,
                                labelspacing=0.1, borderpad=0.1)

        if x_label:
            self.x_label = x_label

        if x_infos:
            self.x_infos = x_infos

        self.xlim = (self.x_infos[0][0], self.x_infos[0][-1])

        if y_label:
            self.y_label = y_label

        if y_infos:
            self.y_infos = y_infos

        self._plot(plt, ax, save_file, file_name)
