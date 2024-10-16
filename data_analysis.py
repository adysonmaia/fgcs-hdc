import math
import os
from glob import glob
from itertools import cycle, islice

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from future.utils import iteritems


def s_to_ms(value):
    return 1000.0 * value


def to_percent(value):
    return 100.0 * value


def bool_to_int(value):
    return int(value)


def plot_metrics(optimizers, experiments, metrics, data_path, x_label):
    cloud_opt_id = "CloudOptimizer"

    metrics_data = []
    for exp in experiments:
        exp_path = os.path.join(data_path, exp["path"], "[0-9]*")
        runs_path = glob(exp_path)
        for opt in optimizers:
            for run, run_path in enumerate(runs_path):
                filename = os.path.join(run_path, opt["id"], "metrics.json")
                if not os.path.isfile(filename):
                    continue
                df = pd.read_json(filename, orient="records")

                cloud_filename = os.path.join(run_path, cloud_opt_id, "metrics.json")
                cloud_df = None
                if os.path.isfile(cloud_filename):
                    cloud_df = pd.read_json(cloud_filename, orient="records")

                data = {"opt": opt["label"], "run": run, "x": exp["x"]}
                for metric in metrics:
                    metric_id = metric["id"]
                    if metric_id not in df.columns:
                        continue

                    value = df[metric_id].mean()
                    cloud_value = (
                        cloud_df[metric_id].mean() if cloud_df is not None else 0.0
                    )
                    if "func" in metric:
                        value = metric["func"](value)
                        cloud_value = metric["func"](cloud_value)
                    if (
                        "normalize" in metric
                        and metric["normalize"]
                        and cloud_value > 0.0
                    ):
                        value = value / float(cloud_value)
                    data[metric_id] = value
                metrics_data.append(data)

    metrics_df = pd.DataFrame.from_records(metrics_data)

    sns.set()
    # sns.set_context('paper', font_scale=2.0)
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks", {"axes.grid": True, "grid.linestyle": "--"})
    mpl.rcParams["hatch.linewidth"] = 0.5
    mpl.rcParams["figure.figsize"] = (7, 3)
    # plt.figure().set_figheight(4)

    # metric_id = 'weighted_avg_deadline_violation'
    # metric_id = 'overall_cost'
    # # metric_id = 'weighted_migration_rate'
    # print(metrics_df.groupby(['x', 'opt'])[metric_id].mean())
    # exit()

    # # sns.relplot(x='x', y=metric_id, hue='opt', kind='line', data=metrics_df)
    # # sns.catplot(x='x', y=metric_id, hue='opt', kind='point', ci=None, data=metrics_df, facet_kws=dict(despine=False))
    # # sns.catplot(x='x', y=metric_id, hue='opt', kind='point', col='run', col_wrap=10, ci=None, data=metrics_df)
    # sns.catplot(x='x', y=metric_id, hue='opt', kind='bar', ci=95, data=metrics_df)
    # # sns.boxplot(x='x', y=metric_id, hue='opt', notch=False, data=metrics_df)
    # # print(sns.axes_style())
    # plt.show()

    markers_unique = ["o", "s", "d", "^", "v", "<"]
    opt_count = metrics_df["opt"].nunique()
    x_count = metrics_df["x"].nunique()
    markers = list(islice(cycle(markers_unique), opt_count))

    hatches_unique = ["/", "\\", "-", "x", "+", "*", "o"]
    hatches = list(islice(cycle(hatches_unique), opt_count))

    for metric in metrics:
        metric_id = metric["id"]
        # ax = sns.pointplot(x='x', y=metric_id, hue='opt', kind='point', ci=None, data=metrics_df,
        #                    height=5, aspect=1.5, markers=markers)
        # ax = sns.barplot(x="x", y=metric_id, hue="opt", ci=None, data=metrics_df)
        ax = sns.barplot(x="x", y=metric_id, hue="opt", ci=95, data=metrics_df)
        ax.xaxis.grid(True)
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric["label"])

        if "y_limit" in metric:
            ax.set_ylim(*metric["y_limit"])

        hatch_index = -1
        for index, patch in enumerate(ax.patches):
            # Loop iterates an opt throughout x-axes, then it advances to the next opt
            if index % x_count == 0:
                hatch_index += 1
            hatch = hatches[hatch_index]
            patch.set_hatch(hatch)

            if patch.get_height() == 0:
                ax.annotate(
                    format(patch.get_height(), ".1f"),
                    # (patch.get_x() + patch.get_width() / 2., patch.get_height()),
                    (patch.get_x() + 0.03, patch.get_height()),
                    color=patch.get_facecolor(),
                    ha="center",
                    va="center",
                    size=15,
                    xytext=(0, 9),
                    textcoords="offset points",
                )

        # ax.legend(loc='upper center', bbox_to_anchor=(0.44, -0.15),
        #           ncol=3, title=None, frameon=False, fontsize=15,
        #           handlelength=1.6, columnspacing=0.5, labelspacing=0.0, handletextpad=0.0)
        # plt.subplots_adjust(bottom=0.29, top=0.992, left=0.14, right=0.995)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.97, 1.0),
            ncol=1,
            title=None,
            frameon=False,
            fontsize=15,
            handlelength=1.0,
            columnspacing=0.5,
            labelspacing=0.0,
            handletextpad=0.0,
        )
        plt.subplots_adjust(bottom=0.2, top=0.992, left=0.11, right=0.68)

        # if 'legend_pos' in metric:
        #     legend_pos = metric['legend_pos']
        #     ax.legend(loc='center', ncol=3, title=None,
        #               bbox_to_anchor=legend_pos,
        #               columnspacing=0.0, labelspacing=0.0, handletextpad=0.0)
        # else:
        #     ax.legend(loc='best', ncol=3, title=None,
        #               columnspacing=0.0, labelspacing=0.0, handletextpad=0.0)
        # plt.subplots_adjust(bottom=0.14, top=0.995, left=0.14, right=0.995)
        if "fig_file" in metric:
            # plt.savefig(metric['fig_file'], dpi=100, bbox_inches='tight')
            plt.savefig(metric["fig_file"], dpi=100)
            plt.clf()
        else:
            plt.show()


def plot_placement_per_node_type(
    optimizers, experiments, output_path, x_label, fig_path=None
):
    data = []
    for exp in experiments:
        exp_path = os.path.join(output_path, exp["path"], "[0-9]*")
        runs_path = glob(exp_path)
        for opt in optimizers:
            for run, run_path in enumerate(runs_path):
                filename = os.path.join(run_path, opt["id"], "placement.json")
                if not os.path.isfile(filename):
                    continue
                df = pd.read_json(filename, orient="records")
                df["place"] = df["place"].astype(int)

                nodes_id = df["node"].unique()
                last_node_id = df["node"].max()

                place_df = df.groupby(["node", "time"])["place"].sum().reset_index()
                place_ts = place_df.groupby(["node"])["place"].mean()
                # place_ts = df.groupby(['node'])['place'].mean()

                total_place = place_ts.sum()
                load_per_node_type = [
                    ("cloud", place_ts[last_node_id]),
                    ("core", place_ts[last_node_id - 1]),
                    ("bs", place_ts[place_ts.index < last_node_id - 1].sum()),
                    ("non_bs", place_ts[place_ts.index >= last_node_id - 1].sum()),
                    ("all", place_ts.sum()),
                ]
                for node_type, place in load_per_node_type:
                    place_percent = (
                        place / float(total_place) if total_place > 0.0 else 0.0
                    )
                    place_percent *= 100.0
                    datum = {
                        "x": exp["x"],
                        "opt": opt["label"],
                        "run": run,
                        "node": node_type,
                        "place": place,
                        "place_percent": place_percent,
                    }
                    data.append(datum)

    place_df = pd.DataFrame.from_records(data)

    opt_count = place_df["opt"].nunique()
    x_count = place_df["x"].nunique()

    hatches_unique = ["/", "\\", "-", "x", "+", "*", "o"]
    hatches = list(islice(cycle(hatches_unique), opt_count))

    node_types = [
        # {'id': 'cloud', 'title': 'Cloud'},
        # {'id': 'core', 'title': 'Core'},
        {"id": "non_bs", "title": "Cloud and Core"},
        {"id": "bs", "title": "Base Stations (RAN)"},
    ]

    sns.set()
    # sns.set_context('paper', font_scale=2.0)
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks", {"axes.grid": True, "grid.linestyle": "--"})
    mpl.rcParams["hatch.linewidth"] = 0.5
    mpl.rcParams["figure.figsize"] = (7, 3)
    # plt.figure().set_figheight(4)

    # y_label = 'Number of App. Replicas (%)'
    # y_label = 'Number of App. Replicas'
    y_label = "Number of Replicas"
    y_limit = 24
    for node_type in node_types:
        df = place_df[place_df["node"] == node_type["id"]]

        # ax = sns.barplot(x='x', y='place_percent', hue='opt', ci=None, data=df)
        # ax = sns.barplot(x="x", y="place", hue="opt", ci=None, data=df)
        ax = sns.barplot(x="x", y="place", hue="opt", ci=95, data=df)
        ax.xaxis.grid(True)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_ylim(0, y_limit)
        # ax.set_title(node_type['title'])

        hatch_index = -1
        for index, patch in enumerate(ax.patches):
            # Loop iterates an opt throughout x-axes, then it advances to the next opt
            if index % x_count == 0:
                hatch_index += 1
            hatch = hatches[hatch_index]
            patch.set_hatch(hatch)

            if patch.get_height() == 0:
                ax.annotate(
                    format(patch.get_height(), ".1f"),
                    # (patch.get_x() + patch.get_width() / 2., patch.get_height()),
                    (patch.get_x() + 0.03, patch.get_height()),
                    color=patch.get_facecolor(),
                    ha="center",
                    va="center",
                    size=15,
                    xytext=(0, 9),
                    textcoords="offset points",
                )

        # ax.legend(loc='upper center', bbox_to_anchor=(0.42, -0.17),
        #           ncol=3, title=None, frameon=False, fontsize=15,
        #           handlelength=1.6, columnspacing=0.5, labelspacing=0.0, handletextpad=0.0)
        # plt.subplots_adjust(bottom=0.3, top=0.992, left=0.14, right=0.995)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.97, 1.0),
            ncol=1,
            title=None,
            frameon=False,
            fontsize=15,
            handlelength=1.0,
            columnspacing=0.5,
            labelspacing=0.0,
            handletextpad=0.0,
        )
        plt.subplots_adjust(bottom=0.2, top=0.992, left=0.11, right=0.68)

        if fig_path is None:
            plt.show()
        else:
            plt.savefig(
                os.path.join(fig_path, "fig_place_" + node_type["id"] + ".png"), dpi=100
            )
            plt.clf()

    # df = place_df[place_df['node'] == 'all']
    # ax = sns.barplot(x='x', y='place', hue='opt', ci=None, data=df)
    # plt.show()

    # fg = sns.catplot(x='x', y='place_percent', hue='opt', col="node", kind='bar', ci=None, data=place_df,
    #                  legend=False)
    # fg.set_axis_labels(x_label, y_label)
    # fg.set_titles(col_template="{col_name}")
    # # fg.add_legend()
    # # fg.tight_layout()
    # # print(fg.legend)
    #


def plot_times(optimizers, experiments, data_path, fig_path=None):
    times_data = []
    for exp in experiments:
        # exp_path = os.path.join(data_path, exp['path'], '[0-9]*')
        exp_path = os.path.join(data_path, exp["path"], "0")
        runs_path = glob(exp_path)
        for opt in optimizers:
            for run, run_path in enumerate(runs_path):
                filename = os.path.join(run_path, opt["id"], "times.json")
                if not os.path.isfile(filename):
                    continue
                df = pd.read_json(filename, orient="records")
                df = (
                    df.groupby(["time", "id", "type"])["elapsed_time"]
                    .sum()
                    .reset_index()
                )

                global_df = df[df["type"] == "global"].reset_index()
                cluster_df = df[df["type"] == "cluster"].reset_index()
                if cluster_df.empty:
                    cluster_df = global_df
                df_map = {"global": global_df, "cluster": cluster_df}
                for ctrl_type, ctrl_df in iteritems(df_map):
                    for index, row in ctrl_df.iterrows():
                        elapsed_time = row["elapsed_time"]
                        if math.isnan(elapsed_time):
                            elapsed_time = 0.0
                        datum = {
                            "opt": opt["label"],
                            "run": run,
                            "x": exp["x"],
                            "elapsed_time": elapsed_time,
                            "step": index,
                            "ctrl_type": ctrl_type,
                        }
                        times_data.append(datum)

                # global_et = df[df["type"] == "global"]["elapsed_time"].mean()

                # cluster_et = global_et
                # max_ctrl_id = df[df["type"] == "cluster"]["id"].max()
                # if not math.isnan(max_ctrl_id):
                #     # cluster_et = df[(df['type'] == 'cluster') & (df['id'] < max_ctrl_id - 1)]['elapsed_time'].mean()
                #     cluster_et = df[df["type"] == "cluster"]["elapsed_time"].mean()

                # elapsed_times = {"global": global_et, "cluster": cluster_et}
                # for ctrl_type, elapsed_time in iteritems(elapsed_times):
                #     if math.isnan(elapsed_time):
                #         elapsed_time = 0.0
                #     datum = {
                #         "opt": opt["label"],
                #         "run": run,
                #         "x": exp["x"],
                #         "elapsed_time": elapsed_time,
                #         "ctrl_type": ctrl_type,
                #     }
                #     times_data.append(datum)

    times_df = pd.DataFrame.from_records(times_data)

    sns.set()
    # sns.set_context('paper', font_scale=2.0)
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks", {"axes.grid": True, "grid.linestyle": "--"})
    mpl.rcParams["hatch.linewidth"] = 0.5
    mpl.rcParams["figure.figsize"] = (7, 3)
    # plt.figure().set_figheight(4)

    opt_count = times_df["opt"].nunique()
    x_count = times_df["x"].nunique()
    hatches_unique = ["/", "\\", "-", "x", "+", "*", "o"]
    hatches = list(islice(cycle(hatches_unique), opt_count))

    ctrl_types = ["global", "cluster"]
    for ctrl_type in ctrl_types:
        df = times_df[times_df["ctrl_type"] == ctrl_type]
        # ax = sns.barplot(x="x", y="elapsed_time", hue="opt", ci=None, data=df)
        ax = sns.barplot(x="x", y="elapsed_time", hue="opt", ci=95, data=df)
        ax.xaxis.grid(True)
        ax.set_xlabel("Number of Subsystems")
        ax.set_ylabel("Execution Time (s)")

        hatch_index = -1
        for index, patch in enumerate(ax.patches):
            # Loop iterates an opt throughout x-axes, then it advances to the next opt
            if index % x_count == 0:
                hatch_index += 1
            hatch = hatches[hatch_index]
            patch.set_hatch(hatch)

            if patch.get_height() <= 1.0:
                value = int(round(patch.get_height(), 3) * 1000)
                ax.annotate(
                    "{:d}ms".format(value),
                    # (patch.get_x() + patch.get_width() / 2., patch.get_height()),
                    # (patch.get_x() + 0.05, patch.get_height()),
                    (patch.get_x() + 0.03, patch.get_height()),
                    color=patch.get_facecolor(),
                    ha="center",
                    va="center",
                    size=15,
                    xytext=(0, 9),
                    textcoords="offset points",
                )

        # ax.legend(loc='upper center', bbox_to_anchor=(0.44, -0.15),
        #           ncol=3, title=None, frameon=False, fontsize=15,
        #           handlelength=1.6, columnspacing=0.5, labelspacing=0.0, handletextpad=0.0)
        # plt.subplots_adjust(bottom=0.29, top=0.992, left=0.14, right=0.995)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.97, 1.0),
            ncol=1,
            title=None,
            frameon=False,
            fontsize=15,
            handlelength=1.0,
            columnspacing=0.5,
            labelspacing=0.0,
            handletextpad=0.0,
        )
        plt.subplots_adjust(bottom=0.2, top=0.992, left=0.09, right=0.68)

        if fig_path is not None:
            fig_filename = os.path.join(fig_path, "fig_times_" + ctrl_type + ".png")
            plt.savefig(fig_filename, dpi=100)
            plt.clf()
        else:
            plt.show()


def main():
    fig_path = "figs/"
    try:
        os.makedirs(fig_path)
    except OSError:
        pass

    data_path = "data"
    optimizers = [
        {"id": "CloudOptimizer", "label": "Cloud"},
        {"id": "LLCOptimizer_sga_w1_c25", "label": "Centralized"},
        {
            "id": "GlobalMOGAOptimizer_GeneralClusterLLGAOperator_w1_i0",
            "label": "Non-Coop-HDC",
        },
        {
            "id": "GlobalMOGAOptimizer_GeneralClusterLLGAOperator_w1_i1",
            "label": r"Coop-HDC $it_{\mathrm{max}}=1$",
        },
        {
            "id": "GlobalMOGAOptimizer_GeneralClusterLLGAOperator_w1_i2",
            "label": r"Coop-HDC $it_{\mathrm{max}}=2$",
        },
    ]
    experiments = [
        {"path": "n9_a10_u10000_c03", "x": 3},
        {"path": "n9_a10_u10000_c05", "x": 5},
        {"path": "n9_a10_u10000_c07", "x": 7},
        {"path": "n9_a10_u10000_c11", "x": 11},
    ]
    x_label = "Number of Subsystems"
    metrics = [
        {
            "id": "weighted_avg_deadline_violation",
            "label": "N. Deadline Violation",
            "normalize": True,
            "fig_file": os.path.join(fig_path, "fig_dv.png"),
        },
        {
            "id": "overall_cost",
            "label": "Operational Cost",
            "fig_file": os.path.join(fig_path, "fig_oc.png"),
        },
        {
            "id": "weighted_migration_rate",
            "label": "Migration Cost (%)",
            "func": to_percent,
            "y_limit": (0.0, 0.55),
            "fig_file": os.path.join(fig_path, "fig_mc.png"),
        },
    ]

    plot_metrics(optimizers, experiments, metrics, data_path, x_label)
    plot_placement_per_node_type(optimizers, experiments, data_path, x_label, fig_path)
    plot_times(optimizers, experiments, data_path, fig_path)


if __name__ == "__main__":
    main()
