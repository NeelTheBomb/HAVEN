import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

DEFAULT_FIGURE_CONFIG = {
    "figsize": (12, 7),
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
}


def box_plot(df, x_col, y_col, output_file_path, baseline=None, figure_config=DEFAULT_FIGURE_CONFIG):
    pre_plot_config(figure_config)
    ax = sns.boxplot(data=df, x=x_col, y=y_col)
    if baseline is not None:
        ax.axhline(baseline, color="gray", linestyle="--")
    ax.set_ylim(0.0, 1)
    ax.set_xlabel("", size=20, labelpad=5)
    ax.set_ylabel(y_col.upper(), size=20)
    wrap_ticklabels(ax, axis="x", label_width=14, break_long_words=True)
    plt.xticks(rotation=90)
    view(output_file_path)


def curve_plot(df, x_col, y_col, color_group_col, style_group_col, output_file_path, metadata=None, figure_config=DEFAULT_FIGURE_CONFIG):
    pre_plot_config(figure_config)
    hue_order = None
    if metadata is not None:
        df = df.replace({color_group_col: metadata})
        hue_order = list(metadata.values())
    ax = sns.lineplot(data=df, x=x_col, y=y_col, hue=color_group_col, style=style_group_col, hue_order=hue_order)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(x_col.upper(), size=20, labelpad=5)
    ax.set_ylabel(y_col.upper(), size=20)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    view(output_file_path)


def heat_map(df, output_file_path=None, figure_config=DEFAULT_FIGURE_CONFIG):
    pre_plot_config(figure_config)
    sns.heatmap(df)
    view(output_file_path)


def class_distribution_plot(df, output_file_path, figure_config=DEFAULT_FIGURE_CONFIG):
    pre_plot_config(figure_config)
    ax = sns.barplot(data=df, x="label", y="label_count", hue="group")
    ax.bar_label(ax.containers[0], fontsize=10)
    plt.xticks(rotation=20)
    view(output_file_path)


def feature_prevalence_distribution_plot(df, output_file_path, figure_config=DEFAULT_FIGURE_CONFIG):
    pre_plot_config(figure_config)
    ax = sns.displot(data=df)
    ax.set_xlabels("Prevalence of feature (%)")
    ax.set_ylabels("Number of features")
    view(output_file_path)


def top_k_features_box_plot(df, output_file_path, figure_config=DEFAULT_FIGURE_CONFIG):
    pre_plot_config(figure_config)
    ax = sns.boxplot(data=df, x="value", y="variable")
    ax.set_xlabel("Mean absolute coefficient across iterations")
    ax.set_ylabel("3-mer")
    view(output_file_path)


def feature_imp_by_prevalence_scatter_plot(df, output_file_path, figure_config=DEFAULT_FIGURE_CONFIG):
    pre_plot_config(figure_config)
    ax = sns.scatterplot(data=df, x="prevalence", y="imp", s=4)
    ax.set_xlabel("Prevalence (%)")
    ax.set_ylabel("Mean absolute coefficient across iterations")
    view(output_file_path)


def validation_scores_multiline_plot(df, output_file_path, figure_config=DEFAULT_FIGURE_CONFIG):
    pre_plot_config(figure_config)
    sns.lineplot(data=df, x="variable", y="value", hue="split")
    plt.xticks(rotation=90)
    view(output_file_path)


def pre_plot_config(figure_config=DEFAULT_FIGURE_CONFIG):
    plt.clf()
    plt.figure(figsize=figure_config["figsize"])
    sns.set_theme()
    sns.set_style("whitegrid")
    sns.color_palette("colorblind", as_cmap=True)
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["xtick.labelsize"] = figure_config["xtick.labelsize"]
    plt.rcParams["ytick.labelsize"] = figure_config["ytick.labelsize"]


def view(output_file_path=None):
    plt.tight_layout()
    if output_file_path:
        plt.savefig(output_file_path)


def wrap_ticklabels(ax, axis, label_width=10, break_long_words=False):
    wrapped_labels = []
    labels=None
    if axis == "x":
        labels = ax.get_xticklabels()
    elif axis == "y":
        labels = ax.get_yticklabels()
    else:
        print(f"ERROR: Invalid axis = {axis}. Supported values = 'x', 'y'.")

    for label in labels:
        label_text = label.get_text()
        wrapped_labels.append(textwrap.fill(text=label_text, width=label_width, break_long_words=break_long_words))

    if axis == "x":
        ax.set_xticklabels(wrapped_labels, rotation=0)
    elif axis == "y":
        ax.set_yticklabels(wrapped_labels, rotation=0)
    else:
        print(f"ERROR: Invalid axis = {axis}. Supported values = 'x', 'y'.")


def get_p_value_annotation(p_value):
    if 5e-2 < p_value <= 1:
        return "ns"
    elif 1e-2 < p_value <= 5e-2:
        return "*"
    elif 1e-3 < p_value <= 1e-2:
        return "**"
    elif 1e-4 < p_value <= 1e-3:
        return "***"
    elif p_value <= 1e-4:
        return "****"
    else:
        return None
