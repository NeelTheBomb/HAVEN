from evaluation.evaluation_base import EvaluationBase
from sklearn.metrics import roc_curve, accuracy_score, f1_score, auc, precision_recall_curve
import pandas as pd
from statistics import mean
from utils import visualization_utils
from random import sample


class MultiClassEvaluation(EvaluationBase):
    def __init__(self, df, evaluation_settings, evaluation_output_file_base_path, visualization_output_file_base_path,
                 output_file_name, label_mappings):
        super().__init__(df, evaluation_settings, evaluation_output_file_base_path, visualization_output_file_base_path,
                         output_file_name)
        self.df = self.get_selected_df(label_mappings)
        self.y_pred_columns = self.df[self.y_true_col].unique()
        self.class_col = "class"

    # def get_y_pred_columns(self):
    #     y_pred_columns = list(self.df.columns.values)
    #     y_pred_columns.remove(self.itr_col)
    #     y_pred_columns.remove(self.y_true_col)
    #     y_pred_columns.remove(self.experiment_col)
    #     return y_pred_columns

    def get_selected_df(self, label_mappings):
        selected_labels = list(label_mappings.values())
        print(f"Size of results dataset = {self.df.shape[0]}")
        print(f"Selected labels = {selected_labels}")
        print(f"Number of selected labels = {len(selected_labels)}")
        selected_df = self.df[self.df[self.y_true_col].isin(selected_labels)]
        print(f"Size of selected results dataset = {selected_df.shape[0]}")
        return selected_df

    def compute_accuracy(self, df_itr):
        y_pred = self.convert_probability_to_prediction(df_itr)
        return accuracy_score(y_true=df_itr[self.y_true_col].values, y_pred=y_pred)

    def compute_f1(self, df_itr):
        y_pred = self.convert_probability_to_prediction(df_itr)
        return f1_score(y_true=df_itr[self.y_true_col].values, y_pred=y_pred, average="macro")

    def compute_auroc(self, df_itr):
        # macro auroc = unweighted average of auroc for each class
        # only for one-vs-rest setting
        roc_curves = []
        aurocs = []
        for y_pred_column in self.y_pred_columns:
            fpr, tpr, _ = roc_curve(y_true=df_itr[self.y_true_col].values, y_score=df_itr[y_pred_column].values, pos_label=y_pred_column)
            roc_curves.append(pd.DataFrame({"fpr": fpr, "tpr": tpr, self.class_col: y_pred_column}))
            aurocs.append(auc(fpr, tpr))

        return pd.concat(roc_curves, ignore_index=True), mean(aurocs)

    def compute_auprc(self, df_itr):
        # macro auprc = unweighted average of auprc for each class
        # only for one-vs-rest setting
        pr_curves = []
        auprcs = []
        for y_pred_column in self.y_pred_columns:
            precision, recall, _ = precision_recall_curve(y_true=df_itr[self.y_true_col].values, probas_pred=df_itr[y_pred_column].values, pos_label=y_pred_column)
            pr_curves.append(pd.DataFrame({"precision": precision, "recall": recall, self.class_col: y_pred_column}))
            auprcs.append(auc(recall, precision))

        return pd.concat(pr_curves, ignore_index=True), mean(auprcs)

    def convert_probability_to_prediction(self, df_itr):
        y_pred_prob = df_itr[self.y_pred_columns]
        return [y for y in y_pred_prob.idxmax(axis="columns")]

    def plot_visualizations(self):
        super().plot_visualizations()
        # for multiclass evaluation, we will plot the curves for one iteration (selected randomly) for each model_params.
        itr_selected = sample(list(self.evaluation_metrics_df[self.itr_col].values), 1).pop()
        if self.evaluation_settings["auroc"]:
            visualization_utils.box_plot(self.evaluation_metrics_df, self.experiment_col, "auroc",
                                         self.visualization_output_file_path + "_auroc_boxplot.pdf")

            visualization_utils.curve_plot(df=self.roc_curves_df[self.roc_curves_df[self.itr_col] == itr_selected], x_col="fpr", y_col="tpr",
                                           color_group_col=self.class_col, style_group_col=self.experiment_col,
                                           output_file_path=self.visualization_output_file_path + "_roc_curves.pdf", metadata=self.metadata)
        if self.evaluation_settings["auprc"]:
            visualization_utils.box_plot(self.evaluation_metrics_df, self.experiment_col, "auprc",
                                         self.visualization_output_file_path + "_auprc_boxplot.pdf")
            visualization_utils.curve_plot(df=self.pr_curves_df[self.pr_curves_df[self.itr_col] == itr_selected], x_col="recall", y_col="precision",
                                           color_group_col=self.class_col, style_group_col=self.experiment_col,
                                           output_file_path=self.visualization_output_file_path + "_precision_recall_curves.pdf", metadata=self.metadata)
        return

