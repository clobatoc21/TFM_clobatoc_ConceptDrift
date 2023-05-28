import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def format_experimental_scores(experiment):
    return (
        pd.DataFrame(experiment.experiment_metrics["scores"])
        .set_index(0)
        .rename(columns={1: experiment.name})
    )


def plot_experiment_error(experiment, show_trainings=True, ax=None):
    """Utility for plotting single experiment's cumulative accuracy"""

    scores_df = format_experimental_scores(experiment)
    
    ax.plot(scores_df, label=experiment.name)
    ax.title.set_text(f"Label Expense: {experiment.experiment_metrics['label_expense']['num_labels_requested']};\
    Total Train Time: {experiment.experiment_metrics['total_train_time']};\
    Drifts detected: {sum(experiment.drift_signals)}")
    ax.legend()
    ax.set_xlim([0, len(experiment.dataset.full_df)+50])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Cumulative Accuracy")

    trainings = [item['training_spot'] for item in experiment.experiment_metrics["training"]]

    if show_trainings: 
        [
            ax.axvline(tr, color="black", linestyle=":", linewidth=0.75)
            for tr in trainings
        ]


def plot_multiple_experiments(experiments, change_points=None):
    """Utility for plotting multiple experiment's cumulative accuracy"""

    exp_dfs = [format_experimental_scores(experiment) for experiment in experiments]

    ax = pd.concat(exp_dfs, axis=1).plot(
        figsize=(12, 4),
        title="Cumulative Accuracy on Data Stream by Drift Detection Method",
        xlabel="Observations",
        ylabel="Cumulative Accuracy",
    )

    if change_points:
        [
            ax.axvline(i, color="black", linestyle=":", linewidth=0.75)
            for i in change_points
            if i != 0
        ]
    plt.show()


def aggregate_experiment_metrics(experiments):
    """Utility for aggregating metrics given multiple experiments"""

    metrics = []

    for exp in experiments:

        t_n = sum([True if not exp.drift_signals[i] and not exp.drift_occurences[i] else False
            for i in range(len(exp.drift_signals))])

        t_p = sum([True if exp.drift_signals[i] and exp.drift_occurences[i] else False
            for i in range(len(exp.drift_signals))])
        
        f_n = sum(exp.false_negatives)
        f_p = sum(exp.false_positives)

        #acc = (t_p + t_n)/len(exp.drift_signals) # How many were guessed right
        if t_p != 0 or f_p !=0:
            prec = t_p/(t_p + f_p) # How many out of the said trues were actually true
        else:
            prec = "N/A"

        if t_p != 0 or f_n !=0:
            recall = t_p/(t_p + f_n) # How many of the trues were detected as true
        else:
            recall = "N/A"

        if (prec=="N/A") or (recall=="N/A") or (prec + recall == 0):
            f1 = 0
        else:
            f1 = 2*prec*recall/(prec + recall)

        metrics.append(
            {
                "Experiment": exp.name,
                "Total training": exp.experiment_metrics["total_train_time"],
                "% Labels used": exp.experiment_metrics["label_expense"][
                    "percent_total_labels"],
                "Drift signaled": sum(exp.drift_signals),
                "Drift occurred": sum(exp.drift_occurences),
                "False positives": f_p,
                "False negatives": f_n,
                "Cum. accuracy": exp.experiment_metrics["scores"][-1][1],
                #"Accuracy": acc,
                "F1 Score": f1,
            }
        )

    return pd.DataFrame(metrics).set_index("Experiment")


def plot_KS_drift_distributions(results, w_size):
    """Utility for UncertaintyKSExperiment"""

    for size_w in w_size:

        results_size_w = results[results["Window_size"]==size_w]
        drift_entries = results_size_w.loc[results_size_w["Drift_signaled"]==True,
                                    "Detection_end"]
        
        if len(drift_entries)>0:
            results_drifts = pd.DataFrame(columns=["Reference_dist","Detection_dist","Detection_end"])

            for drift in drift_entries:
                ref_dist = results_size_w.loc[results_size_w["Detection_end"]==drift, "Ref_dist"].values.tolist()
                ref_dist = [float(el) for el in ref_dist[0].replace("[","").replace("]","").split(', ')]
                det_dist = results_size_w.loc[results_size_w["Detection_end"]==drift, "Det_dist"].values.tolist()
                det_dist = [float(el) for el in det_dist[0].replace("[","").replace("]","").split(', ')]

                dist_df = pd.DataFrame({"Reference_dist":ref_dist,"Detection_dist":det_dist})
                dist_df["Detection_end"] = drift
                #print(dist_df)
                results_drifts = pd.concat([results_drifts,dist_df],axis=0)

            #print(results_drifts)
            df_melt = results_drifts.melt(id_vars=["Detection_end"], var_name="Window Type")

            g = sns.FacetGrid(df_melt, col="Detection_end", hue="Window Type", col_wrap=4)
            g.map_dataframe(sns.kdeplot, "value", fill=True)
            g.add_legend()
            g.fig.subplots_adjust(top=0.75)
            g.fig.suptitle(f'Reference and Detection distributions on suggested drifts, window size = {size_w}')


def plot_XS_drift_margin_distributions(results, w_size):
    """Utility for UncertaintyX2Experiment"""

    for size_w in w_size:

        results_size_w = results[results["Window_size"]==size_w]
        drift_entries = results_size_w.loc[results_size_w["Drift_signaled"]==True,
                                    "Detection_end"]
        
        if len(drift_entries)>0:
            results_drifts = pd.DataFrame(columns=["Reference_margin_dist","Detection_margin_dist","Detection_end"])

            for drift in drift_entries:
                ref_dist = results_size_w.loc[results_size_w["Detection_end"]==drift, "Ref_margins"].values.tolist()
                ref_dist = [float(el) for el in ref_dist[0].replace("[","").replace("]","").split(', ')]
                det_dist = results_size_w.loc[results_size_w["Detection_end"]==drift, "Det_margins"].values.tolist()
                det_dist = [float(el) for el in det_dist[0].replace("[","").replace("]","").split(', ')]

                dist_df = pd.DataFrame({"Reference_margin_dist":ref_dist,"Detection_margin_dist":det_dist})
                dist_df["Detection_end"] = drift
                #print(dist_df)
                results_drifts = pd.concat([results_drifts,dist_df],axis=0)

            #print(results_drifts)
            df_melt = results_drifts.melt(id_vars=["Detection_end"], var_name="Window Type")

            g = sns.FacetGrid(df_melt, col="Detection_end", hue="Window Type", col_wrap=4)
            g.map_dataframe(sns.kdeplot, "value", fill=True)
            g.add_legend()
            g.fig.subplots_adjust(top=0.75)
            g.fig.suptitle(f'Reference and Detection Margin distributions on suggested drifts, window size = {size_w}')