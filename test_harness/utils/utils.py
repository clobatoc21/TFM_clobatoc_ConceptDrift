# Import libraries
import pandas as pd
import seaborn as sns


def format_experimental_scores(experiment):
    """Utility for extracting and formatting the scores of an experiment"""

    return (
        pd.DataFrame(experiment.experiment_metrics["scores"])
        .set_index(0)
        .rename(columns={1: experiment.name})
    )


def plot_experiment_error(experiment, show_trainings=True, ax=None):
    """Utility for plotting one experiment's cumulative accuracy"""

    scores_df = format_experimental_scores(experiment) # Obtain scores
    
    # Plot scores together with label expense, training time and number of drifts signaled
    ax.plot(scores_df, label=experiment.name)
    ax.title.set_text(f"Label Expense: {experiment.experiment_metrics['label_expense']['num_labels_requested']};\
    Total Train Time: {experiment.experiment_metrics['total_train_time']};\
    Drifts detected: {sum(experiment.drift_signals)}")
    ax.legend()
    ax.set_xlim([0, len(experiment.dataset.full_df)+50])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel("Cumulative Accuracy")

    # If requested, display training instances on the graph
    trainings = [item['training_spot'] for item in experiment.experiment_metrics["training"]]
    if show_trainings: 
        [
            ax.axvline(tr, color="black", linestyle=":", linewidth=0.75)
            for tr in trainings
        ]


def aggregate_experiment_metrics(metrics, exp, change_points, w_size, type_eval="Strict"):
    """Utility for aggregating metrics given an experiment"""

    for size_w in w_size:

        # Define allowed distance between true and detected drift for each evaluation scenario
        if type_eval == "Strict":
            limit = 200
        elif type_eval == "Medium":
            limit = 500
        elif type_eval == "Relaxed":
            limit = 800


        # Extract instances of drift signalization
        exp_size_w = exp[exp["Window_size"]==size_w]
        drift_entries = exp_size_w.loc[exp_size_w["Drift_signaled"]==True,
                                "Detection_end"].reset_index(drop=True)
        

        # Calculate true positives, false positives, true negatives, false negatives
        t_p = 0
        f_p = 0
        f_n = 0

        if len(drift_entries) == 0:
            f_n = len(change_points) # If no drift is detected, all true drifts are false negatives
        else:
            for j in range(len(change_points)):
                count_detected = 0
                for i in range(len(drift_entries)):
                    # If drift signal is between real drift and allowed distance, true positive.
                    # Count only if no other drift has been signaled for this real drift
                    if change_points[j] <= drift_entries[i] <= (change_points[j] + limit):
                        if count_detected == 0:
                            count_detected +=1
                            t_p += 1
                if count_detected == 0:
                    f_n += 1 # If no drift has been signaled, false negative
            f_p = len(drift_entries)-t_p

        t_n = 20000 - t_p - f_p - f_n # Length of dataset-(tp + fp + fn)


        # Calculate sensitivity, specificity and balanced accuracy
        if t_p != 0 or f_n !=0:
            sens = t_p/(t_p + f_n) # How many of the trues were detected as true (recall)
        else:
            sens = "N/A"

        if t_n != 0 or f_p !=0:
            spec = t_n/(t_n + f_p) # How many of the falses were detected as false
        else:
            spec = "N/A"

        if (sens=="N/A") or (spec=="N/A"):
            bal_acc = 0
        else:
            bal_acc = (sens+spec)/2


        # Consolidate metrics
        metrics.append(
            {
                "Evaluation": type_eval,
                "Experiment": exp_size_w["Exp_name"].iloc[-1],
                "Window sizes": size_w,
                "Drift signaled": len(drift_entries),
                "Drift occurred": len(change_points),
                "True positives": t_p,
                "False positives": f_p,
                "True negatives": t_n,
                "False negatives": f_n,
                "Cum. accuracy": exp_size_w["Det_acc"].iloc[-1],
                "Sensitivity": sens,
                "Specificity": spec,
                "Bal. acc": bal_acc,
            }
        )

    return metrics


def plot_KS_drift_distributions(results, w_size):
    """Utility for Exp 2 - Uncertainty KS Experiment"""

    for size_w in w_size:

        # Extract instances of drift signalization
        results_size_w = results[results["Window_size"]==size_w]
        drift_entries = results_size_w.loc[results_size_w["Drift_signaled"]==True,
                                    "Detection_end"]
        
        # Plot distribution of reference vs detection window for every drift signaled
        if len(drift_entries)>0:
            results_drifts = pd.DataFrame(columns=["Reference_dist","Detection_dist","Detection_end"])

            for drift in drift_entries:
                ref_dist = results_size_w.loc[results_size_w["Detection_end"]==drift, "Ref_dist"].values.tolist()
                ref_dist = [float(el) for el in ref_dist[0].replace("[","").replace("]","").split(', ')]
                det_dist = results_size_w.loc[results_size_w["Detection_end"]==drift, "Det_dist"].values.tolist()
                det_dist = [float(el) for el in det_dist[0].replace("[","").replace("]","").split(', ')]

                dist_df = pd.DataFrame({"Reference_dist":ref_dist,"Detection_dist":det_dist})
                dist_df["Detection_end"] = drift
                results_drifts = pd.concat([results_drifts,dist_df],axis=0)

            df_melt = results_drifts.melt(id_vars=["Detection_end"], var_name="Window Type")

            g = sns.FacetGrid(df_melt, col="Detection_end", hue="Window Type", col_wrap=4)
            g.map_dataframe(sns.kdeplot, "value", fill=True)
            g.add_legend()
            g.fig.subplots_adjust(top=0.75)
            g.fig.suptitle(f'Reference and Detection distributions on suggested drifts, window size = {size_w}')


def plot_XS_drift_margin_distributions(results, w_size):
    """Utility for Exp 3 - Uncertainty X2 Experiment"""

    for size_w in w_size:

        # Extract instances of drift signalization
        results_size_w = results[results["Window_size"]==size_w]
        drift_entries = results_size_w.loc[results_size_w["Drift_signaled"]==True,
                                    "Detection_end"]
        
        # Plot distribution of reference vs detection window for every drift signaled
        if len(drift_entries)>0:
            results_drifts = pd.DataFrame(columns=["Reference_margin_dist","Detection_margin_dist","Detection_end"])

            for drift in drift_entries:
                ref_dist = results_size_w.loc[results_size_w["Detection_end"]==drift, "Ref_margins"].values.tolist()
                ref_dist = [float(el) for el in ref_dist[0].replace("[","").replace("]","").split(', ')]
                det_dist = results_size_w.loc[results_size_w["Detection_end"]==drift, "Det_margins"].values.tolist()
                det_dist = [float(el) for el in det_dist[0].replace("[","").replace("]","").split(', ')]

                dist_df = pd.DataFrame({"Reference_margin_dist":ref_dist,"Detection_margin_dist":det_dist})
                dist_df["Detection_end"] = drift
                results_drifts = pd.concat([results_drifts,dist_df],axis=0)

            df_melt = results_drifts.melt(id_vars=["Detection_end"], var_name="Window Type")

            g = sns.FacetGrid(df_melt, col="Detection_end", hue="Window Type", col_wrap=4)
            g.map_dataframe(sns.kdeplot, "value", fill=True)
            g.add_legend()
            g.fig.subplots_adjust(top=0.75)
            g.fig.suptitle(f'Reference and Detection Margin distributions on suggested drifts, window size = {size_w}')