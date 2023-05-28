
# import libraries
import numpy as np
import pandas as pd
from csv import writer
from scipy.stats import chisquare, fisher_exact
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from test_harness.experiments._1_baseline_experiment import BaselineExperiment

# define class
class UncertaintyX2Experiment(BaselineExperiment):
    def __init__(
        self, model, dataset, k, significance_thresh, margin_width, param_grid=None,delete_csv=False
    ):
        super().__init__(model, dataset, param_grid)
        self.name = "3_Margin_uncertainty"
        self.k = k
        self.significance_thresh = significance_thresh
        self.ref_distributions = []
        self.ref_margins = []
        self.det_distributions = []
        self.det_margins = []
        self.p_vals = []
        self.margin_width = margin_width
        self.drift_entries = []
        self.acc_det_aux = 100 # random number out of range so that first diff is significant
        self.entry_det_aux = 0 # 0 so that first diff is significant
        self.delete_csv = delete_csv

    @staticmethod
    def make_kfold_predictions(X, y, model, dataset, k):
        """A KFold version of LeaveOneOut predictions.

        When k == len(X), this is equivalent to LeaveOneOut: expensive, but robust. Reducing k
        saves computation, but reduces robustness of model.

        Args:
            X (pd.Dataframe) - features in evaluation window
            y (pd.Series) - labels in evaluation window
            k (int) - number of folds
            type (str) - specified kfold or LeaveOneOut split methodology

        Returns:
            preds (np.array) - an array of predictions for each X in the input (NOT IN ORDER OF INPUT)
        """

        splitter = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)

        preds = np.array([])
        pred_margins = np.array([])
        split_ACCs = np.array([])

        for train_indicies, test_indicies in splitter.split(X, y):

            # create column transformer
            column_transformer = ColumnTransformer(
                [
                    (
                        "floats",
                        StandardScaler(),
                        dataset.column_mapping["float_features"],
                    ),
                    (
                        "integers",
                        "passthrough",
                        dataset.column_mapping["int_features"],
                    ),
                ]
            )

            # instantiate training pipeline
            pipe = Pipeline(
                steps=[
                    ("scaler", column_transformer),
                    ("clf", model),
                ]
            )

            # fit it
            pipe.fit(X.iloc[train_indicies], y.iloc[train_indicies])

            # score it on this Kfold's test data
            y_preds_split = pipe.predict_proba(X.iloc[test_indicies])

            # get positive class prediction
            y_preds_split_posclass_proba = y_preds_split[:, 1]
            preds = np.append(preds, y_preds_split_posclass_proba)

            # get pred margins
            # https://github.com/SeldonIO/alibi-detect/blob/86dc3148ee5a3726fb6229d5369c38e7e97b6040/alibi_detect/cd/preprocess.py#L49
            top_2_probs = -np.partition(-y_preds_split, kth=1, axis=-1)
            diffs = top_2_probs[:, 0] - top_2_probs[:, 1] # Difference in the probability of 2 top classes
            pred_margins = np.append(pred_margins, diffs)

            # get accuracy for split
            split_ACC = pipe.score(X.iloc[test_indicies], y.iloc[test_indicies])
            split_ACCs = np.append(split_ACCs, split_ACC)

        return preds, pred_margins, split_ACCs

    def get_reference_response_distribution(self):

        # get data in reference window
        window_start = self.reference_window_start
        window_end = self.reference_window_end
        
        X_train, y_train = self.dataset.get_data_by_idx(window_start, window_end, split_labels=True)

        # perform kfoldsplits to get predictions
        preds, pred_margins, split_ACCs = self.make_kfold_predictions(
            X_train, y_train, self.model, self.dataset, self.k
        )

        ref_ACC = np.mean(split_ACCs)
        ref_ACC_SD = np.std(split_ACCs)

        return preds, pred_margins, ref_ACC, ref_ACC_SD

    def get_detection_response_distribution(self):

        # get data in prediction window
        window_start = self.detection_window_start
        window_end = self.detection_window_end
        X_test, y_test = self.dataset.get_data_by_idx(window_start, window_end, split_labels=True)

        # use trained model to get response distribution
        y_preds_split = self.trained_model.predict_proba(X_test)
        preds = y_preds_split[:, 1] # positive class prediction

        # get pred margins
        # https://github.com/SeldonIO/alibi-detect/blob/86dc3148ee5a3726fb6229d5369c38e7e97b6040/alibi_detect/cd/preprocess.py#L49
        top_2_probs = -np.partition(-y_preds_split, kth=1, axis=-1)
        pred_margins = top_2_probs[:, 0] - top_2_probs[:, 1] # Difference in the probability of 2 top classes

        # get accuracy for detection window
        det_ACC = self.evaluate_model_aggregate(window="detection")[1]

        return preds, pred_margins, det_ACC

    def calculate_errors(self):

        self.false_positives = [
            True if self.drift_signals[i] and not self.drift_occurences[i] else False
            for i in range(len(self.drift_signals))
        ]
        self.false_negatives = [
            True if not self.drift_signals[i] and self.drift_occurences[i] else False
            for i in range(len(self.drift_signals))
        ]

    def run(self):
        """Response Margin Uncertainty Experiment

        This experiment uses a X2 test to detect changes in the margin of the target distribution between
        the reference window and the detection window.

        Logic flow:
            - Train on initial reference window
            - Perform Stratified KFold to obtain prediction distribution on reference window
            - Use trained model to generate predictions on detection window
            - Calculate the difference between confidence values of binary classes for each observation in both windows (aka margin)
            - Use the specified margin threshold (e.g. 0.1 for [0.45, 0.55]) to assign binary class to each observation (e.g. in or out of margin)
            - Perform Chi-Squared Goodness of Fit Test between reference and detection window response margins
                - If different, retrain and update both windows
                - If from same distribution, update detection window and repeat
        """

        self.train_model_gscv(window="reference", gscv=True)

        # ------------------------Create csv for storing results----------------#
        if self.delete_csv==True:
            cols = ["Exp_name", "Window_size", "Sign_thres", "Margin_width", "Detection_end", "Det_acc", "Ref_dist", 
                    "Det_dist", "Ref_margins", "Det_margins", "Ref_uncert", "Det_uncert","Chi_result",
                     "Drift_signaled", "Real_drift", "Total_Training_time"]

            entries_df = pd.DataFrame(columns=cols)
            entries_df.to_csv(f"./results/{self.dataset.name}_{self.name}_results.csv", index=False)
        # ---------------------------------------------------------------------#

        # initialize score
        self.experiment_metrics["scores"].append(self.evaluate_model_aggregate())
        self.update_detection_window()

        CALC_REF_RESPONSE = True

        while self.detection_window_end <= len(self.dataset.full_df):

            # log actual score on detection window
            self.experiment_metrics["scores"].append(self.evaluate_model_incremental())

            # get reference window response distribution with kfold and the detection response distribution
            if CALC_REF_RESPONSE:
                ref_response_dist, ref_response_margins, ref_ACC, ref_ACC_SD = self.get_reference_response_distribution()
            
            det_response_dist, det_response_margins, det_ACC = self.get_detection_response_distribution()

            # save reference window items
            self.ref_distributions.append(ref_response_dist)
            self.ref_margins.append(ref_response_margins)

            # save detection window items
            self.det_distributions.append(det_response_dist)
            self.det_margins.append(det_response_margins)

            # compare change in margin use Chi Squared test for goodness of fit
            ref_uncertainties = (ref_response_margins < self.margin_width).astype(int)
            det_uncertainties = (det_response_margins < self.margin_width).astype(int)

            len_ref = len(ref_response_margins)
            len_det = len(det_response_margins)

            if sum(det_uncertainties)==0:
                observed = [len_det,0]
            elif sum(det_uncertainties)==len_det:
                observed = [0,len_det]
            else:
                observed = pd.Series(det_uncertainties).value_counts(normalize=False).tolist()

            if sum(ref_uncertainties)==0:
                expected = [len_ref,0]
                x2_result = fisher_exact([observed, expected]) # dividing by 0 generates error, hence using fisher instead
            elif sum(ref_uncertainties)==len_ref:
                expected = [0,len_ref]
                x2_result = fisher_exact([observed, expected]) # dividing by 0 generates error, hence using fisher instead
            else:
                expected = pd.Series(ref_uncertainties).value_counts(normalize=False).tolist()
                x2_result = chisquare(f_obs=observed, f_exp=expected)
        
            self.p_vals.append(x2_result.pvalue)

            significant_change = (
                True if x2_result[1] < self.significance_thresh else False
            )
            self.drift_signals.append(significant_change)

            # compare accuracies to see if detection was false alarm
            # i.e. check if change in accuracy is significant
            delta_ACC = np.absolute(det_ACC - ref_ACC)
            delta_ACC_det = np.absolute(det_ACC - self.acc_det_aux)
            diff_entries = self.detection_window_end - self.entry_det_aux
            threshold_ACC = 3 * ref_ACC_SD  # considering outside 3 SD significant
            if (
                (self.acc_det_aux==100 or delta_ACC_det > threshold_ACC) # difference vs previous entry
                 and delta_ACC > threshold_ACC                           # difference vs reference
                 and diff_entries > self.dataset.window_size+1           # distance from previous True
                 ):
                significant_ACC_change = True
                self.acc_det_aux = det_ACC
                self.entry_det_aux = self.detection_window_end
            else:
                significant_ACC_change = False
            self.drift_occurences.append(significant_ACC_change)


            # ------------------------Store results------------------------------#
            if significant_change:
                new_entry = [self.name, self.dataset.window_size, self.significance_thresh, 
                            self.margin_width, self.detection_window_end, self.acc, ref_response_dist.tolist(),
                            det_response_dist.tolist(), ref_response_margins.tolist(), det_response_margins.tolist(), 
                            ref_uncertainties.tolist(), det_uncertainties.tolist(), x2_result.pvalue, significant_change,
                            significant_ACC_change, self.total_train_time]
            else:
                new_entry = [self.name, self.dataset.window_size, self.significance_thresh, 
                            self.margin_width, self.detection_window_end, self.acc, "",
                            "", "", "", "", "", x2_result.pvalue, significant_change,
                            significant_ACC_change, self.total_train_time]
            
            
            with open(f"./results/{self.dataset.name}_{self.name}_results.csv", 'a', newline='') as f_object:
 
                # Pass this file object to csv.writer() and get a writer object
                writer_object = writer(f_object)
 
                # Pass the list as an argument into the writerow()
                writer_object.writerow(new_entry)
 
                # Close the file object
                f_object.close()

            #----------------------------------------------------------------------#

            if significant_change:
                self.drift_entries.append(self.detection_window_end)

                # reject null hyp, distributions are NOT the same --> retrain
                self.reference_window_start = self.detection_window_start
                self.reference_window_end = self.detection_window_end
                self.train_model_gscv(window="reference", gscv=True)

                # update detection window and reset score
                self.detection_window_start = self.detection_window_end
                self.detection_window_end = self.detection_window_end + self.dataset.window_size
                self.experiment_metrics["scores"].append(self.evaluate_model_aggregate())
                CALC_REF_RESPONSE = True
            else:
                CALC_REF_RESPONSE = False

            self.update_reference_window()
            self.update_detection_window()

        self.calculate_label_expense()
        self.calculate_train_expense()
        self.calculate_errors()
