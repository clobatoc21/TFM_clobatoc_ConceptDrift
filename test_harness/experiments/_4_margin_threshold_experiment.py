
# import libraries
import numpy as np
import pandas as pd
from csv import writer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from test_harness.experiments._1_baseline_experiment import BaselineExperiment

# define class
class MarginThresholdExperiment(BaselineExperiment):
    def __init__(
        self, model, dataset, k, margin_width, sensitivity, param_grid=None,delete_csv=False
    ):
        super().__init__(model, dataset, param_grid)
        self.name = f"4_Margin_threshold-S{str(sensitivity).replace('.','_')}"
        self.k = k
        self.margin_width = margin_width
        self.sensitivity = sensitivity
        self.drift_entries = []
        self.delete_csv = delete_csv

        self.ref_distributions = []
        self.ref_margins = []
        self.ref_MDs = []
        self.ref_SDs = []
        self.ref_ACCs = []
        self.ref_ACC_SDs = []

        self.det_distributions = []
        self.det_margins = []
        self.det_MDs = []
        self.det_ACCs = []
        self.acc_det_aux = 100 # random number out of range so that first diff is significant
        self.entry_det_aux = 0 # 0 so that first diff is significant

    @staticmethod
    def make_kfold_predictions(X, y, model, dataset, k, margin_width):
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
        split_MDs = np.array([])
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

            # get margin density for split (# entries in-margin / # entries total)
            aux_split_MD = pd.Series(diffs < margin_width).astype(int)

            if sum(aux_split_MD)==0:
                split_MD = 0
            else:
                split_MD = aux_split_MD.value_counts(normalize=True)[1]

            split_MDs = np.append(split_MDs, split_MD)

            # get accuracy for split
            split_ACC = pipe.score(X.iloc[test_indicies], y.iloc[test_indicies])
            split_ACCs = np.append(split_ACCs, split_ACC)

        return preds, pred_margins, split_MDs, split_ACCs

    def get_reference_response_distribution(self):

        # get data in reference window
        window_start = self.reference_window_start
        window_end = self.reference_window_end

        X_train, y_train = self.dataset.get_data_by_idx(window_start, window_end, split_labels=True)

        # perform kfoldsplits to get predictions
        preds, pred_margins, split_MDs, split_ACCs = self.make_kfold_predictions(
            X_train, y_train, self.model, self.dataset, self.k, self.margin_width
        )

        ref_MD = np.mean(split_MDs)
        ref_SD = np.std(split_MDs)

        ref_ACC = np.mean(split_ACCs)
        ref_ACC_SD = np.std(split_ACCs)

        return preds, pred_margins, ref_MD, ref_SD, ref_ACC, ref_ACC_SD

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

        aux_det_MD = pd.Series(pred_margins < self.margin_width).astype(int)
            
        if sum(aux_det_MD)==0:
            det_MD = 0
        else:
            det_MD = aux_det_MD.value_counts(normalize=True)[1]

        # get accuracy for detection window
        det_ACC = self.evaluate_model_aggregate(window="detection")[1]

        return preds, pred_margins, det_MD, det_ACC

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
        """Response Margin Threshold Experiment

        This experiment uses a threshold/sensitivity to detect changes in the margin of the target distribution between
        the reference window and the detection window.

        Logic flow:
            - Train on initial reference window
            - Perform Stratified KFold to obtain "margin" values for each prediction
                - Apply a user defined "margin threshold" to classify each observation as "in-margin" or "out-of-margin"
                - Calculate a MD metric for the given split (MD = # samples in-margin / # samples total)
                - Summarize the kfold split MD metrics into:
                    - The expected margin density (MD_reference): average MD over k-splits
                    - The acceptable deviation of the MD metric (SD_reference): standard deviation of MD over k-splits
            - Use trained model to generate predictions on detection window and calculate MD_detection
            - Check if MD_detection deviates by more than S standard deviations from MD_reference (if so, a concept drift is detected)
                - If drift is detected, retrain and update both windows
                - If drift is not-detected, update detection window and repeat
        """

        self.train_model_gscv(window="reference", gscv=True)

        # ------------------------Create csv for storing results----------------#
        if self.delete_csv==True:
            cols = ["Exp_name", "Window_size", "Margin_width", "Sensitivity", "Threshold", "Detection_end", "Det_acc",
                    "Ref_dist", "Det_dist", "Ref_margins", "Det_margins", "Ref_MD", "Det_MD", 
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
                ref_response_dist, ref_response_margins, ref_MD, ref_SD, ref_ACC, ref_ACC_SD = self.get_reference_response_distribution()

            det_response_dist, det_response_margins, det_MD, det_ACC = self.get_detection_response_distribution()

            # save reference window items
            self.ref_distributions.append(ref_response_dist)
            self.ref_margins.append(ref_response_margins)
            self.ref_MDs.append(ref_MD)
            self.ref_SDs.append(ref_SD)
            self.ref_ACCs.append(ref_ACC)
            self.ref_ACC_SDs.append(ref_ACC_SD)

            # save detection window items
            self.det_distributions.append(det_response_dist)
            self.det_margins.append(det_response_margins)
            self.det_MDs.append(det_MD)
            self.det_ACCs.append(det_ACC)

            # compare margin densities to detect drift
            delta_MD = np.absolute(det_MD - ref_MD)
            threshold = self.sensitivity * ref_SD
            significant_MD_change = True if delta_MD > threshold else False
            self.drift_signals.append(significant_MD_change)

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
            if significant_MD_change:
                new_entry = [self.name, self.dataset.window_size, self.margin_width, self.sensitivity, threshold,
                            self.detection_window_end, self.acc, ref_response_dist.tolist(), det_response_dist.tolist(),
                            ref_response_margins.tolist(), det_response_margins.tolist(), ref_MD, 
                            det_MD, significant_MD_change, significant_ACC_change, self.total_train_time]
            else:
                new_entry = [self.name, self.dataset.window_size, self.margin_width, self.sensitivity, threshold,
                            self.detection_window_end, self.acc, "", "", "", "", ref_MD,
                            det_MD, significant_MD_change, significant_ACC_change, self.total_train_time]                
            
            with open(f"./results/{self.dataset.name}_{self.name}_results.csv", 'a', newline='') as f_object:
 
                # Pass this file object to csv.writer() and get a writer object
                writer_object = writer(f_object)
 
                # Pass the list as an argument into the writerow()
                writer_object.writerow(new_entry)
 
                # Close the file object
                f_object.close()

            #----------------------------------------------------------------------#

            if significant_MD_change:
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
