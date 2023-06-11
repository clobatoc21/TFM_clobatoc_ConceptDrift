
# import libraries
import numpy as np
import pandas as pd
from csv import writer
from scipy.stats import ks_2samp
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from test_harness.experiments._1_baseline_experiment import BaselineExperiment

# define class
class UncertaintyKSExperiment(BaselineExperiment):
    def __init__(self, model, dataset, k, significance_thresh, param_grid=None,delete_csv=False):
        super().__init__(model, dataset, param_grid)
        self.name = "2_Response_uncertainty"
        self.k = k
        self.significance_thresh = significance_thresh
        self.ref_distributions = []
        self.det_distributions = []
        self.p_vals = []
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

            # get accuracy for split
            split_ACC = pipe.score(X.iloc[test_indicies], y.iloc[test_indicies])
            split_ACCs = np.append(split_ACCs, split_ACC)

        return preds, split_ACCs

    def get_reference_response_distribution(self):

        # get data in reference window
        window_start = self.reference_window_start
        window_end = self.reference_window_end
        
        X_train, y_train = self.dataset.get_data_by_idx(window_start, window_end, split_labels=True)

        # perform kfoldsplits to get predictions
        preds, split_ACCs = self.make_kfold_predictions(
            X_train, y_train, self.model, self.dataset, self.k
        )

        ref_ACC = np.mean(split_ACCs)
        ref_ACC_SD = np.std(split_ACCs)

        return preds, ref_ACC, ref_ACC_SD

    def get_detection_response_distribution(self):

        # get data in prediction window
        window_start = self.detection_window_start
        window_end = self.detection_window_end
        X_test, y_test = self.dataset.get_data_by_idx(window_start, window_end, split_labels=True)

        # use trained model to get response distribution
        preds = self.trained_model.predict_proba(X_test)[:, 1] # positive class prediction

        # get accuracy for detection window
        det_ACC = self.evaluate_model_aggregate(window="detection")[1]

        return preds, det_ACC

    @staticmethod
    def perform_ks_test(dist1, dist2):
        return ks_2samp(dist1, dist2, method='asymp')

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
        """Response Uncertainty Experiment

        This experiment uses a KS test to detect changes in the target distribution between
        the reference window and the detection window.

        Logic flow:
            - Train on initial reference window
            - Perform Stratified KFold to obtain prediction distribution on reference window
            - Use trained model to generate predictions on detection window
            - Perform statistical test (KS) between reference and detection window response distributions
                - If different, retrain and update both windows
                - If from same distribution, update detection window and repeat
        """
 
        # ------------------------Create csv for storing results----------------#
        if self.delete_csv==True:
            cols = ["Exp_name", "Window_size", "Sign_thres", "Detection_end", "Det_acc", "Ref_dist", 
                    "Det_dist", "KS_result", "Drift_signaled", "Real_drift", "Total_Training_time"]

            entries_df = pd.DataFrame(columns=cols)
            entries_df.to_csv(f"./results/{self.dataset.name}_{self.name}_results.csv", index=False)
        # ---------------------------------------------------------------------#

        # train
        self.train_model_gscv(window="reference", gscv=True)
        
        # initialize score
        self.experiment_metrics["scores"].append(self.evaluate_model_aggregate())
        self.update_detection_window()

        CALC_REF_RESPONSE = True
        entries_without_drift =  self.dataset.window_size # Assigning length of window to start
        
        while self.detection_window_end <= len(self.dataset.full_df):

            # log actual score on detection window
            self.experiment_metrics["scores"].append(self.evaluate_model_incremental())

            # get reference window response distribution with kfold and the detection response distribution
            if CALC_REF_RESPONSE:
                ref_response_dist, ref_ACC, ref_ACC_SD = self.get_reference_response_distribution()
            
            det_response_dist, det_ACC = self.get_detection_response_distribution()

            self.ref_distributions.append(ref_response_dist)
            self.det_distributions.append(det_response_dist)

            # compare distributions
            ks_result = self.perform_ks_test(
                dist1=ref_response_dist, dist2=det_response_dist
            )
            self.p_vals.append(ks_result.pvalue)

            if entries_without_drift > round(self.dataset.window_size/2,0):
                if ks_result[1] < self.significance_thresh:
                    significant_change = True
                    entries_without_drift = 0
                    self.drift_entries.append(self.detection_window_end)
                else:
                    significant_change = False
                    entries_without_drift +=1
            else:
                significant_change = False
                entries_without_drift +=1

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
                            self.detection_window_end, self.acc, ref_response_dist.tolist(),
                            det_response_dist.tolist(), ks_result.pvalue, significant_change,
                            significant_ACC_change, self.total_train_time]
            else:
                new_entry = [self.name, self.dataset.window_size, self.significance_thresh, 
                            self.detection_window_end, self.acc, "",
                            "", ks_result.pvalue, significant_change,
                            significant_ACC_change, self.total_train_time]

            
            with open(f"./results/{self.dataset.name}_{self.name}_results.csv", 'a', newline='') as f_object:
 
                # Pass this file object to csv.writer() and get a writer object
                writer_object = writer(f_object)
 
                # Pass the list as an argument into the writerow()
                writer_object.writerow(new_entry)
 
                # Close the file object
                f_object.close()
            #----------------------------------------------------------------------#
            
            if entries_without_drift == round(self.dataset.window_size/2,0):
                # reject null hyp, distributions are NOT the same --> retrain
                self.reference_window_start = self.detection_window_start
                self.reference_window_end = self.detection_window_end
                self.train_model_gscv(window="reference", gscv=True)

                # update detection window and reset score
                self.detection_window_start = self.detection_window_end
                self.detection_window_end = self.detection_window_end + self.dataset.window_size
                if self.detection_window_end > len(self.dataset.full_df):
                    self.detection_window_end = len(self.dataset.full_df)
                self.experiment_metrics["scores"].append(self.evaluate_model_aggregate())
                CALC_REF_RESPONSE = True
            else:
                CALC_REF_RESPONSE = False

            self.update_reference_window()
            self.update_detection_window()

        self.calculate_label_expense()
        self.calculate_train_expense()
        self.calculate_errors()
