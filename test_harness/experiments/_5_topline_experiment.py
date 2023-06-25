# Import libraries
import pandas as pd
from csv import writer
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from test_harness.experiments._1_baseline_experiment import BaselineExperiment

# Define class
class ToplineExperiment(BaselineExperiment):
    def __init__(self, model, dataset, k, param_grid=None,delete_csv=False):

        super().__init__(model, dataset, param_grid)
        self.name = "5_Topline"
        self.k = k

        self.drift_entries = []
        self.acc_det_aux = 100 # random number out of range so that first diff is significant
        self.entry_det_aux = 0 # 0 so that first diff is significant


    @staticmethod
    def make_kfold_predictions(X, y, model, dataset, k):
        """A KFold version of LeaveOneOut predictions.

        When k == len(X), this is equivalent to LeaveOneOut: expensive, but robust. Reducing k
        saves computation, but reduces robustness of model.

        Args:
            X (pd.Dataframe) - features in evaluation window
            y (pd.Series) - labels in evaluation window
            k (int) - number of folds

        Returns:
            - split_ACCs (np.array): an array of accuracies for each split
        """

        splitter = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)

        split_ACCs = np.array([])

        for train_indicies, test_indicies in splitter.split(X, y):

            # Create column transformer
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

            # Instantiate training pipeline
            pipe = Pipeline(
                steps=[
                    ("scaler", column_transformer),
                    ("clf", model),
                ]
            )

            # Fit pipeline
            pipe.fit(X.iloc[train_indicies], y.iloc[train_indicies])

            # Get accuracy for split
            split_ACC = pipe.score(X.iloc[test_indicies], y.iloc[test_indicies])
            split_ACCs = np.append(split_ACCs, split_ACC)

        return split_ACCs


    def get_reference_response_distribution(self):
        """A method to obtain the accuracy of the reference window"""

        # Get data in reference window
        window_start = self.reference_window_start
        window_end = self.reference_window_end
        X_train, y_train = self.dataset.get_data_by_idx(window_start, window_end, split_labels=True)

        # Perform kfoldsplits to get predictions
        split_ACCs = self.make_kfold_predictions(
            X_train, y_train, self.model, self.dataset, self.k
        )

        # Obtain accuracy of reference window as mean of accuracies of the splits; obtain standard deviation
        ref_ACC = np.mean(split_ACCs)
        ref_ACC_SD = np.std(split_ACCs)

        return ref_ACC, ref_ACC_SD


    def run(self):
        """Topline Experiment 
        
        This experiment retrains the model on each incremental window, serving as the most greedy possible scenario.
        It should incur high label cost.

        Logic flow:
            - Train on initial window
            - Evaluate on aggregated detection window
            - Update reference window and retrain
            - Repeat until finished
        """
        
        # ---------------------------------- Create csv for storing results ----------------------------------#
        if self.delete_csv==True:
            cols = ["Exp_name", "Window_size", "Detection_end", "Det_acc", 
                    "Drift_signaled", "Real_drift", "Total_Training_time"]

            entries_df = pd.DataFrame(columns=cols)
            entries_df.to_csv(f"./results/{self.dataset.name}_{self.name}_results.csv", index=False)
        # ----------------------------------------------------------------------------------------------------#

        # Perform initial training and aggregate evaluation
        self.train_model_gscv(window="reference", gscv=True)
        self.experiment_metrics["scores"].append(self.evaluate_model_aggregate())
        
        self.update_detection_window()

        while self.detection_window_end <= len(self.dataset.full_df):

            # Log incremental accuracy score on detection window
            self.experiment_metrics["scores"].append(self.evaluate_model_incremental())

            # Get accuracy of reference and detection windows
            ref_ACC, ref_ACC_SD = self.get_reference_response_distribution()
            det_ACC = self.evaluate_model_aggregate(window="detection")[1]

            # Compare accuracies to see if difference is significant
            # This part is useful if there is no control over real drift, results will be ignored for this thesis
            delta_ACC = np.absolute(det_ACC - ref_ACC)
            delta_ACC_det = np.absolute(det_ACC - self.acc_det_aux)
            diff_entries = self.detection_window_end - self.entry_det_aux
            threshold_ACC = 3 * ref_ACC_SD  # Considering outside 3 SD significant
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

            self.drift_signals.append(True)  # Every iteration signals a new drift
            self.drift_entries.append(self.detection_window_end)

            # --------------------------------- Store results in csv ----------------------------------#
            new_entry = [self.name, self.dataset.window_size, 
                         self.detection_window_end, self.acc, 
                         "True", significant_ACC_change, self.total_train_time]
            
            with open(f"./results/{self.dataset.name}_{self.name}_results.csv", 'a', newline='') as f_object:
                # Pass this file object to csv.writer() and get a writer object
                writer_object = writer(f_object)
 
                # Pass the list as an argument into the writerow()
                writer_object.writerow(new_entry)
 
                # Close the file object
                f_object.close()
            # ------------------------------------------------------------------------------------------#

            # Retrain, update reference and detection windows
            self.train_model_gscv(window="detection", gscv=True)
            self.reference_window_start = self.detection_window_start
            self.reference_window_end = self.detection_window_end
            self.detection_window_start = self.detection_window_end
            self.detection_window_end = self.detection_window_end + self.dataset.window_size

        self.calculate_label_expense()
        self.calculate_train_expense()
