# Import libraries
import pandas as pd
from csv import writer
import time
from collections import defaultdict
from river import metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from test_harness.experiments.base_class_experiment import Experiment

# Define class
class BaselineExperiment(Experiment):
    def __init__(self, model, dataset, param_grid=None,delete_csv=False):

        self.name = "1_Baseline"
        self.dataset = dataset
        self.model = model
        self.trained_model = None
        self.delete_csv = delete_csv

        self.reference_window_start = 0
        self.reference_window_end = dataset.window_size
        self.detection_window_start = dataset.window_size-1 # -1 so that first incremental is performed on real initial detection window
        self.detection_window_end = 2*dataset.window_size-1 # -1 so that first incremental is performed on real initial detection window
        
        self.experiment_metrics = defaultdict(list)
        self.incremental_metric = metrics.Accuracy()
        self.metric = "accuracy"
        self.param_grid = param_grid

        self.training_times = 0
        self.last_training = 0
        self.drift_signals = []
        self.drift_occurences = []
        self.acc = None
        self.total_train_time = 0

    def update_reference_window(self):
        """Advances reference window by 1 entry."""
        self.reference_window_start += 1
        self.reference_window_end += 1

    def update_detection_window(self):
        """Advances detection window by 1 entry."""
        self.detection_window_start += 1
        self.detection_window_end += 1

    def train_model_gscv(self, window="reference", gscv=False):
        """Trains model using grid search cross validation on specified window and updates 'trained_model' attribute."""

        self.training_times+=1 # increase by 1 every time the model is trained
        
        # Gather training data
        window_start = (
            self.reference_window_start
            if window == "reference"
            else self.detection_window_start
        )

        window_end = (
            self.reference_window_end
            if window == "reference"
            else self.detection_window_end
        )

        X_train, y_train = self.dataset.get_data_by_idx(window_start, window_end, split_labels=True)
        self.last_training = window_end
        
        # Create column transformer
        column_transformer = ColumnTransformer(
            [
                (
                    "floats",
                    StandardScaler(),
                    self.dataset.column_mapping["float_features"],
                ),
                (
                    "integers",
                    "passthrough",
                    self.dataset.column_mapping["int_features"],
                ),
            ]
        )
        

        # Instantiate training pipeline
        pipe = Pipeline(
            steps=[
                ("transformer", column_transformer),
                ("clf", self.model),
            ]
        )

        # To help ensure there is no overfit, perform GridsearchCV eachtime a new model is fit on a window
        if gscv:
            if self.param_grid is None:
                raise AttributeError("Training with GSCV, but no param_grid provided.")

            gs = GridSearchCV(
                estimator=pipe,
                param_grid=self.param_grid,
                cv=5,
                scoring=self.metric,
                n_jobs=-1,
                refit=True,
                return_train_score=True,
            )

            gs.fit(X_train, y_train)

            self.trained_model = gs.best_estimator_
            train_time = gs.refit_time_
            eval_score = gs.cv_results_["mean_train_score"][gs.best_index_]
            gscv_test_score = gs.best_score_

            # Update self.model to hold best parameters
            self.model = self.trained_model.get_params()["clf"]

        else:

            # Fit model
            start_time = time.time()
            self.trained_model = pipe.fit(X_train, y_train)
            end_time = time.time()
            train_time = end_time - start_time

            # Evaluate training
            eval_score = self.evaluate_model_aggregate(window=window)
            gscv_test_score = None

        # Save training metrics
        metrics = {
            "times_trained": self.training_times,
            "training_spot": self.last_training,
            "num_train_examples": len(y_train),
            "train_time": train_time,
            "eval_score": eval_score,
            "gscv_test_score": gscv_test_score,
        }
        self.total_train_time = self.total_train_time + train_time
        self.experiment_metrics["training"].append(metrics)

    def evaluate_model_aggregate(self, window="detection"):
        """
        Evaluates the saved model on all data in the specified window

        Args:
            window (str) - specifies full window to evaluate on (detection/reference)

        Returns:
            tuple composed of:
            - idx_end (int): index of the last entry in the window
            - acc_new (float): aggregate score on selected window
        """

        # Gather evaluation data
        window_start = (
            self.reference_window_start
            if window == "reference"
            else self.detection_window_start
        )

        window_end = (
            self.reference_window_end
            if window == "reference"
            else self.detection_window_end
        )

        X_test, y_test = self.dataset.get_data_by_idx(window_start, window_end, split_labels=True)

        # Evaluate and update accuracy score
        acc_new = self.trained_model.score(X_test, y_test)

        self.acc = acc_new
        idx_end = window_end-1

        return (idx_end, acc_new)

    def evaluate_model_incremental(self, window="detection"):
        """
        Evaluates the saved model in an incremental way by updating the acc score based on the latest
        entry in the specified window.

        Args:
            window (str) - specifies full window to evaluate on (detection/reference)

        Returns:
            tuple composed of:
            - idx_end (int): index of the new entry considered for the incremental score
            - acc_new (float): incremental score on selected window
        """

        # Gather evaluation data
        window_start = (
            self.reference_window_start
            if window == "reference"
            else self.detection_window_start
        )

        window_end = (
            self.reference_window_end
            if window == "reference"
            else self.detection_window_end
        )

        # Obtain score for last entry in the window
        idx_end = window_end-1
        X_new, y_new = self.dataset.get_data_by_idx(idx_end, window_end, split_labels=True)
        y_pred_new = self.trained_model.predict(X_new)
        acc_new = round(1/self.dataset.window_size,4) if int(y_pred_new)==int(y_new) else 0

        # Obtain score from first entry in previous window
        idx_st = window_start-1
        X_remove, y_remove = self.dataset.get_data_by_idx(idx_st, window_start, split_labels=True)
        y_pred_remove = self.trained_model.predict(X_remove)
        acc_remove = round(1/self.dataset.window_size,4) if int(y_pred_remove)==int(y_remove) else 0

        # Calculate new accuracy and update accuracy score
        acc_new = self.acc - acc_remove + acc_new
        self.acc = acc_new

        return (idx_end, acc_new)
    

    def calculate_label_expense(self):
        """A postprocessing step to aggregate and save label expense metrics"""

        num_labels_requested = sum(
            [
                train_run["num_train_examples"]
                for train_run in self.experiment_metrics["training"]
            ]
        )
        percent_total_labels = round(
            num_labels_requested/len(self.dataset.full_df), 4
        )

        label_metrics = {
            "num_labels_requested": num_labels_requested,
            "percent_total_labels": percent_total_labels,
        }

        self.experiment_metrics["label_expense"] = label_metrics


    def calculate_train_expense(self):
        """A postprocessing step to aggregate and save training expense metrics"""

        total_train_time = round(
            sum(
                [
                    train_run["train_time"]
                    for train_run in self.experiment_metrics["training"]
                ]
            ),
            2,
        )

        self.experiment_metrics["total_train_time"] = total_train_time


    def run(self):
        """Baseline Experiment
         
        This experiment trains a model on the initial reference window and then evaluates
        incrementally on each new entry with no retraining. It produces the least
        accurate scenario and should incur minimal label cost at the expense of accuracy.

        Logic flow:
            - Train on initial reference window
            - Evaluate on aggregated detection window
            - Update detection window
            - Evaluate incrementally on detection window
            - Repeat steps 3 and 4 until finished
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

            # --------------------------------- Store results in csv ----------------------------------#
            new_entry = [self.name, self.dataset.window_size, 
                         self.detection_window_end, self.acc, 
                         "False", "False", self.total_train_time]
            
            with open(f"./results/{self.dataset.name}_{self.name}_results.csv", 'a', newline='') as f_object:
                # Pass this file object to csv.writer() and get a writer object
                writer_object = writer(f_object)
 
                # Pass the list as an argument into the writerow()
                writer_object.writerow(new_entry)
 
                # Close the file object
                f_object.close()
            #-----------------------------------------------------------------------------------------#

            #self.update_reference_window() # commented because it is useless for Baseline
            self.update_detection_window()

        self.calculate_label_expense()
        self.calculate_train_expense()
