import time
import json
import os
import tensorflow as tf
from DataLoaderPipeline import DataLoaderPipeline
import tensorflow as tf
import os
import numpy as np

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d



class TrainingPipeline:
    """
    A comprehensive training pipeline that:
    - Validates experiment configurations for completeness and correctness.
    - Dynamically initializes and compiles models based on provided architecture functions.
    - Implements a custom callback to compute EER at the end of each epoch using a single pass over the validation data.
    - Computes final test metrics (Loss, Accuracy, EER) in a single pass after training completion.
    - Maintains a master record of all experiments, configurations, and results in a JSON file for easy analysis and reproducibility.
    - Designed for flexibility to accommodate various model architectures and training configurations with minimal code changes.

    """
    def __init__(self, configs, experiment_id, logs_output_path="/kaggle/working/logs", models_output_path="/kaggle/working/models", note=""):
        """
        Initializes the training pipeline with experiment configurations and sets up the output structure.
        Args:
            configs (list of dict): A list of experiment configurations, each containing data, filtering, augmentation, model, and training parameters.
            experiment_id (str): A unique identifier for the experiment suite (e.g., "CASIA_FASD_v3_SimpleCNN").
            logs_output_path (str): The directory where training logs will be saved.
            models_output_path (str): The directory where model weights will be saved.
            note (str): Optional notes about the experiment to be included in the master record metadata.
        """
        self.configs = configs
        self.val_configs()  # Validate configurations before proceeding
        
        self.experiment_id = experiment_id
        self.logs_output_path = logs_output_path
        self.models_output_path = models_output_path

        if not os.path.exists(logs_output_path):
            raise ValueError(f"Logs output path '{logs_output_path}' does not exist. Please create it before running the pipeline.")
        
        if not os.path.exists(models_output_path):
            raise ValueError(f"Models output path '{models_output_path}' does not exist. Please create it before running the pipeline.")
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        self.record_file = os.path.join(self.logs_output_path, f"{experiment_id}_{timestamp}_records.json")
        self.master_record = {
            "metadata": {"experiment_id": experiment_id, "status": "initialized", "notes": note},
            "records": []
        }

        self._fil_filtering_params()  # Pre-process filtering parameters for logging

    def val_configs(self):
        """Validates the provided configurations for completeness and correctness."""
        required_keys = ["data_params", "filtering_params", "model_params", "training_params"]
        for i, cfg in enumerate(self.configs):
            for key in required_keys:
                if key not in cfg:
                    raise ValueError(f"Config {i} is missing required key: '{key}'")
        
        models = [cfg["model_params"]["architecture"] for cfg in self.configs]
        for i, model_fn in enumerate(models):
            if not callable(model_fn):
                raise ValueError(f"Config {i} has an invalid model architecture: '{model_fn}' is not callable.")

        dataset_paths = [cfg["data_params"]["dataset_path"] for cfg in self.configs]
        for i, path in enumerate(dataset_paths):
            if not os.path.exists(path):
                raise ValueError(f"Config {i} has an invalid dataset path: '{path}' does not exist.")
            
        data_maps = [cfg["filtering_params"]["data_map_path"] for cfg in self.configs]
        for i, dm in enumerate(data_maps):
            if not os.path.exists(dm):
                raise ValueError(f"Config {i} has an invalid data map path: '{dm}' does not exist.")
        
        for i, cfg in enumerate(self.configs):
            if not "initial_epochs" in cfg["training_params"] or (cfg["training_params"]["initial_epochs"] <= 0):
                raise ValueError(f"Config {i} has invalid 'initial_epochs': must be > 0.")
            
            if not "steps_per_epoch" in cfg["training_params"] or (cfg["training_params"]["steps_per_epoch"] <= 0):
                raise ValueError(f"Config {i} has invalid 'steps_per_epoch': must be > 0.")
            
            if not "pixel_range" in cfg["data_params"] or len(cfg["data_params"]["pixel_range"]) != 2:
                raise ValueError(f"Config {i} has invalid 'pixel_range': must be a list of two values [min, max].")
            
            if not "input_size" in cfg["data_params"] or len(cfg["data_params"]["input_size"]) != 3:
                raise ValueError(f"Config {i} has invalid 'input_size': must be a list of three values [height, width, channels].")
            
            if not "batch_size" in cfg["data_params"] or cfg["data_params"]["batch_size"] <= 0:
                raise ValueError(f"Config {i} has invalid 'batch_size': must be a positive integer.")
            
        print("All configurations validated successfully.")

    def _save_state(self):
        with open(self.record_file, "w") as f:
            json.dump(self.master_record, f, indent=4, default=lambda x: str(x))
   
    def _sanitize_config(self, cfg):
        """Converts function references to strings for JSON saving."""
        clean = json.loads(json.dumps(cfg, default=lambda x: x.__name__ if hasattr(x, '__name__') else str(x)))
        return clean
    
    def _fil_filtering_params(self):
        """Extracts and formats filtering parameters for logging."""
        for i, cfg in enumerate(self.configs):
            filtering_params = cfg['filtering_params']
            data_map_path = filtering_params['data_map_path']
            data_map_metadata = json.load(open(data_map_path, 'r'))['metadata']
            keep_ratio = data_map_metadata.get('keep_ratio', 'N/A')
            filter_function = data_map_metadata.get('filter_function', 'N/A')
            self.configs[i]["filtering_params"]["keep_ratio"] = keep_ratio
            self.configs[i]["filtering_params"]["filter_function"] = filter_function
            

    def init_model(self, cfg):
        """
        Dynamically initializes and compiles the model based on the experiment config.
        """
        m_params = cfg["model_params"]
        t_params = cfg["training_params"]
        d_params = cfg["data_params"]

        arch_fn = m_params["architecture"]

        input_shape = tuple(d_params["input_size"]) 
        model = arch_fn(input_shape=input_shape)

      
        optimizer = tf.keras.optimizers.Adam(learning_rate=t_params["learning_rate"])
       
        # 4. Compile
        def eer(y_true, y_pred):
            return 0.0
        
        model.compile(
            optimizer=optimizer,
            loss= tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy' , eer]
        )
        
        return model

    def run(self):
        print(f"Starting Suite: {self.experiment_id}")
        self.master_record["metadata"]["status"] = "running"

        all_subjects = [f"{i:02d}" for i in range(1, 51)]
        train_subs = all_subjects[:20]
        val_subs = all_subjects[20:30]
        test_subs = all_subjects[30:] 
           
        for i, cfg in enumerate(self.configs):
            print(f"\n--- Running Sub-Experiment {i+1}/{len(self.configs)} ---")
            print("Configuration:")
            print(cfg)

            model_name = f"{self.experiment_id}__model_{i}.keras"
            
            model_path = f"{self.models_output_path}/{model_name}"
            entry = {
                "config": self._sanitize_config(cfg),
                "logs": {
                    "best_model_name": model_name
                }
            }
            self.master_record["records"].append(entry)
            self._save_state()
            
            
            dlp = DataLoaderPipeline(
                data_params=cfg['data_params'],
                filtering_params=cfg['filtering_params'], 
                augmentation_params=cfg['augmentation_params']
                )
            
            train_ds = dlp.build_pipeline(train_subs, balanced=True, augment=True)
            val_ds = dlp.build_pipeline(val_subs, balanced=False , augment=False, shuffle=False)
            test_ds = dlp.build_pipeline(test_subs, balanced=False, augment=False, shuffle=False)           
            

            model = self.init_model(cfg)
            callbacks = [
                EERCallback(val_ds),  # Custom callback for single-pass EER evaluation
                tf.keras.callbacks.EarlyStopping(monitor="eer", 
                                                patience=cfg["training_params"]["early_stopping_patience"], 
                                                mode="min", 
                                                restore_best_weights=True), 
                
                tf.keras.callbacks.ModelCheckpoint(model_path, 
                                                   save_best_only=True, 
                                                   monitor='eer',
                                                   mode='min')
                                                   
            ]
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=cfg["training_params"]["initial_epochs"],
                steps_per_epoch=cfg["training_params"]["steps_per_epoch"],
                verbose=2,
                callbacks=callbacks
            )
            
            final_test_metrics = self.compute_single_pass_metrics(test_ds, model)
            
            self.master_record["records"][i]["model_params"]["parameters_count"] = model.count_params()
            self.master_record["records"][i]["logs"]["training_history"] = history.history
            self.master_record["records"][i]["logs"]["epochs"] = len(history.history['loss'])
            self.master_record["records"][i]["logs"]["final_test_metrics"] = final_test_metrics

        self.master_record["metadata"]["status"] = "completed"
        self._save_state()
        print(f"\nExperiment Suite '{self.experiment_id}' completed. Records saved to '{self.record_file}'.")


    def compute_single_pass_metrics(self, ds, model):

        """
        Computes Loss, Accuracy, and EER in exactly one pass over the data.
        """
        # Use TF's built-in metrics for speed (stays on GPU)
        loss_tracker = tf.keras.metrics.Mean()
        acc_tracker = tf.keras.metrics.BinaryAccuracy()

        y_true_list = []
        y_pred_list = []

        # Get the loss function from the model (e.g., BinaryCrossentropy)
        loss_fn = model.loss

        # SINGLE PASS: Inference + Loss + Accuracy + Label Extraction
        for images, labels in ds:
            # 1. Forward Pass
            preds = model(images, training=False)

            # 2. Update Loss and Acc (Vectorized on GPU)
            batch_loss = loss_fn(labels, preds)
            loss_tracker.update_state(batch_loss)
            acc_tracker.update_state(labels, preds)

            # 3. Store for EER (Move to CPU/RAM)
            y_true_list.append(labels.numpy())
            y_pred_list.append(preds.numpy())

        # Finalize Loss and Acc
        final_loss = loss_tracker.result().numpy()
        final_acc = acc_tracker.result().numpy()

        # Flatten the results for EER
        y_true = np.concatenate(y_true_list).ravel()
        y_scores = np.concatenate(y_pred_list).ravel()

        # 4. EER Calculation (Sklearn is CPU-based)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        frr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute(fpr - frr))]

        return {
            "loss": float(final_loss),
            "accuracy": float(final_acc),
            "eer": float(eer)
        }
    

class EERCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds):
        super(EERCallback, self).__init__()
        self.val_ds = val_ds

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        y_true_list = []
        y_pred_list = []

        # Single pass to gather predictions and ground truth
        for images, labels in self.val_ds:
            preds = self.model(images, training=False)
            
            y_true_list.append(labels.numpy())
            y_pred_list.append(preds.numpy())

        # Flatten arrays
        y_true = np.concatenate(y_true_list).ravel()
        y_scores = np.concatenate(y_pred_list).ravel()

        # EER Calculation logic
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        frr = 1 - tpr
        # Find the point where FPR is closest to FRR
        eer = fpr[np.nanargmin(np.absolute(fpr - frr))]

        # Feed back to Keras
        logs["val_eer"] = float(eer)

