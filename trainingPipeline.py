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
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.experiment_id =  f"{experiment_id}_{timestamp}"
        self.logs_output_path = logs_output_path
        self.models_output_path = models_output_path

        if not os.path.exists(logs_output_path):
            raise ValueError(f"Logs output path '{logs_output_path}' does not exist. Please create it before running the pipeline.")
        
        if not os.path.exists(models_output_path):
            raise ValueError(f"Models output path '{models_output_path}' does not exist. Please create it before running the pipeline.")
        
        
        self.record_file = os.path.join(self.logs_output_path, f"{self.experiment_id}_records.json")
        self.master_record = {
            "metadata": {"experiment_id": self.experiment_id, "status": "initialized", "notes": note},
            "records": []
        }

        self._fil_filtering_params()  # Pre-process filtering parameters for logging

        self.loss_tracker = tf.keras.metrics.Mean()
        self.acc_tracker = tf.keras.metrics.BinaryAccuracy()

    def val_configs(self):
        """Validates the provided configurations for completeness and correctness."""
        required_keys = ["data_params", "filtering_params", "model_params", "training_params"]
  
        for i, cfg in enumerate(self.configs):
            for key in required_keys:
                if key not in cfg:
                    raise ValueError(f"Config {i} is missing required key: '{key}'")
        
            dataset_path = cfg["data_params"]["dataset_path"] 
            if not os.path.exists(dataset_path):
                raise ValueError(f"Config {i} has an invalid dataset path: '{dataset_path}' does not exist.")
            
            if not "input_size" in cfg["data_params"] or len(cfg["data_params"]["input_size"]) != 3:
                raise ValueError(f"Config {i} has invalid 'input_size': must be a list of three values [height, width, channels].")
            
            if not "pixel_range" in cfg["data_params"] or len(cfg["data_params"]["pixel_range"]) != 2 or cfg["data_params"]["pixel_range"][0] >= cfg["data_params"]["pixel_range"][1]:
                raise ValueError(f"Config {i} has invalid 'pixel_range': must be a list of two values [min, max].")
        
            dm = cfg["filtering_params"]["data_map_path"]
            if not os.path.exists(dm):
                raise ValueError(f"Config {i} has an invalid data map path: '{dm}' does not exist.")
                
            if not "initial_epochs" in cfg["training_params"] or (cfg["training_params"]["initial_epochs"] <= 0):
                raise ValueError(f"Config {i} has invalid 'initial_epochs': must be > 0.")
            
            if not "batch_size" in cfg["data_params"] or cfg["data_params"]["batch_size"] <= 0:
                raise ValueError(f"Config {i} has invalid 'batch_size': must be a positive integer.")
            
            if not "learning_rate" in cfg["training_params"] or cfg["training_params"]["learning_rate"] <= 0:
                raise ValueError(f"Config {i} has invalid 'learning_rate': must be > 0.")
            
            if not "early_stopping_patience" in cfg["training_params"] or cfg["training_params"]["early_stopping_patience"] < 0:
                raise ValueError(f"Config {i} has invalid 'early_stopping_patience': must be >= 0.")
            
            model_init_function = cfg["model_params"].get("model_init_function", None)
            if not model_init_function or not callable(model_init_function):
                raise ValueError(f"Config {i} has an invalid  model_init_function: '{model_init_function}' is not callable.")
            
            
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

        arch_fn = m_params["model_init_function"]

        input_shape = tuple(d_params["input_size"]) 
        model = arch_fn(input_shape=input_shape)

      
        optimizer = tf.keras.optimizers.Adam(learning_rate=t_params["learning_rate"])
       
        # 4. Compile
        def eer(y_true, y_pred):
            return 0.0  # Placeholder, actual EER is computed in the callback and logs
        
        
        label_smoothing = m_params.get("label_smoothing", 0.0)
        isFocalLoss = m_params.get("isFocalLoss", 0)
        
        if isFocalLoss:
            alpha = m_params.get("focal_alpha", 0.25)
            gamma = m_params.get("focal_gamma", 2.0)
            apply_class_balancing = m_params.get("apply_class_balancing", 0) == 1
            loss = tf.keras.losses.BinaryFocalCrossentropy(
                label_smoothing=label_smoothing,
                alpha=alpha,
                gamma=gamma,
                apply_class_balancing=apply_class_balancing
            )
        else:
            loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy' , eer]
        )
        
        return model

    def run(self):
        print(f"Starting Suite: {self.experiment_id}")
        self.master_record["metadata"]["status"] = "running"

        all_subjects = [f"{i:02d}" for i in range(1, 51)]
        test_subs = all_subjects[20:]
        global_training_subjects = all_subjects[:20]
        global_training_subjects_shuffled = np.random.permutation(global_training_subjects)

           
        for i, cfg in enumerate(self.configs):
            
            validation_subjects_num = cfg["data_params"].get("validation_subjects_number", 0) 
            get_random_subjects_for_validation = cfg["data_params"].get("get_random_subjects_for_validation", 0) == 1
            
            if not validation_subjects_num : 
                train_subs = global_training_subjects
                val_subs = None
            else:
                if get_random_subjects_for_validation:
                    val_subs = global_training_subjects_shuffled[:validation_subjects_num]
                    train_subs = global_training_subjects_shuffled[validation_subjects_num:]
                else:
                    val_subs = global_training_subjects[:validation_subjects_num]
                    train_subs = global_training_subjects[validation_subjects_num:20]

            
            print(f"\n--- Running Sub-Experiment {i+1}/{len(self.configs)} ---")
            print("Configuration:")

            model = self.init_model(cfg)
            model_name = f"{self.experiment_id}_{i}_model_{model.name}.keras" 
            model_path = f"{self.models_output_path}/{model_name}"
            
            cfg['model_params']['parameters_count'] = model.count_params()
            cfg['model_params']['architecture'] = model.name
            
            entry = {
                "config": self._sanitize_config(cfg),
                "logs": {
                    "best_model_name": model_name
                }
            }
            self.master_record["records"].append(entry)
            self._save_state()
            print(cfg)
        
            
            # Build data pipelines
            dlp = DataLoaderPipeline(
                data_params=cfg['data_params'],
                filtering_params=cfg['filtering_params'], 
                augmentation_params=cfg['augmentation_params']
                )
            
            is_val_ds_balanced = cfg["data_params"].get("validation_dataset_baleance", 0) == 1
            
            train_ds = dlp.build_pipeline(train_subs, balanced=True, augment=True)

            
            callbacks = []

            # If validation subjects are specified, create validation dataset and add custom EER logging callback
            if validation_subjects_num :
                val_ds = dlp.build_pipeline(val_subs, balanced=is_val_ds_balanced, augment=False, shuffle=False)
                callbacks.append(ValidationEERLogger(val_ds))  # Custom callback for evaluation
                monitor_metric = 'val_eer'
            else:
                # No validation dataset, so we monitor training loss for callbacks
                monitor_metric = 'loss'
                

            # Early Stopping and ReduceLROnPlateau based on config
            early_stopping_patience = cfg["training_params"].get("early_stopping_patience", None)
            if early_stopping_patience is not None and early_stopping_patience >= 0:
                callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=monitor_metric, 
                                                    patience=early_stopping_patience, 
                                                    mode="min", 
                                                    restore_best_weights=True,
                                                    verbose=1))    

            reduce_on_plateau_patience = cfg["training_params"].get("ReduceLROnPlateau_patience", None)
            reduce_on_plateau_factor = cfg["training_params"].get("ReduceLROnPlateau_factor", None)
            if reduce_on_plateau_patience is not None and reduce_on_plateau_factor is not None:
                callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_metric,
                                                    factor=reduce_on_plateau_factor,
                                                    patience=reduce_on_plateau_patience,
                                                    mode='min',
                                                    verbose=1))

            
            # Always save the best model based on the monitored metric
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(model_path, 
                                                    save_best_only=True, 
                                                    monitor=monitor_metric,
                                                    mode='min',
                                                    verbose=1))

            history = model.fit(
                train_ds,
                # validation_data=val_ds,
                epochs=cfg["training_params"]["initial_epochs"],
                verbose=2,
                callbacks=callbacks
            )
            
            test_ds = dlp.build_pipeline(test_subs, balanced=False, augment=False, shuffle=False)           
            final_test_metrics = self.compute_metrics(test_ds, model)
            print(f"Final Test Metrics for Sub-Experiment {i+1}: {final_test_metrics}")
            
            # Update master record with training history and final test metrics
            self.master_record["records"][i]["logs"]["training_history"] = history.history
            self.master_record["records"][i]["logs"]["epochs"] = len(history.history['loss'])
            self.master_record["records"][i]["logs"]["final_test_metrics"] = final_test_metrics

        self.master_record["metadata"]["status"] = "completed"
        self._save_state()
        print(f"\nExperiment Suite '{self.experiment_id}' completed. Records saved to '{self.record_file}'.")


    def compute_metrics(self, ds, model):
        # 1. Execute the entire loop on GPU in one shot
        # This is where the CPU 'hands over' control to the GPU
        # Reset trackers before the validation pass
        self.loss_tracker.reset_state()
        self.acc_tracker.reset_state()
        
        # Run the engine
        y_true_gpu, y_pred_gpu = run_full_gpu_pass(
            ds, 
            model, 
            self.loss_tracker, 
            self.acc_tracker
        )

        # 2. Single Sync: Bring the final results to RAM
        y_true = y_true_gpu.numpy()
        y_scores = y_pred_gpu.numpy()
        
        # 3. Final EER calculation on CPU
        eer = compute_eer(y_true, y_scores)

        return {
            "loss": float(self.loss_tracker.result().numpy()),
            "accuracy": float(self.acc_tracker.result().numpy()),
            "eer": float(eer)
        }
    

def compute_eer(y_true, y_scores):
    """Calculates EER with safety fallback."""
    try:
        # Check if we have at least one sample of each class
        if len(np.unique(y_true)) < 2:
            return 0.5
            
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        if len(fpr) < 2:
            return 0.5
            
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer
    except Exception:
        return 0.5

@tf.function
def run_full_gpu_pass(ds, model, loss_tracker, acc_tracker):
    """
    Optimized GPU Engine. Prefetches data and runs inference in VRAM.
    """
    y_true_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    y_pred_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    
    idx = 0
    for images, labels in ds:
        preds = model(images, training=False)
        
        # Update metrics directly on GPU
        loss_tracker.update_state(model.loss(labels, preds))
        acc_tracker.update_state(labels, preds)
        
        # Store results
        y_true_array = y_true_array.write(idx, tf.cast(tf.reshape(labels, [-1]), tf.float32))
        y_pred_array = y_pred_array.write(idx, tf.reshape(preds, [-1]))
        idx += 1
        
    return y_true_array.concat(), y_pred_array.concat()


class ValidationEERLogger(tf.keras.callbacks.Callback):
    def __init__(self, val_ds):
        super(ValidationEERLogger, self).__init__()
        self.val_ds = val_ds
        # Trackers created once here to avoid variable creation error
        self.loss_tracker = tf.keras.metrics.Mean()
        self.acc_tracker = tf.keras.metrics.BinaryAccuracy()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.loss_tracker.reset_state()
        self.acc_tracker.reset_state()
        
        # Execute high-speed pass
        y_true_gpu, y_pred_gpu = run_full_gpu_pass(
            self.val_ds, self.model, self.loss_tracker, self.acc_tracker
        )

        y_true = y_true_gpu.numpy()
        y_scores = y_pred_gpu.numpy()
        eer_val = compute_eer(y_true, y_scores)

        # Update Keras logs so EarlyStopping/Checkpoint can see them
        logs["val_eer"] = float(eer_val)
        logs["val_loss"] = float(self.loss_tracker.result().numpy())
        logs["val_accuracy"] = float(self.acc_tracker.result().numpy())
        
        # Clean output for Kaggle/Terminal
        # print(f" - val_loss: {logs['val_loss']:.4f} - val_acc: {logs['val_accuracy']:.4f} - val_eer: {eer_val:.5f}")