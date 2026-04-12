import unittest
import os
import json
import shutil
import tensorflow as tf
import numpy as np
from unittest.mock import MagicMock, patch
from trainingPipeline import TrainingPipeline, EERCallback
from trainingPipeline import model_architecture_example,TrainingPipeline,EERCallback


def mock_architecture(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

class TestTrainingPipeline(unittest.TestCase):

    def setUp(self):
        """Set up a temporary environment for each test."""
        self.test_dir = "./test_temp"
        self.logs_path = os.path.join(self.test_dir, "logs")
        self.models_path = os.path.join(self.test_dir, "models")
        
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.logs_path)
        os.makedirs(self.models_path)

        # Create a dummy data map JSON
        self.data_map_path = os.path.join(self.test_dir, "map.json")
        with open(self.data_map_path, "w") as f:
            json.dump({"subjects": {}}, f)

        # Dummy Dataset Path
        self.dataset_path = os.path.join(self.test_dir, "dataset")
        os.makedirs(self.dataset_path)

        self.valid_config = {
            "data_params": {
                "dataset_path": self.dataset_path,
                "input_size": [224, 224, 3],
                "pixel_range": [0.0, 1.0],
                "batch_size": 32
            },
            "filtering_params": {
                "data_map_path": self.data_map_path,
                "keep_ratio": 0.8
            },
            "augmentation_params": {
                "rotation_range": 0.1,
                "brightness_range": [0.8, 1.2]
            },
            "model_params": {
                "architecture": mock_architecture
            },
            "training_params": {
                "early_stopping_patience": 5,
                "learning_rate": 0.001,
                "steps_per_epoch": 10,
                "initial_epochs": 1
            }
        }

    def tearDown(self):
        """Clean up the temporary directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization_and_path_creation(self):
        """Test if directories are created correctly during __init__."""
        tp = TrainingPipeline([self.valid_config], "test_exp", self.logs_path, self.models_path)
        self.assertTrue(os.path.exists(self.logs_path))
        self.assertTrue(os.path.exists(self.models_path))
        self.assertEqual(tp.master_record["metadata"]["experiment_id"], "test_exp")

    def test_val_configs_success(self):
        """Test validation with a correct configuration."""
        # Should not raise any error
        TrainingPipeline([self.valid_config], "test", self.logs_path, self.models_path)

    def test_val_configs_missing_key(self):
        """Test validation failure when a required key is missing."""
        invalid_cfg = self.valid_config.copy()
        del invalid_cfg["data_params"]
        with self.assertRaises(ValueError):
            TrainingPipeline([invalid_cfg], "test", self.logs_path, self.models_path)

    def test_sanitize_config(self):
        """Test if function references are correctly converted to strings for JSON."""
        tp = TrainingPipeline([self.valid_config], "test", self.logs_path, self.models_path)
        sanitized = tp._sanitize_config(self.valid_config)
        self.assertEqual(sanitized["model_params"]["architecture"], "mock_architecture")

    def test_init_model(self):
        """Test model dynamic initialization and compilation."""
        tp = TrainingPipeline([self.valid_config], "test", self.logs_path, self.models_path)
        model = tp.init_model(self.valid_config)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertTrue(model.compiled)
        # Check if parameters_count was injected into config
        self.assertIn("parameters_count", self.valid_config["model_params"])

    @patch('trainingPipeline.DataLoaderPipeline')
    def test_compute_single_pass_metrics(self, MockDLP):
        tp = TrainingPipeline([self.valid_config], "test", self.logs_path, self.models_path)
        model = tp.init_model(self.valid_config)
        
        # FIX: Provide BOTH labels (0 and 1) so the ROC curve can be calculated
        imgs = tf.random.uniform((2, 224, 224, 3))
        lbls = tf.constant([[1.0], [0.0]]) # One Real, One Spoof
        mock_ds = tf.data.Dataset.from_tensors((imgs, lbls))
        
        metrics = tp.compute_single_pass_metrics(mock_ds, model)
        self.assertIn("eer", metrics)
        self.assertIsInstance(metrics["eer"], float)

    def test_eer_callback_logic(self):
        # Ensure we have both classes (0 and 1) for a valid ROC curve
        imgs = tf.random.uniform((4, 224, 224, 3))
        lbls = tf.constant([[1.0], [0.0], [1.0], [0.0]])
        mock_ds = tf.data.Dataset.from_tensor_slices((imgs, lbls)).batch(2)
        
        callback = EERCallback(mock_ds)
        test_model = mock_architecture((224, 224, 3))
        callback.set_model(test_model)
        
        # Pre-populate logs to simulate Keras behavior
        logs = {"loss": 0.5, "accuracy": 0.5} 
        
        callback.on_epoch_end(0, logs)
        
        # Check for the key your pipeline actually monitors
        self.assertIn("val_eer", logs, "The callback failed to inject 'eer' into logs")
        self.assertIsInstance(logs["val_eer"], float)

if __name__ == "__main__":
    unittest.main()