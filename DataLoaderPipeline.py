import os  # File and path utilities.
import json  # JSON loading for dataset metadata.
import tensorflow as tf  # TensorFlow data pipeline, image decoding, and batching.
import matplotlib.pyplot as plt  # Visualization for sample inspection.
from numpy import clip # Clipping function for visualization.
from numpy import min as npmin
from numpy import max as npmax
class DataLoaderPipeline:
    """
    Data loading and preprocessing pipeline for a face anti-spoofing dataset.

    Expected dataset structure:
        data_path/
            subject_id/
                real/
                    image1.jpg
                    image2.jpg
                spoof/
                    image3.jpg

    The JSON map is expected to contain:
        {
            "subjects": {
                "subject_id": {
                    "real": ["img1.jpg", "img2.jpg"],
                    "spoof": ["img3.jpg"]
                }
            },
            "metadata": {...}
        }
    """

    def __init__(self, data_map_path, data_path, pix_range=(0.0, 255.0), img_size=(224, 224), batch_size=32):
        """
        Initialize the pipeline configuration.

        Args:
            data_map_path: Path to the JSON file containing the subject map.
            data_path: Root folder of the dataset.
            pix_range: tuple of floats, Expected pixel range after preprocessing.
            img_size: Target image size.
            batch_size: Number of samples per batch.
        """
        self.img_size = img_size  # Final image size used by the model.
        self.batch_size = batch_size  # Batch size for training/evaluation.
        self.AUTOTUNE = tf.data.AUTOTUNE  # TensorFlow performance optimization.

        # Normalize the dataset path to avoid trailing slash issues.
        self.data_path = data_path if data_path else None

        # JSON file path that stores the dataset index/map.
        self.data_map_path = data_map_path

        # Expected pixel range, used by clipping and auditing.
        self.pix_range = pix_range

        # Loaded subject map: subject_id -> {label_name: [file_names]}
        self.subject_data = {}

        # Optional metadata loaded from JSON.
        self.metadata = {}

        self.load_from_json()

    # -------------------------------------------------------------------------
    # 1) DATA LOADING
    # -------------------------------------------------------------------------
    def load_from_json(self):
        """
        Load the subject map and metadata from the JSON file.

        This overwrites self.subject_data and self.metadata.
        """
        # Check whether the JSON file exists before trying to load it.
        if not os.path.exists(self.data_map_path):
            print(f"\033[1;31mError: JSON file {self.data_map_path} not found.\033[0m")
            return

        # Read and parse the JSON file.
        with open(self.data_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load data safely using defaults if keys are missing.
        self.subject_data = data.get("subjects", {})
        self.metadata = data.get("metadata", {})

        # Report successful loading.
        print(f"\033[1;32mSuccessfully loaded {len(self.subject_data)} subjects from JSON.\033[0m")

    # -------------------------------------------------------------------------
    # 2) PATH COLLECTION
    # -------------------------------------------------------------------------
    def _get_paths_by_subjects(self, subject_list):
        """
        Collect full file paths for the provided subject IDs.

        Args:
            subject_list: List of subject IDs to include.

        Returns:
            A tuple of (real_paths, spoof_paths).
        """
        real_paths = []  # Holds paths to real samples.
        spoof_paths = []  # Holds paths to spoof samples.

        for sub in subject_list:
            # Skip unknown subjects instead of raising a KeyError.
            if sub not in self.subject_data:
                continue

            # Iterate over the labels stored for this subject.
            for label_key in self.subject_data[sub]:
                # Build the folder path for this subject/label combination.
                folder = os.path.join(self.data_path, sub, label_key)
                

                # Convert file names into full paths.
                paths = [os.path.join(folder, f) for f in self.subject_data[sub][label_key]]

                # Treat common spoof labels as spoof; everything else as real.
                if label_key in ["spoof", "attack", "fake"]:
                    spoof_paths.extend(paths)
                else:
                    real_paths.extend(paths)

        return real_paths, spoof_paths

    # -------------------------------------------------------------------------
    # 3) PREPROCESSING AND AUGMENTATION
    # -------------------------------------------------------------------------
    def _base_preprocess(self, path, label):
        """
        Read an image from disk and resize it to the configured input size.

        Args:
            path: File path to the image.
            label: Class label associated with the image.

        Returns:
            A tuple of (image_tensor, label).
        """
        # Read the raw file bytes from disk.
        img = tf.io.read_file(path)

        # Decode JPEG images as 3-channel RGB.
        img = tf.image.decode_jpeg(img, channels=3)

        # Resize to the model input shape.
        img = tf.image.resize(img, self.img_size)

       # Assuming self.pix_range is a tuple like (0.0, 1.0) or (-1.0, 1.0)
        lower, upper = self.pix_range

        # 1. Cast to float first to avoid integer division errors
        img = tf.cast(img, tf.float32)

        # 2. General Linear Scaling Formula
        # Maps [0, 255] -> [lower, upper]
        img = lower + (img * (upper - lower) / 255.0)

        return img, label

    def _apply_augmentation(self, img, label):
        """
        Apply random augmentation to an image.

        Augmentations:
            - random horizontal flip
            - random brightness adjustment
            - random crop and resize back to original size
            - clipping to the configured pixel range

        Returns:
            A tuple of (augmented_image, label).
        """
        # Randomly mirror the image horizontally.
        img = tf.image.random_flip_left_right(img)

        # Randomly change brightness slightly.
        img = tf.image.random_brightness(img, max_delta=0.1)

        # Store the current image shape so we can restore it after cropping.
        original_shape = tf.shape(img)

        # Sample a crop ratio between 0.9 and 1.0 of the original dimensions.
        crop_size = tf.cast(
            tf.cast(original_shape[:2], tf.float32) * tf.random.uniform([], 0.9, 1.0),
            tf.int32,
        )

        # Randomly crop the image using the computed crop size.
        img = tf.image.random_crop(img, size=[crop_size[0], crop_size[1], 3])

        # Resize back to the original shape to preserve the pipeline shape.
        img = tf.image.resize(img, [original_shape[0], original_shape[1]])

        # Clip values to the configured pixel range to avoid out-of-range pixels.
        img = tf.clip_by_value(img, self.pix_range[0], self.pix_range[1])

        return img, label

    # -------------------------------------------------------------------------
    # 4) DATASET GENERATION
    # -------------------------------------------------------------------------
    def build_pipeline(self, subject_ids, balanced=True, augment=False, shuffle=True):
        """
        Build a tf.data pipeline for the selected subjects.

        Args:
            subject_ids: List of subject IDs to include.
            balanced: If True, sample real and spoof data equally.
            augment: If True, apply augmentation after preprocessing.
            shuffle: If True, shuffle each class-specific dataset.

        Returns:
            A batched and prefetched tf.data.Dataset.
        """
        # Collect all file paths for the requested subjects.
        real_p, spoof_p = self._get_paths_by_subjects(subject_ids)

        # Create labels for each class.
        real_labels = [1] * len(real_p)   # Real samples get label 1.
        spoof_labels = [0] * len(spoof_p)  # Spoof samples get label 0.

        # Build class-specific datasets from file paths and labels.
        real_ds = tf.data.Dataset.from_tensor_slices((real_p, real_labels))
        spoof_ds = tf.data.Dataset.from_tensor_slices((spoof_p, spoof_labels))

        # Shuffle each class stream independently when requested.
        if shuffle:
            real_ds = real_ds.shuffle(len(real_p) + 1)
            spoof_ds = spoof_ds.shuffle(len(spoof_p) + 1)

        # Combine datasets either with balanced sampling or simple concatenation.
        if balanced:
            # Repeat both datasets indefinitely and sample them with equal weights.
            ds = tf.data.Dataset.sample_from_datasets(
                [real_ds.repeat(), spoof_ds.repeat()],
                weights=[0.5, 0.5],
            )
        else:
            # Concatenate datasets for validation/test-style iteration.
            ds = real_ds.concatenate(spoof_ds)

        # Read files, decode images, resize, and normalize.
        ds = ds.map(self._base_preprocess, num_parallel_calls=self.AUTOTUNE)

        # Apply augmentation only when explicitly requested.
        if augment:
            ds = ds.map(self._apply_augmentation, num_parallel_calls=self.AUTOTUNE)

        # Batch and prefetch for performance.
        return ds.batch(self.batch_size).prefetch(self.AUTOTUNE)

    # -------------------------------------------------------------------------
    # 5) VISUALIZATION
    # -------------------------------------------------------------------------    
    def display_sample(self, ds, title="Dataset Sample"):
        """
        Display one batch of images, automatically handling normalization ranges.
        """
        for images, labels in ds.take(1):
            plt.figure(figsize=(10, 10))
            plt.suptitle(title, fontsize=16)

            # Determine number of images to show
            num_to_show = min(9, images.shape[0])

            for i in range(num_to_show):
                plt.subplot(3, 3, i + 1)
                
                img_to_show = images[i].numpy()

                # --- DYNAMIC RANGE HANDLING ---
                # Matplotlib's imshow expects [0, 1] for floats or [0, 255] for uint8.
                # If your data is [-1, 1], we must shift it back to [0, 1].
                if npmin(img_to_show) < 0:
                    img_to_show = (img_to_show + 1.0) / 2.0
                # If your data is [0, 255] as float, rescale it.
                elif npmax(img_to_show) > 1.01:
                    img_to_show = img_to_show / 255.0
                
                plt.imshow(clip(img_to_show, 0, 1))

                # Display label and color code (Green for Real, Red for Spoof)
                is_real = int(labels[i]) == 1
                label_text = "Real" if is_real else "Spoof"
                color = "green" if is_real else "red"
                
                plt.title(label_text, color=color, fontweight='bold')
                plt.axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # -------------------------------------------------------------------------
    # 6) AUDIT / VALIDATION
    # -------------------------------------------------------------------------
    def audit_dataset(self, dataset, batchs=20):
        """
        Inspect a dataset for class balance and pixel range.

        Args:
            dataset: A tf.data.Dataset yielding (images, labels).
            batchs: Number of batches to inspect.
        """
        count_real = 0  # Number of real samples found.
        count_spoof = 0  # Number of spoof samples found.
        total_images = 0  # Total number of samples processed.

        # Initialize pixel range tracking.
        min_pixel = self.pix_range[1]  # Start from the expected maximum.
        max_pixel = self.pix_range[0]  # Start from the expected minimum.

        print(f"Auditing up to {batchs} batches...")

        # Iterate through the requested number of batches.
        for _, (images, labels) in enumerate(dataset.take(batchs)):
            batch_labels = labels.numpy()

            # Count labels in the current batch.
            count_real += int((batch_labels == 1).sum())
            count_spoof += int((batch_labels == 0).sum())
            total_images += len(batch_labels)

            # Update observed pixel min/max across the batch.
            min_pixel = min(min_pixel, float(tf.reduce_min(images).numpy()))
            max_pixel = max(max_pixel, float(tf.reduce_max(images).numpy()))

        # Avoid division by zero if the dataset is empty.
        if total_images == 0:
            print("\033[1;31mWarning: No samples found in the audited dataset.\033[0m")
            return

        # Compute class percentages.
        real_pct = (count_real / total_images) * 100
        spoof_pct = (count_spoof / total_images) * 100

        # Print a formatted report.
        print("\n" + "=" * 30)
        print("POST-CREATION DATASET AUDIT")
        print("=" * 30)
        print(f"Total Samples Audited: {total_images}")
        print(f"Real Samples:  {count_real} ({real_pct:.1f}%)")
        print(f"Spoof Samples: {count_spoof} ({spoof_pct:.1f}%)")
        print("-" * 30)
        print(f"Pixel Range: [{min_pixel:.2f} to {max_pixel:.2f}]")

        # Confirm that values stay inside the expected range.
        if self.pix_range[0] <= min_pixel and max_pixel <= self.pix_range[1]:
            print("\033[1;32mStatus: Normalization OK\033[0m")
        else:
            print("\033[1;31mWarning: Normalization Out of Range!\033[0m")

        # Check whether the dataset is approximately balanced.
        if abs(real_pct - 50) > 5:
            print("\033[1;31mWarning: Dataset is Unbalanced!\033[0m")
        else:
            print("\033[1;32mStatus: Balance OK (~50/50)\033[0m")

        print("=" * 30 + "\n")


# usage example:
"""
DATA_PATH = "./data/CASIA_FASD_V3/DATA"
DATA_MAP_PATH = "./CASIA_FASD_V3_10percent.json"
all_subjects = [f"{i:02d}" for i in range(1, 51)]  # Generate subject IDs from '01' to '50'.
train_subs = all_subjects[:20]
val_subs = all_subjects[20:25] 
test_subs = all_subjects[20:] 

dlp = DataLoaderPipeline(data_map_path=DATA_MAP_PATH, data_path=DATA_PATH, pix_range=(-1.0,1.0), img_size=(224,224), batch_size=32)
    
train_ds = dlp.build_pipeline(train_subs, balanced=True, augment=True)
val_ds   = dlp.build_pipeline(val_subs, balanced=False, augment=False)
test_ds  = dlp.build_pipeline(test_subs, balanced=False, augment=False)

"""