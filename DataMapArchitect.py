import os
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# 1. GLOBAL FILTER FUNCTIONS (PICKLE-SAFE)
# ==========================================

import custom_image_selecter as cis

def uniform_sampling(frame_paths, keep_ratio=1.0, image_selection_config=None):
    """Simple uniform downsampling."""
    
    file_name_len = 13
    frame_paths = [f[-file_name_len:] for f in frame_paths]  # Keep only the last part of the filename for sorting
    return frame_paths[::max(1, int(1/keep_ratio))] if keep_ratio < 1.0 else frame_paths


def brightness_contrast_movement_based_selecting(frame_paths, keep_ratio=1.0, image_selection_config=None):
    """Advanced selection using the CustomImageSelector."""

    try:
        frameSelector = cis.CustomImageSelector(imageSelectionConfig=image_selection_config)
        num_frames_to_keep = max(1, int(len(frame_paths) * keep_ratio))
        return frameSelector.select_best_frames_paths(frame_paths, num_frames_to_keep)
    except Exception as e:
        print(f"Filter Error: {e}")
        return frame_paths  # Fallback to all frames on error

# ==========================================
# 2. STANDALONE WORKER (THE ENGINE)
# ==========================================

def _process_video_unit(img_folder, frame_list, filter_fn, keep_ratio, sub, label_key, image_selection_config=None):
    """
    This function runs on a separate CPU core. 
    It returns the processed paths AND the metadata to rebuild the map.
    """
    frame_paths = [os.path.join(img_folder, f) for f in frame_list]
    
    if filter_fn is not None:
        processed = filter_fn(frame_paths, keep_ratio, image_selection_config)
    else:
        processed = frame_paths
        
    return sub, label_key, processed

# ==========================================
# 3. THE ARCHITECT CLASS
# ==========================================

class DataMapArchitect:
    def __init__(self, video_id_pos_in_file_name=1, video_id_segment_number=1, separator="_"):
        """
        video_id_pos_in_file_name: The position of the video ID in the filename when split by the separator.
        video_id_segment_number: How many segments to combine for the video ID (useful if the ID is composed of multiple parts).
        separator: The character used to split the filename into segments (default is "_").

        """
        self.video_id_pos = video_id_pos_in_file_name
        self.video_id_seg_num = video_id_segment_number
        self.separator = separator

    def _get_image_files(self, folder_path):
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        return [f for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)]

    def _extract_group_id(self, filename):
        name_part = filename.split('.')[0]
        segments = name_part.split(self.separator)
        if len(segments) <= self.video_id_pos:
            return "unknown_video"
        id_segments = segments[self.video_id_pos : self.video_id_pos + self.video_id_seg_num]
        return self.separator.join(id_segments)

    def _group_by_video(self, file_list):
        groups = {}
        for f in file_list:
            vid_id = self._extract_group_id(f)
            if vid_id not in groups: groups[vid_id] = []
            groups[vid_id].append(f)
        return groups


    def create_map_parallel(self, dataroute, max_workers=None, keep_ratio=1.0, filter_fn=None, image_selection_config=None):
        """
        Main function to create the data map with parallel processing.
         - dataroute: Root directory of the dataset.
            - max_workers: Number of parallel processes (defaults to number of CPU cores).
            - keep_ratio: Proportion of frames to keep (0 < keep_ratio <= 1).
            - filter_fn: Function to select frames (must be defined at the global level).
            - image_selection_config: Configuration object for the filter function (if needed).
            Returns a structured dictionary with the mapping of subjects to their real/spoof frames.
        """
        # Initialize the new structure
        final_output = {
            'metadata': {
                'keep_ratio': keep_ratio,
                'filter_function': filter_fn.__name__ if filter_fn else "all_frames",
                'selection_config': image_selection_config.__dict__ if image_selection_config else cis.ImageSelectionConfig().__dict__,
            },
            'subjects': {}
        }
        
        subjects = [s for s in os.listdir(dataroute) if os.path.isdir(os.path.join(dataroute, s))]
        all_video_tasks = []

        # 1. Prepare Task List (Scanning)
        for sub in subjects:
            sub_path = os.path.join(dataroute, sub)
            final_output['subjects'][sub] = {'real': [], 'spoof': []}

            for label_dir in os.listdir(sub_path):
                label_key = 'spoof' if any(x in label_dir.lower() for x in ['spoof', 'fake', 'attack']) else 'real'
                img_folder = os.path.join(sub_path, label_dir)
                if not os.path.isdir(img_folder): continue

                images = self._get_image_files(img_folder)
                groups = self._group_by_video(images)

                for vid_id, frame_list in groups.items():
                    all_video_tasks.append({
                        'img_folder': img_folder,
                        'frame_list': frame_list,
                        'sub': sub,
                        'label_key': label_key,
                        'image_selection_config': image_selection_config
                    })

        # 2. Execute with Multiprocessing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _process_video_unit, 
                    t['img_folder'], t['frame_list'], filter_fn, keep_ratio, t['sub'], t['label_key'], t['image_selection_config']
                ) for t in all_video_tasks
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Mapping Progress"):
                try:
                    sub, label_key, processed_paths = future.result()
                    final_output['subjects'][sub][label_key].extend(processed_paths)
                except Exception as e:
                    print(f"\nWorker Error: {e}")

        return final_output

    def save_to_json(self, save_path, data_map):
        """Saves the data map to a JSON file, ensuring that all data types are JSON-serializable.
         - save_path: The file path where the JSON will be saved.
         - data_map: The dictionary containing the data map to be saved."""
        
        
        
        def default_conv(obj):
            if isinstance(obj, (np.integer, np.floating)): return obj.item()
            if isinstance(obj, np.ndarray): return obj.tolist()
            return str(obj)

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data_map, f, indent=4, default=default_conv)
        print(f"\n[SUCCESS] Map saved to: {save_path}")


## Example of usage
if __name__ == "__main__":
    # CONFIGURATION
    DATA_PATH = "./CASIA_FASD_v3/DATA" # Change to dataroute
    OUTPUT_JSON = "./casia_map.json"
    
    # Initialize Architect
    architect = DataMapArchitect()
    
    # Run the parallel mapping
    # Note: Pass the function name directly
    results = architect.create_map_parallel(
        dataroute=DATA_PATH,
        max_workers=os.cpu_count(),
        keep_ratio=0.1, 
        filter_fn=brightness_contrast_movement_based_selecting,
        image_selection_config=cis.ImageSelectionConfig(
            min_brightness_value=10, 
            max_brightness_value=90, 
            brightness_step=10,
            min_entropy_value=3.0,
            max_entropy_value=7.0,
            entropy_step=0.5
        )
    )
    
    # Save the final result
    architect.save_to_json(OUTPUT_JSON, results)