import cv2
import matplotlib.pyplot as plt
import winsound
import time
import random
import IPython
import tqdm
import concurrent.futures
import os
import numpy as np


print("utilities imported")

def notify():
    for i in range(3):
        random_freq = random.randint(100, 2000)
        random_duration = random.randint(500, 1000)
        random_sleep = random.uniform(0.1, 0.2)
        winsound.Beep(random_freq, random_duration)
        time.sleep(random_sleep)
    print("done")

def load_landmarks(point_file):
    """Parses CASIA-FASD text files."""
    points = {}
    with open(point_file, 'r') as f:
        for line in f:
            data = list(map(int, line.split()))
            # Map: {frame_idx: [(x,y), (w,h), (eyeL), (eyeR)]}
            points[data[0]] = [(data[1], data[2]), (data[3], data[4]), 
                               (data[5], data[6]), (data[7], data[8])]
    return points

def videoToFrames(videoPath):
    cap = cv2.VideoCapture(videoPath)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def displayFrame(frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.show()

def displayFrames(frames,speed=1):
    for i, frame in enumerate(frames):
        if i % speed == 0:
            displayFrame(frame)
            IPython.display.clear_output(wait=True)

# ////////////////////////////////////////////////////////////////////////


def process_video_to_disk(video_path, points_path, output_prefix, processor):
    points = load_landmarks(points_path)
    cap = cv2.VideoCapture(video_path)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx in points:
            processed = processor.crop_and_resize(frame, points[frame_idx])
            if processed is not None:
                save_path = f"{output_prefix}{frame_idx:03d}.jpg"
                # print(video_path  +"  " + save_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, processed)
        
        frame_idx += 1
    cap.release()

def parallel_process(data_dict, dataFolder, processor, max_workers=4):
    """
    Processes multiple videos in parallel using CPU threads.
    """
    # Wrap the process_video_to_disk function from earlier
    video_list = list(data_dict.items())
    
    with tqdm.tqdm(total=len(video_list), desc="GPU-Ready Processing") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for path, newPath in video_list:
                # Construct paths
                v_path = path + ".avi" 
                p_path = path + ".txt"
                out_prefix = os.path.join(dataFolder, newPath)
                
                # Submit to threads
                futures.append(executor.submit(process_video_to_disk, v_path, p_path, out_prefix, processor))
            
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)

    
class FaceProcessor:
    def __init__(self, target_size=(224, 224), default_margin=30):
        self.target_size = target_size
        self.default_margin = default_margin

    def get_safe_margin(self, frame_shape, x, y, w, h):
        h_f, w_f, _ = frame_shape
        # Calculate distance to all 4 edges
        max_m = min(x, y, w_f - (x + w), h_f - (y + h))
        return max(0, min(self.default_margin, max_m))

    def crop_and_resize(self, frame, coords):
        (x, y), (w, h) = coords[0], coords[1]
        
        if w <= 0 or h <= 0:
            return None
            
        margin = self.get_safe_margin(frame.shape, x, y, w, h)
        
        crop = frame[y-margin : y+h+margin, x-margin : x+w+margin]
        return crop