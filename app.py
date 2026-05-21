import os
import sys
import threading
import numpy as np
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
from sympy import im

# 1. Environment Optimization 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import tensorflow as tf

def init_resNet50V2_FASD_RGB_V8_2(input_shape=(224, 224, 3)):
    img_input = tf.keras.layers.Input(shape=input_shape)
    
    # 1. Advanced Augmentation: Strictly prevents "cheating" on background edges
    augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomErasing(
        factor=0.5, 
        scale=(0.02, 1/3) # Erase between 2% and 33% of the image
        
    ),
    tf.keras.layers.RandomRotation(0.05), # Reduced from 0.15 to keep faces upright
    tf.keras.layers.RandomBrightness(0.0005), # Reduced to avoid blowing out skin texture
    ], name="data_augmentation")
    
    x = augmentation(img_input)

    # 2. Backbone with ImageNet weights 
    # (Starting from scratch makes 0.0 EER nearly impossible)
    base_model = tf.keras.applications.ResNet50V2(
        input_tensor=x,
        include_top=False,
        weights=None
    )

    # Branch B: Mid-level semantic structure (depth cues, surface gradients)
    mid_features = base_model.get_layer("conv3_block3_out").output

    # 4. Squeeze-and-Excitation (SE) on both scales
    def apply_se(tensor, ratio=16):
        f = tensor.shape[-1]
        se = tf.keras.layers.GlobalAveragePooling2D()(tensor)
        se = tf.keras.layers.Dense(f // ratio, activation='relu')(se)
        se = tf.keras.layers.Dense(f, activation='sigmoid')(se)
        se = tf.keras.layers.Reshape((1, 1, f))(se)
        return tf.keras.layers.Multiply()([tensor, se])


    mid_features = apply_se(mid_features)

    # 5. Hybrid Global Fusion
    # We pool both scales to ensure global context and local anomalies are captured
    
    mid_pool = tf.keras.layers.Concatenate()([
        tf.keras.layers.GlobalAveragePooling2D()(mid_features),
        tf.keras.layers.GlobalMaxPooling2D()(mid_features)
    ])

    # 6. Dense Integration Head
    
    
    x = tf.keras.layers.Dense(512, activation='swish')(mid_pool) # Swish often helps deeper convergence
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(128, activation='swish')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=img_input, outputs=output, name="resNet50V2_FASD_RGB_V8_RandomErasing")
    return model

def eer(y_true, y_pred):
    return 0.0

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temporary folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# 2. Global Configurations
MODEL_FILE_PATH = resource_path("./experiment_106_20260511-222122_1_model_resNet50V2_FASD_RGB_V8_RandomErasing.keras")
DETECTOR_ASSET_PATH = resource_path("./detector.tflite")
INPUT_RESOLUTION = (224, 224)
PADDING_MARGIN = 0.05  

class AntiSpoofingEngine:
    """Handles deep learning model execution and face detection tasks."""
    def __init__(self, model_path, detector_path):
        self.model = init_resNet50V2_FASD_RGB_V8_2()
        self.model.load_weights(model_path)
        self.spoof_threshold = 0.5
        self.face_detection_confidence_threshold = 0.7
        self.detector_path = detector_path
        
        base_options = python.BaseOptions(model_asset_path=detector_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self.face_detection_confidence_threshold
        )
        self.face_detector = vision.FaceDetector.create_from_options(options)

    
    def set_face_confidence_threshold(self, new_threshold):
        """Updates the face detection confidence threshold in the engine."""
        self.face_detection_confidence_threshold = new_threshold
        updated_options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=self.detector_path),
            min_detection_confidence=new_threshold
        )
        try:
            new_detector = vision.FaceDetector.create_from_options(updated_options)
            old_detector = self.face_detector
            self.face_detector = new_detector
            old_detector.close()
        except Exception as e:
            print(f"[ERROR] Failed to hot-swap face detector threshold: {e}")
    
    def set_spoof_threshold(self, new_threshold):
        """Updates the spoof classification threshold in the engine."""
        self.spoof_threshold = new_threshold        

    def detect_faces(self, bgr_frame):
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.face_detector.detect(mp_image)

    def preprocess_face(self, face_crop, target_size=INPUT_RESOLUTION):
        resized = cv2.resize(face_crop, target_size)
        rgb_crop = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_crop.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def predict_spoof(self, face_tensor):
        prediction_tensor = self.model(face_tensor, training=False)
        return prediction_tensor.numpy()[0][0]


class ResponsiveAntiSpoofingUI(ctk.CTk):
    """Main Application Window Layout with full Grid Responsiveness."""
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        
        # Initialize Base Window Parameters
        self.title("Intelligent Facial Spoofing Detection System")
        self.geometry("1000x650")
        # self.minimum_size(800, 550) # Prevents layout breaking from extreme shrinking
        
        # Application Runtime Variables
        self.current_prediction = "Scanning..."
        self.confidence_score = 0.0
        self.prediction_probability = 0.0
        self.box_color = (255, 255, 255)
        self.status_color = "#FFCC00"
        
        # Build Grid Layout Structures
        self._configure_window_responsiveness()
        self._build_ui_layout()
        
        # Thread Pipeline Bindings
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.video_thread = threading.Thread(target=self._video_processing_loop, daemon=True)
        self.video_thread.start()
        
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_window_responsiveness(self):
        """Configures row/column weights to define how components scale dynamically."""
        self.grid_columnconfigure(0, weight=4)  
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def _build_ui_layout(self):
        """Constructs a responsive grid structure separating video from metrics."""
        
        # 1. Main Video Container (Left side)
        # Binds to grid location (0,0). sticky="nsew" stretches it to fill its whole cell space
        self.video_container = ctk.CTkFrame(self, corner_radius=12, fg_color="black")
        self.video_container.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="nsew")
        
        # Configure internal centering rules inside the video frame container
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(0, weight=1)
        
        self.video_label = ctk.CTkLabel(self.video_container, text="Initializing Camera...", font=ctk.CTkFont(size=16), text_color="white")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        
        # 2. Control Readout Panel (Right side)
        # Binds to grid location (0,1).
        self.sidebar = ctk.CTkFrame(self, width=400, corner_radius=12)
        self.sidebar.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nsew")
        
        # Configure rows within sidebar for distributed component spacing
        self.sidebar.grid_columnconfigure(0, weight=1)
        
        self.title_label = ctk.CTkLabel(
            self.sidebar, text="SYSTEM STATUS", font=ctk.CTkFont(size=18, weight="bold")
        )
        self.title_label.grid(row=0, column=0, pady=(25, 35), padx=20, sticky="ew")
        
        self.status_card = ctk.CTkLabel(
            self.sidebar, text="INITIALIZING", font=ctk.CTkFont(size=24, weight="bold"), text_color=self.status_color
        )
        self.status_card.grid(row=1, column=0, pady=5, padx=20, sticky="ew")
        
        # Dynamic Metrics Labels (updated in real-time by video processing loop)
        self.prob_label = ctk.CTkLabel(
            self.sidebar, text="Live Probability: 0.00%", font=ctk.CTkFont(size=15)
        )
        self.prob_label.grid(row=2, column=0, pady=(15,0), padx=0, sticky="ew")
        
        self.conf_label = ctk.CTkLabel(
            self.sidebar, text="Confidence: 0.00%", font=ctk.CTkFont(size=15)
        )
        self.conf_label.grid(row=3, column=0, pady=0, padx=0, sticky="ew")
        
        # Structural Divider Line
        self.divider = ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30")
        self.divider.grid(row=4, column=0, pady=25, padx=15, sticky="ew")
        
       
        self.slider_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.slider_frame.grid(row=5, column=0, pady=0, padx=15, sticky="ew")
        self.slider_frame.grid_columnconfigure(0, weight=1)
        
        self.face_confidence_slider_title = ctk.CTkLabel(
            self.slider_frame, 
            text="Face Confidence Threshold: 0.7%", 
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.face_confidence_slider_title.grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.face_confidence_slider = ctk.CTkSlider(
            self.slider_frame,
            from_=0.0,          
            to=1.0,                
            number_of_steps=100,
            command=self._on_face_confidence_slider_change
        )
        self.face_confidence_slider.grid(row=1, column=0, sticky="ew")
        self.face_confidence_slider.set(0.7) # Set initial position to match your default (0.7)
        
        self.liveness_slider_title = ctk.CTkLabel(
            self.slider_frame, 
            text="Spoof Threshold: 0.5%", 
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.liveness_slider_title.grid(row=2, column=0, sticky="w", pady=(20, 5))
        self.liveness_slider = ctk.CTkSlider(
            self.slider_frame,
            from_=0.0,            
            to=1.0,             
            number_of_steps=100, 
            command=self._on_liveness_slider_change
        )
        self.liveness_slider.grid(row=3, column=0, sticky="ew")
        self.liveness_slider.set(0.5) # Set initial position to match your default (0.5)
        
        self.margin_slider_title = ctk.CTkLabel(
            self.slider_frame, 
            text="Face Margin: 5%", 
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.margin_slider_title.grid(row=4, column=0, sticky="w", pady=(20, 5))
        self.margin_slider = ctk.CTkSlider(
            self.slider_frame,
            from_=1,            
            to=100,             
            number_of_steps=100, 
            command=self._on_margin_slider_change
        )
        self.margin_slider.grid(row=5, column=0, sticky="ew")
        self.margin_slider.set(5) # Set initial position to match your default (50)
        

        self.sidebar.grid_rowconfigure(6, weight=1)
    
        param_text = (
            f"Classifier: {self.engine.model.name}\n"
            f"accuracy  : 98%\n"
            f"EER       : 0.46%\n"
            f"Resolution: {INPUT_RESOLUTION[0]}x{INPUT_RESOLUTION[1]}\n"
        )
        self.param_label = ctk.CTkLabel(
            self.sidebar, 
            text=param_text, 
            justify="left",
            font=ctk.CTkFont(size=12, family="Courier")
        )
        self.param_label.grid(row=7, column=0, pady=(0, 0), padx=15, sticky="sw")
    
    def _on_face_confidence_slider_change(self, value):
        """Callback for when the face confidence threshold slider is adjusted."""
        new_threshold = float(value)
        self.engine.set_face_confidence_threshold(new_threshold)
        self.face_confidence_slider_title.configure(text=f"Face Confidence Threshold: {new_threshold * 100:.1f}%")  

    def _on_liveness_slider_change(self, value):
        """Callback for when the spoof threshold slider is adjusted."""
        new_threshold = float(value)
        self.engine.set_spoof_threshold(new_threshold)
        self.liveness_slider_title.configure(text=f"Spoof Threshold: {new_threshold * 100:.1f}%")
    def _on_margin_slider_change(self, value):
        """Callback for when the face margin slider is adjusted."""
        global PADDING_MARGIN
        PADDING_MARGIN = float(value) / 100.0
        self.margin_slider_title.configure(text=f"Face Margin: {int(PADDING_MARGIN * 100)}%")

    def _resize_preserve_aspect(self, frame, container_w, container_h):
        """
        Resizes an image frame while completely preserving its original aspect ratio.
        Pads the remaining empty space with solid black bars (letterboxing/pillarboxing).
        """
        frame_h, frame_w = frame.shape[:2]
        
        # Determine the maximum scale factor that can fit inside the current UI container
        scale = min(container_w / frame_w, container_h / frame_h)
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)
        
        if new_w == frame_w and new_h == frame_h:
            return frame
        # Scale down cleanly using AREA interpolation to prevent pixel aliasing artifacts
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a blank black canvas matching the exact structural dimensions of the UI container
        canvas = np.zeros((container_h, container_w, 3), dtype=np.uint8)
        
        # Calculate pixel offset coordinates to center the resized frame perfectly on the canvas
        x_offset = (container_w - new_w) // 2
        y_offset = (container_h - new_h) // 2
        
        # Overlay the scaled camera frame onto the centered coordinates of the canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
        return canvas

    def _video_processing_loop(self):
        """Asynchronous execution worker loop processing video arrays dynamically."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            detection_result = self.engine.detect_faces(frame)

            if detection_result and detection_result.detections:
                for detection in detection_result.detections:
                    bbox = detection.bounding_box
                    xmin, ymin = bbox.origin_x, bbox.origin_y
                    box_w, box_h = bbox.width, bbox.height
                    detection_score = detection.categories[0].score
                    
                    pad_w = int(box_w * PADDING_MARGIN)
                    pad_h = int(box_h * PADDING_MARGIN)
                    
                    x1 = max(0, xmin - pad_w)
                    y1 = max(0, ymin - pad_h)
                    x2 = min(w, xmin + box_w + pad_w)
                    y2 = min(h, ymin + box_h + pad_h)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        input_tensor = self.engine.preprocess_face(face_crop)
                        prediction = self.engine.predict_spoof(input_tensor)
                        # prediction = 1.0
                        
                        if prediction > self.engine.spoof_threshold:
                            self.current_prediction = "LIVE FACE"
                            self.confidence_score = prediction * 100
                            self.box_color = (0, 255, 0)       
                            self.status_color = "#00FF00"
                        else:
                            self.current_prediction = f"SPOOF ATTACK"
                            self.confidence_score = (1.0 - prediction) * 100
                            self.box_color = (0, 0, 255)       
                            self.status_color = "#FF0000"
                            
                        
                        self.status_card.configure(text=self.current_prediction, text_color=self.status_color)
                        self.prob_label.configure(text=f"Live Probability: {prediction * 100:.2f}%")
                        self.conf_label.configure(text=f"Confidence: {self.confidence_score:.2f}%")
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, 2)
                    cv2.putText(frame, f"face : {detection_score * 100:.2f}%", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.box_color, 2)
                    break  
            else:
                self.status_card.configure(text="NO FACE", text_color="#FFFFFF")
                self.prob_label.configure(text="Live Probability: 0.00%")
                self.conf_label.configure(text="Confidence: 0.00%")
            
            # --- RESPONSIVE FRAMERATE RENDERING ENGINE ---
            # Get current application window coordinates to resize camera frame smoothly
            self.update_idletasks() 
            target_w = max(100, self.video_container.winfo_width())
            target_h = max(100, self.video_container.winfo_height())

            frame_resized = self._resize_preserve_aspect(frame, target_w, target_h)
            rgb_render_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            
            img_pil = Image.fromarray(rgb_render_frame)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.video_label.configure(image=img_tk, text="")
            self.video_label.image = img_tk

    def on_close(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.destroy()
        sys.exit(0)


if __name__ == "__main__":
    try:
        if not os.path.exists(MODEL_FILE_PATH):
            raise FileNotFoundError(f"Missing weight target: {MODEL_FILE_PATH}")
        if not os.path.exists(DETECTOR_ASSET_PATH):
            raise FileNotFoundError(f"Missing model file asset: {DETECTOR_ASSET_PATH}")
            
        engine = AntiSpoofingEngine(MODEL_FILE_PATH, DETECTOR_ASSET_PATH)
        app = ResponsiveAntiSpoofingUI(engine)
        app.mainloop()
        
    except Exception as err:
        print(f"[FATAL ERROR]: {err}")