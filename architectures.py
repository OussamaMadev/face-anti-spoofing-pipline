from tensorflow.keras import layers, models
import tensorflow as tf

def SimpleCasiaNet(input_shape=(224, 224, 3)):
  model = models.Sequential(name="SimpleCasiaNet")
  
  model.add(layers.Input(shape=input_shape))

  model.add(layers.Conv2D(32, (3, 3), activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D((2, 2)))

  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D((2, 2)))

  model.add(layers.Conv2D(128, (3, 3), activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D((2, 2)))

  model.add(layers.Conv2D(224, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))

  model.add(layers.Flatten())
  model.add(layers.Dense(256, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(1, activation='sigmoid')) 

  return model

def model_architecture_example(input_shape=(224, 224, 3)):
    """
    The smallest functional CNN for pipeline verification.
    Extremely fast to train and minimal memory footprint.
    """
    model = tf.keras.Sequential([
        # Single convolutional layer to process 2D spatial data
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Flatten and use a single Dense layer
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid') # 1=Real, 0=Spoof
    ])
    
    return model


def mc_resnet_rgb_hsv(input_shape=(224, 224, 3)):
    """A more complex architecture that combines RGB and HSV color spaces for better spoofing detection.
    This model is designed to capture both color and texture cues that are crucial for distinguishing real faces from spoofs.
    It uses residual blocks for deeper feature extraction and an attention mechanism to focus on important regions of the face."""
    input_rgb = layers.Input(shape=input_shape, name="input_rgb")
    
    # 1. Color Space Transformations
    input_hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x), name="rgb_to_hsv")(input_rgb)
    
    
    # 2. Multi-Channel Fusion with initial Noise Filtering
    merged = layers.Concatenate(axis=-1)([input_rgb, input_hsv])
    
    # 3. Entry Block: Depthwise Separable to treat color channels independently first
    x = layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu')(merged)
    x = layers.BatchNormalization()(x)
    
    # 4. Residual Blocks (Better for deeper learning without vanishing gradients)
    def res_block(tensor, filters):
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(tensor)
        
        val = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(tensor)
        val = layers.BatchNormalization()(val)
        val = layers.Conv2D(filters, (3, 3), padding='same')(val)
        val = layers.BatchNormalization()(val)
        
        return layers.Add()([shortcut, val])

    x = res_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x) # 112x112
    
    x = res_block(x, 128)
    # Simple Attention Mechanism: Focus on face texture, ignore background
    attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    x = layers.Multiply()([x, attention])
    x = layers.MaxPooling2D((2, 2))(x) # 56x56
    
    # 5. Spatial Pyramid Pooling (SPP) - Captures multi-scale spoofing cues
    # Useful for finding tiny screen pixels vs. large paper edges
    pool1 = layers.GlobalAveragePooling2D()(x)
    pool2 = layers.GlobalMaxPooling2D()(x)
    
    x = layers.Concatenate()([pool1, pool2])
    
    # 6. Dense Neck with stronger Regularization
    x = layers.Dense(256, activation='relu', kernel_regularizer='l2')(x)
    x = layers.Dropout(0.6)(x) # Higher dropout to prevent overfitting
    x = layers.Dense(64, activation='relu')(x)
    
    output = layers.Dense(1, activation='sigmoid', name="classifier")(x)
    
    return models.Model(inputs=input_rgb, outputs=output, name = "MC-ResNet-RGB-HSV")


def build_mc_cdcn_model(input_shape=(224, 224, 3), theta=0.7):
    """
    Builds the MC-CDCN model for Intelligent Facial Spoofing Detection.
    Combines RGB, HSV, YCrCb with Central Difference Convolutions.
    """

    # --- Custom CDC Layer inside the function for portability ---
    class CDC2D(layers.Layer):
        def __init__(self, filters, kernel_size=3, theta=0.7, **kwargs):
            super().__init__(**kwargs)
            self.filters, self.kernel_size, self.theta = filters, kernel_size, theta
            self.conv = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)
        def call(self, x):
            out_normal = self.conv(x)
            kernel_sum = tf.reduce_sum(self.conv.kernel, axis=(0, 1))
            out_center = tf.nn.convolution(x, tf.reshape(kernel_sum, (1, 1, x.shape[-1], self.filters)), padding='SAME')
            return out_normal - self.theta * out_center
        def get_config(self):
            return {**super().get_config(), "filters": self.filters, "kernel_size": self.kernel_size, "theta": self.theta}

    # --- 1. Inputs & Multi-Chromatic Fusion ---
    input_rgb = layers.Input(shape=input_shape, name="input_rgb")

    # Normalize and transform color spaces
    # Note: Assumes input is [0, 255], scales to [0, 1] for transformations
    x_norm = layers.Rescaling(1./255)(input_rgb)
    input_hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x), name="hsv_transform")(x_norm)
    input_ycbcr = layers.Lambda(lambda x: tf.image.rgb_to_yuv(x), name="ycbcr_transform")(x_norm)

    merged = layers.Concatenate(axis=-1, name="9_channel_fusion")([x_norm, input_hsv, input_ycbcr])

    # --- 2. Feature Extraction Blocks (CDC + Residual) ---
    def cdc_res_block(x, filters, stride=1):
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(x)

        # Branch with Central Difference
        val = CDC2D(filters, theta=theta)(x)
        val = layers.BatchNormalization()(val)
        val = layers.Activation('relu')(val)
        if stride > 1: val = layers.MaxPooling2D((stride, stride))(val)

        val = layers.Conv2D(filters, (3, 3), padding='same')(val)
        val = layers.BatchNormalization()(val)

        return layers.Add()([shortcut, val])

    # Block 1: Initial texture extraction
    x = cdc_res_block(merged, 64)
    x = layers.MaxPooling2D((2, 2))(x) # 112x112

    # Block 2: Middle-level features with Spatial Attention
    x = cdc_res_block(x, 128)
    attn = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    x = layers.Multiply()([x, attn])
    x = layers.MaxPooling2D((2, 2))(x) # 56x56

    # Block 3: High-level semantics
    x = cdc_res_block(x, 256)

    # --- 3. Global Head ---
    # Combine Average and Max pooling to catch both global structure and local "glints"
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    combined = layers.Concatenate()([gap, gmp])

    x = layers.Dense(128, activation='relu', kernel_regularizer='l2')(combined)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid', name="classifier")(x)

    return models.Model(inputs=input_rgb, outputs=output, name="MC_CDCN")


def hs_ConvNeXtTiny(input_shape=(224, 224, 3)):
    img_input = layers.Input(shape=input_shape)
    
    # Preprocessing: RGB -> HSV -> Extract H & S
    hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))(img_input)
    hs = layers.Lambda(lambda x: x[:, :, :, 0:2])(hsv)
    
    # Expand 2 channels to 3 for ConvNeXt compatibility
    x = layers.Conv2D(3, (1, 1), padding='same')(hs)
    
    base_model = tf.keras.applications.ConvNeXtTiny(
        include_top=False, 
        weights=None, 
        input_tensor=x
    )
    
    # Global Head
    x = layers.GlobalAveragePooling2D()(base_model.output)
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    return models.Model(img_input, output, name="hs_ConvNeXtTiny")


def hsv_ConvNeXtTiny(input_shape=(224, 224, 3)):
    img_input = layers.Input(shape=input_shape)
    
    # Preprocessing: RGB -> HSV
    hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))(img_input)
    
    base_model = tf.keras.applications.ConvNeXtTiny(
        include_top=False, 
        weights=None, 
        input_tensor=hsv
    )
    
    x = layers.GlobalAveragePooling2D()(base_model.output)
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    return models.Model(img_input, output, name="hsv_ConvNeXtTiny")



def get_lbp_gpu(image_channel):
    # Define constants as Tensors immediately
    a = tf.constant(0.20710678, dtype=tf.float32) 
    b = tf.constant(0.5, dtype=tf.float32)        
    c = tf.constant(0.91421354, dtype=tf.float32) 

    # Create the kernels using tf.stack instead of np.array
    # This keeps everything inside the TF ecosystem
    k0 = [[0, 0, 0], [0, -1, 1], [0, 0, 0]]
    k1 = [[0, a, b], [0, -c, a], [0, 0, 0]]
    k2 = [[0, 1, 0], [0, -1, 0], [0, 0, 0]]
    k3 = [[b, a, 0], [a, -c, 0], [0, 0, 0]]
    k4 = [[0, 0, 0], [1, -1, 0], [0, 0, 0]]
    k5 = [[0, 0, 0], [a, -c, 0], [b, a, 0]]
    k6 = [[0, 0, 0], [0, -1, 0], [0, 1, 0]]
    k7 = [[0, 0, 0], [0, -c, a], [0, a, b]]

    # Stack and reshape to [height, width, in_channels, out_channels]
    kernels = tf.stack([k0, k1, k2, k3, k4, k5, k6, k7], axis=0)
    kernels = tf.transpose(kernels, [1, 2, 0])
    kernels = tf.reshape(kernels, (3, 3, 1, 8))

    weights = tf.reshape(tf.constant([1, 2, 4, 8, 16, 32, 64, 128], dtype=tf.float32), (1, 1, 1, 8))

    # Standard Conv2D logic
    diff = tf.nn.conv2d(image_channel, kernels, strides=[1, 1, 1, 1], padding='SAME')
    binary_bits = tf.cast(diff >= -1e-7, tf.float32)
    
    return tf.reduce_sum(binary_bits * weights, axis=-1, keepdims=True) / 255.0

def HS_LBP_ConvNeXt(input_shape=(224, 224, 3)):
    # 1. Input: Raw RGB
    rgb_input = layers.Input(shape=input_shape, name="input_rgb")
    
    # 2. Preprocessing Stream (RGB -> HSV)
    # Using 0-1 range for internal calculations
    hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x), name="hsv_transform")(rgb_input)
    
    h = layers.Lambda(lambda x: x[:, :, :, 0:1], name="hue_channel")(hsv)
    s = layers.Lambda(lambda x: x[:, :, :, 1:2], name="saturation_channel")(hsv)
    v = layers.Lambda(lambda x: x[:, :, :, 2:3], name="value_channel")(hsv)
    
    # 3. Feature Engineering: Extract LBP from Value channel
    lbp_v = layers.Lambda(get_lbp_gpu, name="lbp_texture_extraction")(v)
    
    # 4. Feature Fusion: Stack [H, S, LBP(V)] as a 3-channel input
    # This allows us to use ImageNet-pretrained weights on the backbone
    fused_input = layers.Concatenate(axis=-1, name="spectral_texture_fusion")([h, s, lbp_v])
    
    # 5. Backbone: ConvNeXt Tiny
    base_model = tf.keras.applications.ConvNeXtTiny(
        include_top=False,
        weights=None,
        input_tensor=fused_input
    )
    
    # 6. Global Head (Optimized for Binary Classification)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x) # Robustness against overfitting
    
    # Sigmoid Output: 0 = Spoof, 1 = Real
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name="classification")(x)
    
    model = models.Model(rgb_input, outputs, name="HS_LBP_ConvNeXt")
    
    return model


def build_gram_net(input_shape=(224, 224, 3)):
# --- 1. Define the Custom Gram Layer ---
    class GramMatrixLayer(layers.Layer):
        def __init__(self, **kwargs):
            super(GramMatrixLayer, self).__init__(**kwargs)

        def call(self, x):
            shape = tf.shape(x)
            b, h, w, c = shape[0], shape[1], shape[2], shape[3]
            features = tf.reshape(x, (b, h * w, c))
            # (Batch, C, H*W) @ (Batch, H*W, C) -> (Batch, C, C)
            gram = tf.matmul(features, features, transpose_a=True)
            gram = gram / tf.cast(h * w, tf.float32)
            return tf.reshape(gram, (b, c * c))

    # --- 2. Create the Architecture ---
    img_input = layers.Input(shape=input_shape, name="input_img")
    
    # Pre-trained Backbone
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape, 
        include_top=False, 
        weights=None
        # weights='imagenet'
    )

    # CRITICAL FIX: Pass your img_input through the base model to create a connected graph
    # This creates a "new" version of the base model connected to your Input()
    base_out = base(img_input) 

    # Now we extract internal outputs from the base model
    # To ensure they are connected to img_input, we use the Model API to get intermediate tensors
    # Stage indices for MobileNetV3Small: 13 (low), 36 (mid), 142 (high)
    extractor = models.Model(
        inputs=base.input, 
        outputs=[base.layers[13].output, base.layers[36].output, base.layers[142].output]
    )
    
    # Link the extractor to our specific img_input
    feat_low, feat_mid, feat_high = extractor(img_input)

    # --- 3. Texture Extraction Logic ---
    def process_stage(feat, filters, name):
        # 1x1 Conv to reduce channels (keeps params low)
        x = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(feat)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # Gram Texture
        gram = GramMatrixLayer(name=f"gram_{name}")(x)
        # Spatial Structure
        gap = layers.GlobalAveragePooling2D(name=f"gap_{name}")(feat)
        return gram, gap

    g1, p1 = process_stage(feat_low, 32, "low")
    g2, p2 = process_stage(feat_mid, 32, "mid")
    g3, p3 = process_stage(feat_high, 32, "high")

    # --- 4. Fusion and Head ---
    fused = layers.Concatenate(name="global_fusion")([g1, p1, g2, p2, g3, p3])
    
    x = layers.Dense(512, activation='relu')(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(1, activation='sigmoid', name="decision")(x)

    model = models.Model(inputs=img_input, outputs=output, name="GramNet_ResNet")
    
    return model


def resnet_50v2(input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=input_shape,
        include_top=False,
        weights=None  # As per your Experiment 013 setup
    )
    
    x = base_model.output
    
    # CRITICAL FIX: Flatten the spatial dimensions into a single vector
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Optional: Add a Dropout layer to help with the 1.6% EER bottleneck
    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output, name = "resNet50V2_FASD")
    return model


def build_resnet50v2_hsv(input_shape=(224, 224, 6)):

    img_input = layers.Input(shape=input_shape)
    
    
    hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))(img_input)

    base_model = tf.keras.applications.ResNet50V2(
        input_tensor=hsv,
        include_top=False,
        weights=None 
    )
    
    x = base_model.output
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output, name = "resNet50V2_FASD_HSV")
    return model

def build_resnet50v2_hsv_rgb(input_shape=(224, 224, 6)):

    img_input = layers.Input(shape=input_shape)
    
    
    hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))(img_input)

    rgb_hsv = layers.Concatenate(axis=-1)([img_input, hsv])

    base_model = tf.keras.applications.ResNet50V2(
        input_tensor=rgb_hsv,
        include_top=False,
        weights=None 
    )    
    x = base_model.output
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output, name = "resNet50V2_FASD_HSV_RGB")
    return model

def build_resnet50v2_9channel(input_shape=(224, 224, 3)):
    """
    Expansion of ResNet50V2 to 9 channels: RGB (3) + HSV (3) + YCbCr (3).
    Focuses on chromaticity and luminance gradients for anti-spoofing.
    """
    # 1. Define the original 3-channel RGB input
    img_input = layers.Input(shape=input_shape, name="input_rgb")
    
    # 2. Color Space Transformations (Assuming input is normalized [0, 1])
    hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x), name="hsv_stream")(img_input)
    # Note: tf.image.rgb_to_yuv is the standard TF implementation for YCbCr-like logic
    ycbcr = layers.Lambda(lambda x: tf.image.rgb_to_yuv(x), name="ycbcr_stream")(img_input)

    # 3. Concatenate to 9 Channels
    merged = layers.Concatenate(axis=-1, name="9_channel_fusion")([img_input, hsv, ycbcr])

    # 4. Initialize ResNet50V2 with the 9-channel tensor
    # weights=None is mandatory as ImageNet weights only support 3 channels
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights=None,
        input_tensor=merged
    )
    
    # 5. Global Head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x) # Increased dropout for better generalization
    
    output = layers.Dense(1, activation='sigmoid', name="classifier")(x)
    
    model = models.Model(inputs=img_input, outputs=output, name="ResNet50V2_9Ch")
    return model

def build_improved_resnet50v2_multichannel(input_shape=(224, 224, 3)):
    img_input = layers.Input(shape=input_shape)
    
    # 1. Internal Multi-Chromatic Expansion (Ensure 0-1 range)
    hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))(img_input)
    ycbcr = layers.Lambda(lambda x: tf.image.rgb_to_yuv(x))(img_input)
    combined_input = layers.Concatenate(axis=-1)([img_input, hsv, ycbcr])

    # 2. Backbone with Intermediate Extractions
    base = tf.keras.applications.ResNet50V2(
        input_tensor=combined_input,
        include_top=False,
        weights=None 
    )
    
    # Capture intermediate texture features (Stage 3 and 4)
    feat_mid = base.get_layer("conv4_block6_out").output
    feat_final = base.output
    
    # 3. Multi-Scale Pooling
    p1 = layers.GlobalAveragePooling2D()(feat_mid)
    p2 = layers.GlobalAveragePooling2D()(feat_final)
    p3 = layers.GlobalMaxPooling2D()(feat_final) # Capture sharp glints/pixels
    
    merged = layers.Concatenate()([p1, p2, p3])
    
    # 4. Dense Head with Heavy Regularization
    x = layers.Dense(512, activation='relu', kernel_regularizer='l2')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x) # Aggressive dropout to bridge the gap
    
    output = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(inputs=img_input, outputs=output)

def build_improved_resnet50v2_multichannel_v2(input_shape=(224, 224, 3)):
    img_input = layers.Input(shape=input_shape)
    
    # 1. Internal Multi-Chromatic Expansion (Ensure 0-1 range)
    hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))(img_input)
    combined_input = layers.Concatenate(axis=-1)([img_input, hsv])

    # 2. Backbone with Intermediate Extractions
    base = tf.keras.applications.ResNet50V2(
        input_tensor=combined_input,
        include_top=False,
        weights=None 
    )
    
    # Capture intermediate texture features (Stage 3 and 4)
    feat_mid = base.get_layer("conv4_block6_out").output
    feat_final = base.output
    
    # 3. Multi-Scale Pooling
    p1 = layers.GlobalAveragePooling2D()(feat_mid)
    p2 = layers.GlobalAveragePooling2D()(feat_final)
    p3 = layers.GlobalMaxPooling2D()(feat_final) # Capture sharp glints/pixels
    
    merged = layers.Concatenate()([p1, p2, p3])
    
    # 4. Dense Head with Heavy Regularization
    x = layers.Dense(512, activation='relu', kernel_regularizer='l2')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x) # Aggressive dropout to bridge the gap
    
    output = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(inputs=img_input, outputs=output, name="Improved_ResNet50V2_MultiChannel_RGB_HSV")