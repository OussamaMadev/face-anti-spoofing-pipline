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

def build_resnet50v2_hsv_rgb_yuv(input_shape=(224, 224, 3)):

    img_input = layers.Input(shape=input_shape)
    hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))(img_input)
    yuv = layers.Lambda(lambda x: tf.image.rgb_to_yuv(x))(img_input)

    combined = layers.Concatenate(axis=-1)([img_input, hsv, yuv])

    base_model = tf.keras.applications.ResNet50V2(
        input_tensor=combined,
        include_top=False,
        weights=None 
    )    
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output, name = "resNet50V2_FASD_HSV_RGB_YUV")
    return model


def build_resnet50v2_hsv_rgb_v2(input_shape=(224, 224, 3)):
    img_input = layers.Input(shape=input_shape)
    
    # 1. 9-Channel Input Expansion (RGB + HSV + YUV)
    # Ensure images are float32 [0, 1] to avoid EagerTensor uint8 errors
    hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))(img_input)
    
    combined = layers.Concatenate(axis=-1)([img_input, hsv])

    # 2. ResNet Backbone
    base_model = tf.keras.applications.ResNet50V2(
        input_tensor=combined,
        include_top=False,
        weights=None 
    )
    res_out = base_model.output

    # 3. Dimensional Alignment (Global Pooling)
    # Get the global statistics of the 9-channel input (Mean color, mean YUV)
    combined_pooled = layers.GlobalAveragePooling2D()(combined)
    # Get the global deep features
    res_pooled = layers.GlobalAveragePooling2D()(res_out)
    
    # 4. Concatenation
    # Now both are vectors, so they can be merged
    merged = layers.Concatenate()([combined_pooled, res_pooled])
    
    # 5. Dense Classifier Head
    x = layers.Dense(512, activation='relu')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Using 1 dense layer for output as per your intent
    output = layers.Dense(1, activation='sigmoid', name="pad_output")(x)

    model = models.Model(inputs=img_input, outputs=output, name="resNet50V2_FASD_HSV_RGB_V2")
    return model

def build_resnet50v2_hsv_rgb_v3(input_shape=(224, 224, 3)):
    img_input = layers.Input(shape=input_shape)
    
    # 1. 9-Channel Input Expansion (RGB + HSV + YUV)
    # Ensure images are float32 [0, 1] to avoid EagerTensor uint8 errors
    hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))(img_input)
    
    combined = layers.Concatenate(axis=-1)([img_input, hsv])

    # 2. ResNet Backbone
    base_model = tf.keras.applications.ResNet50V2(
        input_tensor=combined,
        include_top=False,
        weights=None 
    )
    res_out = base_model.output

    a = layers.conv3d()(res_out)
    b = layers.conv3d()(combined)
    

    
    combined_pooled = layers.GlobalAveragePooling2D()(combined)
    # Get the global deep features
    res_pooled = layers.GlobalAveragePooling2D()(res_out)
    
    # 4. Concatenation
    # Now both are vectors, so they can be merged
    merged = layers.Concatenate()([combined_pooled, res_pooled])
    
    # 5. Dense Classifier Head
    x = layers.Dense(512, activation='relu')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Using 1 dense layer for output as per your intent
    output = layers.Dense(1, activation='sigmoid', name="pad_output")(x)

    model = models.Model(inputs=img_input, outputs=output, name="resNet50V2_FASD_HSV_RGB_V3")
    return model

def build_resnet50v2_texture_fusion(input_shape=(224, 224, 3)):
    img_input = layers.Input(shape=input_shape)
    
    # 1. Traditional Color Stream (RGB + HSV)
    hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))(img_input)
    
    # 2. Texture Preprocessing Stream
    # Laplacian highlights high-frequency noise and edges
    # Texture Preprocessing Stream
    gray = layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x))(img_input)
    
    laplacian_matrix = tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=tf.float32)
    # Change from [3,3] -> [3,3,1,1]
    laplacian_kernel = laplacian_matrix[..., tf.newaxis, tf.newaxis] 
    
    laplacian = layers.Lambda(
        lambda x: tf.nn.convolution(x, laplacian_kernel, padding='SAME')
    )(gray)
    
    # 3. 8-Channel Fusion (RGB + HSV + Gray + Laplacian)
    combined = layers.Concatenate(axis=-1)([img_input, hsv, gray, laplacian])

    # 4. ResNet Backbone
    base_model = tf.keras.applications.ResNet50V2(
        input_tensor=combined,
        include_top=False,
        weights=None 
    )
    
    # 5. Hybrid Global Pooling (The "EER Killer")
    # GAP catches the general face structure
    gap = layers.GlobalAveragePooling2D()(base_model.output)
    # GMP catches the "smoking gun" artifacts (pixels, glare)
    gmp = layers.GlobalMaxPooling2D()(base_model.output)
    
    # Merge the global insights
    merged = layers.Concatenate()([gap, gmp])
    
    # 6. Regularized Head
    x = layers.Dense(512, activation='relu')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=img_input, outputs=output, name="ResNet_Texture_Fusion")
    return model


def build_resnet50v2_hsv_rgb_v4(input_shape=(224, 224, 3)):

    img_input = layers.Input(shape=input_shape)
    
    
    hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))(img_input)

    rgb_hsv = layers.Concatenate(axis=-1)([img_input, hsv])

    base_model = tf.keras.applications.ResNet50V2(
        input_tensor=rgb_hsv,
        include_top=False,
        weights=None 
    )    
    
    mid_features = base_model.get_layer("conv3_block3_out").output

    # Hybrid Pooling
    avg_pool = layers.GlobalAveragePooling2D()(mid_features)
    max_pool = layers.GlobalMaxPooling2D()(mid_features)
    hybrid = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dense(256, activation='relu')(hybrid)

    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=img_input, outputs=output, name = "resNet50V2_FASD_HSV_RGB_V4")
    return model



def build_resnet50_hsv_rgb(input_shape=(224, 224, 3)):

    img_input = layers.Input(shape=input_shape)
    
    
    hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))(img_input)

    rgb_hsv = layers.Concatenate(axis=-1)([img_input, hsv])

    base_model = tf.keras.applications.ResNet50(
        input_tensor=rgb_hsv,
        include_top=False,
        weights=None 
    )    
    x = base_model.output
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=img_input, outputs=output, name = "resNet50_FASD_HSV_RGB")
    return model


def build_resnet50v2_hsv_rgb_h_lbp(input_shape=(224, 224, 3)):

    img_input = layers.Input(shape=input_shape)
    
    hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))(img_input)

    h = layers.Lambda(lambda x: x[:, :, :, 0:1], name="hue_channel")(hsv)
    lpb_h = layers.Lambda(get_lbp_gpu, name="lbp_extraction")(h)

    rgb_hsv = layers.Concatenate(axis=-1)([img_input, hsv, lpb_h])

    base_model = tf.keras.applications.ResNet50V2(
        input_tensor=rgb_hsv,
        include_top=False,
        weights=None 
    )    
    x = base_model.output
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=img_input, outputs=output, name = "resNet50V2_FASD_HSV_RGB_H_LBP")
    return model

def build_resnet50v2_hsv_rgb_grayscale_lbp(input_shape=(224, 224, 3)):

    img_input = layers.Input(shape=input_shape)
    
    hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))(img_input)

    grayscale = layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x), name="grayscale_conversion")(img_input)
    lpb_grayscale = layers.Lambda(get_lbp_gpu, name="lbp_extraction")(grayscale)

    rgb_hsv = layers.Concatenate(axis=-1)([img_input, hsv, lpb_grayscale])

    base_model = tf.keras.applications.ResNet50V2(
        input_tensor=rgb_hsv,
        include_top=False,
        weights=None 
    )    
    x = base_model.output
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=img_input, outputs=output, name = "resNet50V2_FASD_HSV_RGB_Grayscale_LBP")
    return model


def build_resnet50v2_hsv_rgb_drop5(input_shape=(224, 224, 3)):

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
    x = tf.keras.layers.Dropout(0.5)(x)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=img_input, outputs=output, name = "resNet50V2_FASD_HSV_RGB_Drop5")
    return model

def build_resnet50v2_hsv_rgb(input_shape=(224, 224, 3)):

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
    
    model = tf.keras.models.Model(inputs=img_input, outputs=output, name = "resNet50V2_FASD_HSV_RGB")
    return model

def build_resnet50v2_hsv_v4(input_shape=(224, 224, 3)):

    img_input = layers.Input(shape=input_shape)
    
    
    hsv = layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))(img_input)

    # rgb_hsv = layers.Concatenate(axis=-1)([img_input, hsv])

    base_model = tf.keras.applications.ResNet50V2(
        input_tensor=hsv,
        include_top=False,
        weights=None 
    )    
    
    mid_features = base_model.get_layer("conv3_block3_out").output

    # Hybrid Pooling
    avg_pool = layers.GlobalAveragePooling2D()(mid_features)
    max_pool = layers.GlobalMaxPooling2D()(mid_features)
    hybrid = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dense(256, activation='relu')(hybrid)

    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=img_input, outputs=output, name = "resNet50V2_FASD_HSV_V4")
    return model


def squeeze_excite_block(input_tensor, ratio=16):
    init = input_tensor
    filters = init.shape[-1]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)

    return layers.multiply([init, se])


def build_resnet50v2_rgb_v4_Fusion(input_shape=(224, 224, 3)):
    img_input = layers.Input(shape=input_shape)
    base_model = tf.keras.applications.ResNet50V2(input_tensor=img_input, include_top=False, weights=None)
    
    # Extract shallow and mid features
    f1 = base_model.get_layer("conv2_block3_out").output # Early texture
    f2 = base_model.get_layer("conv3_block3_out").output # Mid-level patterns

    # Pool both and fuse
    p1 = layers.GlobalAveragePooling2D()(f1)
    p2 = layers.GlobalAveragePooling2D()(f2)
    
    fusion = layers.Concatenate()([p1, p2])
    x = layers.Dense(256, activation='relu')(fusion)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.models.Model(inputs=img_input, outputs=output, name="resNet50V2_Fusion_V4")

def build_resnet50v2_rgb_v4_Patches(input_shape=(224, 224, 3), patch_size=112):
    img_input = layers.Input(shape=input_shape)
    
    # Random Crop Layer (Active during training)
    x = layers.RandomCrop(patch_size, patch_size)(img_input)
    x = layers.Resizing(224, 224)(x) # Resize back to expected input size
    
    base_model = tf.keras.applications.ResNet50V2(input_tensor=x, include_top=False, weights=None)
    mid_features = base_model.get_layer("conv3_block3_out").output

    hybrid = layers.Concatenate()([layers.GlobalAveragePooling2D()(mid_features), 
                                   layers.GlobalMaxPooling2D()(mid_features)])
    
    x = layers.Dense(256, activation='relu')(hybrid)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.models.Model(inputs=img_input, outputs=output, name="resNet50V2_Patches_V4")


def build_resnet50v2_rgb_v4_Cutout(input_shape=(224, 224, 3)):
    img_input = layers.Input(shape=input_shape)
    
    # Randomly "delete" a patch of the image (set to 0)
    # Using a high dropout rate for the spatial area
    x = layers.SpatialDropout2D(0.1)(img_input) 
    
    base_model = tf.keras.applications.ResNet50V2(input_tensor=x, include_top=False, weights=None)
    mid_features = base_model.get_layer("conv3_block3_out").output

    hybrid = layers.Concatenate()([layers.GlobalAveragePooling2D()(mid_features), 
                                   layers.GlobalMaxPooling2D()(mid_features)])
    
    x = layers.Dense(256, activation='relu')(hybrid)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.models.Model(inputs=img_input, outputs=output, name="resNet50V2_Cutout_V4")


def build_resnet50v2_rgb_v4_SE(input_shape=(224, 224, 3)):
    img_input = layers.Input(shape=input_shape)
    base_model = tf.keras.applications.ResNet50V2(input_tensor=img_input, include_top=False, weights=None)
    
    mid_features = base_model.get_layer("conv3_block3_out").output
    
    attended_features = squeeze_excite_block(mid_features)

    avg_pool = layers.GlobalAveragePooling2D()(attended_features)
    max_pool = layers.GlobalMaxPooling2D()(attended_features)
    hybrid = layers.Concatenate()([avg_pool, max_pool])
    
    x = layers.Dense(256, activation='relu')(hybrid)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.models.Model(inputs=img_input, outputs=output, name="resNet50V2_SE_V4")


def build_resnet50v2_rgb_v4_light_classifier(input_shape=(224, 224, 3)):

    img_input = layers.Input(shape=input_shape)
    
    base_model = tf.keras.applications.ResNet50V2(
        input_tensor=img_input,
        include_top=False,
        weights=None 
    )    
    
    mid_features = base_model.get_layer("conv3_block3_out").output

    avg_pool = layers.GlobalAveragePooling2D()(mid_features)
    max_pool = layers.GlobalMaxPooling2D()(mid_features)
    hybrid = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dense(128, activation='relu')(hybrid)

    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=img_input, outputs=output, name = "resNet50V2_FASD_RGB_V4_light_classifier")
    return model

def build_resnet50v2_rgb_v4_4erasedPatches_scale_5(input_shape=(224, 224, 3)):

    img_input = layers.Input(shape=input_shape)

    input = layers.RandomErasing(factor=(1.0,1.0), scale=(0.05, 0.05), fill_value=0.0)(img_input)
    input = layers.RandomErasing(factor=(1.0,1.0), scale=(0.05, 0.05), fill_value=0.0)(input)
    input = layers.RandomErasing(factor=(1.0,1.0), scale=(0.05, 0.05), fill_value=0.0)(input)
    input = layers.RandomErasing(factor=(1.0,1.0), scale=(0.05, 0.05), fill_value=0.0)(input)
    
    base_model = tf.keras.applications.ResNet50V2(
        input_tensor=input,
        include_top=False,
        weights=None 
    )    
    
    mid_features = base_model.get_layer("conv3_block3_out").output

    avg_pool = layers.GlobalAveragePooling2D()(mid_features)
    max_pool = layers.GlobalMaxPooling2D()(mid_features)
    hybrid = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dense(256, activation='relu')(hybrid)

    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=img_input, outputs=output, name = "resNet50V2_FASD_RGB_V4_4erasedPatches_scale_5")
    return model

def build_resnet50v2_rgb_v4_4erasedPatches_scale_5_prob_05(input_shape=(224, 224, 3)):

    img_input = layers.Input(shape=input_shape)
    prob = (0.5, 0.5)  # 50% chance to apply erasing
    scale = (0.05, 0.05)

    input = layers.RandomErasing(factor=prob, scale=scale, fill_value=0.0)(img_input)
    input = layers.RandomErasing(factor=prob, scale=scale, fill_value=0.0)(input)
    input = layers.RandomErasing(factor=prob, scale=scale, fill_value=0.0)(input)
    input = layers.RandomErasing(factor=prob, scale=scale, fill_value=0.0)(input)
    
    base_model = tf.keras.applications.ResNet50V2(
        input_tensor=input,
        include_top=False,
        weights=None 
    )    
    
    mid_features = base_model.get_layer("conv3_block3_out").output

    avg_pool = layers.GlobalAveragePooling2D()(mid_features)
    max_pool = layers.GlobalMaxPooling2D()(mid_features)
    hybrid = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dense(256, activation='relu')(hybrid)

    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=img_input, outputs=output, name = "resNet50V2_FASD_RGB_V4_4erasedPatches_scale_5_prob_0.5")
    return model


def build_resNet50V2_rgb_v4_5streams(input_shape=(224, 224, 3)):
    full_img_input = layers.Input(shape=input_shape, name="original_input")

    def extract_upscale(x, x1, y1, x2, y2, name):
        patch = layers.Lambda(lambda img: img[:, y1:y2, x1:x2, :])(x)
        return layers.Resizing(224, 224, name=f"upscale_{name}")(patch)

    p_tl = extract_upscale(full_img_input, 0, 0, 112, 112, "TL")
    p_tr = extract_upscale(full_img_input, 112, 0, 224, 112, "TR")
    p_bl = extract_upscale(full_img_input, 0, 112, 112, 224, "BL")
    p_br = extract_upscale(full_img_input, 112, 112, 224, 224, "BR")

    p_global = full_img_input 

    shared_resnet = tf.keras.applications.ResNet50V2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    
    feature_extractor = models.Model(
        inputs=shared_resnet.input,
        outputs=shared_resnet.get_layer("conv3_block3_out").output,
        name="shared_backbone_224"
    )

    def get_features(tensor):
        x = feature_extractor(tensor)
        avg_p = layers.GlobalAveragePooling2D()(x)
        max_p = layers.GlobalMaxPooling2D()(x)
        return layers.Concatenate()([avg_p, max_p])

    feat_tl = get_features(p_tl)
    feat_tr = get_features(p_tr)
    feat_bl = get_features(p_bl)
    feat_br = get_features(p_br)
    feat_global = get_features(p_global)

    merged = layers.Concatenate(name="merge_5_streams")(
        [feat_tl, feat_tr, feat_bl, feat_br, feat_global]
    )

    x = layers.Dense(256, activation='relu')(merged)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inputs=full_img_input, outputs=output, name="resNet50V2_rgb_v4_5streams")

def build_resNet50V2_rgb_v4_4streams(input_shape=(224, 224, 3)):
    full_img_input = layers.Input(shape=input_shape, name="original_input")

    def extract(x, x1, y1, x2, y2, name):
        patch = layers.Lambda(lambda img: img[:, y1:y2, x1:x2, :], name=f"patch_{name}")(x)
        return patch

    p_tl = extract(full_img_input, 0, 0, 112, 112, "TL")
    p_tr = extract(full_img_input, 112, 0, 224, 112, "TR")
    p_bl = extract(full_img_input, 0, 112, 112, 224, "BL")
    p_br = extract(full_img_input, 112, 112, 224, 224, "BR")


    shared_resnet = tf.keras.applications.ResNet50V2(
        input_shape=(112, 112, 3),
        include_top=False,
        weights=None
    )
    
    feature_extractor = models.Model(
        inputs=shared_resnet.input,
        outputs=shared_resnet.get_layer("conv3_block3_out").output,
        name="shared_backbone_224"
    )

    def get_features(tensor):
        x = feature_extractor(tensor)
        avg_p = layers.GlobalAveragePooling2D()(x)
        max_p = layers.GlobalMaxPooling2D()(x)
        return layers.Concatenate()([avg_p, max_p])

    feat_tl = get_features(p_tl)
    feat_tr = get_features(p_tr)
    feat_bl = get_features(p_bl)
    feat_br = get_features(p_br)

    merged = layers.Concatenate(name="merge_4_streams")(
        [feat_tl, feat_tr, feat_bl, feat_br]
    )

    x = layers.Dense(256, activation='relu')(merged)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inputs=full_img_input, outputs=output, name="resNet50V2_rgb_v4_4streams")



# --- 1. Custom Random Erasing Layer (The 'Blocker') ---
# This forces the model to learn from multiple parts of the face
def get_erasing_layer():
    return layers.RandomErasing(
        factor=0.5,           # 50% chance to erase
        scale=(0.02, 0.08),   # Small squares to preserve landmarks
        fill_value=0.0,
        name="random_erasing"
    )

def build_ultimate_fasd_v6(input_shape=(224, 224, 3)):
    # Base Input
    full_img_input = layers.Input(shape=input_shape, name="original_input")

    # --- 2. Shared ResNet Backbone (Texture Expert) ---
    # We use conv2 for higher resolution texture features
    base_resnet = tf.keras.applications.ResNet50V2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None # Or 'imagenet' if you want a faster start
    )
    
    # Targeting conv2_block3_out for high-res local artifacts
    inner_resnet = models.Model(
        inputs=base_resnet.input,
        outputs=base_resnet.get_layer("conv2_block3_out").output,
        name="shared_resnet_texture"
    )

    # --- 3. Shared Stream Logic (Including Erasing) ---
    erasing = get_erasing_layer()
    
    def process_stream(tensor, training=True):
        x = erasing(tensor, training=training)
        x = inner_resnet(x)
        # Hybrid Pooling (Max + Avg)
        avg_p = layers.GlobalAveragePooling2D()(x)
        max_p = layers.GlobalMaxPooling2D()(x)
        return layers.Concatenate()([avg_p, max_p])

    # --- 4. Stream Extraction (4 Patches) ---
    def get_patch(x, x1, y1, x2, y2):
        patch = layers.Lambda(lambda img: img[:, y1:y2, x1:x2, :])(x)
        return layers.Resizing(224, 224)(patch)

    # Local Patches (TL, TR, BL, BR)
    feat_tl = process_stream(get_patch(full_img_input, 0, 0, 112, 112))
    feat_tr = process_stream(get_patch(full_img_input, 112, 0, 224, 112))
    feat_bl = process_stream(get_patch(full_img_input, 0, 112, 112, 224))
    feat_br = process_stream(get_patch(full_img_input, 112, 112, 224, 224))

    # --- 5. Global ViT Stream (Consistency Expert) ---
    # We use a simple Transformer encoder block for the global context
    # This checks for unnatural lighting distributions
    def transformer_stream(x):
        # Patchify (8x8 patches)
        p_size = 16
        num_patches = (224 // p_size) ** 2
        x = layers.Conv2D(256, kernel_size=p_size, strides=p_size)(x)
        x = layers.Reshape((num_patches, 256))(x)
        
        # Self-Attention Block
        attn = layers.MultiHeadAttention(num_heads=8, key_dim=256)(x, x)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)
        
        # MLP Block
        mlp = layers.Dense(512, activation='gelu')(x)
        mlp = layers.Dense(256)(mlp)
        x = layers.Add()([x, mlp])
        
        return layers.GlobalAveragePooling1D()(x)

    feat_global = transformer_stream(full_img_input)

    # --- 6. The Fusion Head ---
    merged = layers.Concatenate(name="fusion_layer")(
        [feat_tl, feat_tr, feat_bl, feat_br, feat_global]
    )
    
    # Deep Classifier
    x = layers.Dense(1024, activation='relu')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    output = layers.Dense(1, activation='sigmoid', name="final_decision")(x)

    model = models.Model(inputs=full_img_input, outputs=output, name="FASD_Zero_EER_Candidate")
    return model

def build_pure_vit_fasd(input_shape=(224, 224, 3), patch_size=16, num_layers=8):
    inputs = layers.Input(shape=input_shape)

    # 1. Patchify the image (224/16 = 14x14 = 196 patches)
    # We use a Conv2D layer as a "Patch Encoder"
    projection_dim = 256
    patches = layers.Conv2D(projection_dim, kernel_size=patch_size, strides=patch_size, name="patch_encoder")(inputs)
    
    # Reshape to (Batch, 196, 256)
    num_patches = (input_shape[0] // patch_size) ** 2
    x = layers.Reshape((num_patches, projection_dim))(patches)

    # 2. Positional Embedding (Crucial so ViT knows 'where' the eyes/mouth are)
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    x = x + pos_embedding

    # 3. Transformer Blocks
    for i in range(num_layers):
        # Layer Norm 1 + Multi-Head Attention
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(
            num_heads=8, key_dim=projection_dim // 8, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, x])
        
        # Layer Norm 2 + MLP (Feed Forward)
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(projection_dim)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x = layers.Add()([x3, x2])

    # 4. Global Representation (Class Token equivalent)
    # Instead of a CLS token, we'll use Global Average Pooling for stability
    representation = layers.GlobalAveragePooling1D()(x)
    representation = layers.LayerNormalization(epsilon=1e-6)(representation)

    # 5. Classification Head
    x = layers.Dense(128, activation='relu')(representation)
    x = layers.Dropout(0.5)(x) # High dropout to match your 2% EER issue
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=output, name="Pure_ViT_FASD")
    return model


def build_resnet_vit_hybrid_v7(input_shape=(224, 224, 3)):
    img_input = layers.Input(shape=input_shape)
    
    # 1. Erasing Layer (To force the ViT to look at multiple patches)
    x = layers.RandomErasing(factor=0.3, scale=(0.02, 0.1))(img_input)

    # 2. ResNet Backbone
    base_model = tf.keras.applications.ResNet50V2(input_tensor=x, include_top=False, weights=None)
    mid_features = base_model.get_layer("conv3_block3_out").output # Shape: (28, 28, 512)

    # 3. Lightweight Transformer Block (The "ViT" part)
    # We treat the 28x28 spatial grid as a sequence of 784 tokens
    tokens = layers.Reshape((784, 512))(mid_features)
    
    # Self-Attention allows every part of the face to talk to each other
    attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=128)(tokens, tokens)
    x = layers.Add()([tokens, attn_output])
    x = layers.LayerNormalization()(x)

    # 4. Global Reasoning & Classification
    # This pools the global "attention" into a single vector
    avg_p = layers.GlobalAveragePooling1D()(x)
    max_p = layers.GlobalMaxPooling1D()(x)
    
    # 768 features (256 from avg + 512 from max-like logic, or concat)
    hybrid_feat = layers.Concatenate()([avg_p, max_p])
    
    x = layers.Dense(256, activation='relu')(hybrid_feat)
    x = layers.Dropout(0.4)(x) # Increased dropout to fix your generalization gap
    output = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(inputs=img_input, outputs=output, name="ResNet_ViT_Hybrid_V7")


def build_resnet50v2_rgb_v4_power_classifier(input_shape=(224, 224, 3)):

    img_input = layers.Input(shape=input_shape)
    
    base_model = tf.keras.applications.ResNet50V2(
        input_tensor=img_input,
        include_top=False,
        weights=None 
    )    
    
    mid_features = base_model.get_layer("conv3_block3_out").output

    avg_pool = layers.GlobalAveragePooling2D()(mid_features)
    max_pool = layers.GlobalMaxPooling2D()(mid_features)
    hybrid = layers.Concatenate()([avg_pool, max_pool])

    x = layers.Dense(256, activation='relu', kernel_regularizer='l2')(hybrid) # Added L2 regularization for better generalization
    x = layers.Dropout(0.6)(x) # Higher dropout to prevent overfitting
    x = layers.Dense(64, activation='relu')(x)
    
    output = layers.Dense(1, activation='sigmoid', name="classifier")(x)
    
    model = tf.keras.models.Model(inputs=img_input, outputs=output, name = "resNet50V2_FASD_RGB_V4_Power_Classifier")
    return model

def build_resnet50v2_rgb_v4(input_shape=(224, 224, 3)):

    img_input = layers.Input(shape=input_shape)
    
    base_model = tf.keras.applications.ResNet50V2(
        input_tensor=img_input,
        include_top=False,
        weights=None 
    )    
    
    mid_features = base_model.get_layer("conv3_block3_out").output

    avg_pool = layers.GlobalAveragePooling2D()(mid_features)
    max_pool = layers.GlobalMaxPooling2D()(mid_features)
    hybrid = layers.Concatenate(name = "hybrid_pooling")([avg_pool, max_pool])
    x = layers.Dense(256, activation='relu')(hybrid)

    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=img_input, outputs=output, name = "resNet50V2_FASD_RGB_V4")
    return model

class LGONBPLayer(tf.keras.layers.Layer):
    def __init__(self, P=12, n=3, J=8, **kwargs):
        super().__init__(**kwargs)
        self.P = P
        self.n = n
        self.J = J

    def _get_gpu_histogram(self, data, nbins, min_val, max_val):
        """
        A GPU-compatible histogram replacement using one_hot.
        Works with XLA and JIT compilation.
        """
        # Flatten data per batch item: (Batch, N)
        batch_size = tf.shape(data)[0]
        flattened = tf.reshape(data, [batch_size, -1])
        
        # Scale values to [0, nbins-1]
        scaled_data = (flattened - min_val) / (max_val - min_val + 1e-7)
        indices = tf.cast(scaled_data * tf.cast(nbins - 1, tf.float32), tf.int32)
        indices = tf.clip_by_value(indices, 0, nbins - 1)
        
        # Count occurrences using reduce_sum on one_hot encoding
        # This is the "secret sauce" for GPU histograms
        oh = tf.one_hot(indices, depth=nbins) # (Batch, N, nbins)
        hist = tf.reduce_sum(oh, axis=1)      # (Batch, nbins)
        return hist

    def _get_lgop_features(self, channel):
        img_4d = tf.expand_dims(channel, -1)
        patches = tf.image.extract_patches(
            images=img_4d,
            sizes=[1, 3, 3, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        # Extract neighbors
        neighbors = tf.concat([patches[..., :4], patches[..., 5:]], axis=-1)
        
        # Use GPU Histogram
        return self._get_gpu_histogram(neighbors, nbins=256, min_val=0.0, max_val=255.0)

    def _get_nlbp_features(self, channel):        
        flat_channel = tf.reshape(channel, [tf.shape(channel)[0], -1])
        global_mean = tf.reduce_mean(flat_channel, axis=-1, keepdims=True)
        
        # Center quantization
        center_quantized = tf.where(channel > tf.expand_dims(global_mean, -1), 1.0, 0.0)
        
        # Use GPU Histogram
        return self._get_gpu_histogram(center_quantized, nbins=128, min_val=0.0, max_val=1.0)

    def call(self, inputs):
        hsv = tf.image.rgb_to_hsv(inputs)
        all_hists = []
        
        for i in range(3):
            channel = hsv[..., i]
            all_hists.append(self._get_lgop_features(channel))
            all_hists.append(self._get_nlbp_features(channel))
            
        combined = tf.concat(all_hists, axis=1) 
        return tf.nn.l2_normalize(combined, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1152)

def build_lgonbp_dense_model(input_shape=(224, 224, 3)):
    img_input = tf.keras.layers.Input(shape=input_shape)
    
    # 1. Feature Extraction Layer
    # This replaces the manual SVM input pipeline
    lgonbp_features = LGONBPLayer()(img_input)
    
    # 2. DenseNet-style Classifier Head
    x = tf.keras.layers.Dense(1024)(lgonbp_features)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Residual Dense Block
    shortcut = x
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Add()([x, shortcut])
    
    # Bottleneck and Output
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=img_input, outputs=output, name="LGONBP_Dense_Model")
        
    return model


def build_lgonbp_dense_model_v2(input_shape=(224, 224, 3)):
    img_input = tf.keras.layers.Input(shape=input_shape)
    
    
    lgonbp_features = LGONBPLayer()(img_input)
    
    x = tf.keras.layers.Dense(128, kernel_regularizer='l2')(lgonbp_features)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Dropout(0.6)(x) 
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=img_input, outputs=output)

    return model


"""
resnet50v2_hsv_rgb: A 6-channel input model (RGB+HSV) using a full ResNet50V2 backbone and standard Global Average Pooling (GAP).
(the experiment shows that RGB+HSV is better than RGB alone or HSV alone
and adding label smoothing to the loss function improved the eer slightly
using a batch size of 16 improve the eer compared to batch size of 32)

resnet50v2_hsv_rgb_yuv: A 9-channel input model (RGB+HSV+YUV) using a full ResNet50V2 backbone and standard Global Average Pooling (GAP).
(the experiment shows that adding YUV channels did not improve performance, likely due to redundancy with RGB and HSV)

resnet50v2_hsv_rgb_drop5: Identical to the baseline but increases Dropout to 0.5 to combat the overfitting and improve generalization.
(but the experiment shows 0.3 dropout is better than 0.5 dropout)

resnet50_hsv_rgb: Identical to the baseline but uses the ResNet50 (V1) architecture instead of V2.
(the experiment show that the final layer output of ResNet50V2 is better than ResNet50)

resnet50v2_hsv_rgb_v2: A "Skip-Connection" architecture. It concatenates the final ResNet features with the raw pooled input channels, allowing the classifier to see deep features and global color statistics simultaneously.
(did not perform well, likely due to the classifier being overwhelmed by the combined feature space)

resnet50v2_texture_fusion: An 8-channel fusion model. It adds a Laplacian edge map and Grayscale to the RGB+HSV input. Uses Hybrid Pooling.
(did not perform well compared to the RGB+HSV baseline)

resnet50v2_hsv_rgb_h_lbp: Extract LBP texture specifically from the Hue (H) channel of the HSV space. Targets texture inconsistencies in how color is distributed across the face.
resnet50v2_hsv_rgb_grayscale_lbp: The most robust texture model. It extracts LBP from a Grayscale version of the input, providing the ResNet with a pure "micro-texture" map to detect screen pixels or paper grain.
(resnet50v2_hsv_rgb_grayscale_lbp outperforms resnet50v2_hsv_rgb_h_lbp and give same eer as resnet50v2_hsv_rgb)

resnet50v2_hsv_rgb_v4:use RGB+HSV as input to ResNet50V2, cuts the ResNet at stage 3 (conv3_block3_out) and uses Hybrid Pooling to capture fine-grained textures before they are abstracted away by deeper layers.
resnet50v2_hsv_v4: Uses only HSV as input.
resnet50v2_rgb_v4: Uses RGB instead of HSV as input.
resnet50v2_hsv_rgb_v4: Uses RGB+HSV as input.

the idea of the resnet v4 models series is found in a paper titled " Exploring Hybrid Pooling of Pretrained Residual Network for Face Anti-spoofing "
it suggests that the "sweet spot" for texture-based anti-spoofing is around the mid-level features of ResNet (res3cx) we use conv3_block3_out
and suggests that using a combination of Global Average Pooling and Global Max Pooling (Hybrid Pooling) outperforms using either alone.
note : i trained the model from scrath without the pretrained weights 

(the latest experiments shows that the resnet50v2_rgb_v4 model is the best model)

"""