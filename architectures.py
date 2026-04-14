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