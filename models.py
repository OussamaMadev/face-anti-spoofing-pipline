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