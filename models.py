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

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
      loss = tf.keras.losses.BinaryCrossentropy(),
      metrics=['accuracy']
  )

  return model

