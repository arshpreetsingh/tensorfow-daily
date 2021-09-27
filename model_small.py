import os
import tensorflow as tf
#from tensorflow.keras.optmizers import RMSprop
from tensorflow.keras.optimizers import RMSprop

base_dir = './cats_and_dogs_filtered'
train_dir = os.path.join(base_dir,'train')
validate_dir = os.path.join(base_dir,'validation')

train_cat_dir = os.path.join(train_dir,'cats')
train_dog_dir = os.path.join(train_dir,'dogs')

test_cat_dir = os.path.join(validate_dir,'cats')
test_dog_dir = os.path.join(validate_dir,'dogs')

model=tf.keras.models.Sequential([
    # this is first layer of network called input layer.
    tf.keras.layers.Conv2D(64,(3,3),input_shape=(300,300,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #layer 2
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #layer3
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #layer4
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.summary()
model.compile(optimizer=RMSprop(learning_rate=0.001),loss='binary_cross_entropy',metrics=['accuracy'])




