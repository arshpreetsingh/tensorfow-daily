import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.optimizers import RMSprop
base_dir = './cats_and_dogs_filtered'
train_dir = os.path.join(base_dir,'train')
validate_dir = os.path.join(base_dir,'validation')

train_cat_dir = os.path.join(train_dir,'cats')
train_dog_dir = os.path.join(train_dir,'dogs')

test_cat_dir = os.path.join(validate_dir,'cats')
test_dog_dir = os.path.join(validate_dir,'dogs')

train_data_gen = ImageDataGenerator(rescale=1/255.0)
test_data_gen =  ImageDataGenerator(rescale=1.0/255.0)

model=tf.keras.models.Sequential([
    # this is first layer of network called input layer.
    tf.keras.layers.Conv2D(64,(3,3),input_shape=(150,150,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #layer 2
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #layer3
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #layer4
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.summary()
model.compile(optimizer=RMSprop(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])




train_gen= train_data_gen.flow_from_directory(train_dir,batch_size=10,class_mode='binary',target_size=(150,150))

test_gen = test_data_gen.flow_from_directory(validate_dir,batch_size=10,class_mode='binary',target_size=(150,150))


model.fit(train_gen,validation_data=test_gen,
                    epochs=15,steps_per_epoch=  100,verbose=2, validation_steps=15)