import os
import csv

samples = []
with open('./data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from random import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            flipped_images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data1/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    images.append(image)
                    images.append(np.fliplr(image))

                center_angle = float(batch_sample[3])
                angles.append(center_angle)
                angles.append(-1 * center_angle)
                angles.append(center_angle + 0.2)
                angles.append(-1 * (center_angle + 0.2))
                angles.append(center_angle - 0.2)
                angles.append(-1 * (center_angle - 0.2))

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #print(len(X_train))
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, BatchNormalization, Activation, Dropout, Cropping2D
model = Sequential()

# model.add(Flatten(input_shape=(160,320,3)))
# model.add(Dense(1))

# # Preprocess incoming data, centered around zero with small standard deviation
# model.add(Lambda(lambda x: x/127.5 - 1.,
#         input_shape=(ch, row, col),
#         output_shape=(ch, row, col)))
# model.add(... finish defining the rest of your model architecture here ...)




# Normalization Layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

# Convolutional Layer 1
model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2)))
#model.add(BatchNormalization())
model.add(Activation('relu'))

# Convolutional Layer 2
model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2)))
#model.add(BatchNormalization())
model.add(Activation('relu'))

# Convolutional Layer 3
model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2)))
#model.add(BatchNormalization())
model.add(Activation('relu'))

# Convolutional Layer 4
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1)))
#model.add(BatchNormalization())
model.add(Activation('relu'))

# Convolutional Layer 5
model.add(Conv2D(filters=100, kernel_size=3, strides=(1, 1)))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.5))

# Flatten Layers
model.add(Flatten())

# Fully Connected Layer 1
model.add(Dense(100))
model.add(Activation('relu'))

# Fully Connected Layer 2
model.add(Dense(50))
model.add(Activation('relu'))

# Fully Connected Layer 3
model.add(Dense(10))
model.add(Activation('relu'))

# Output Layer
model.add(Dense(1))



# steps_per_epoch = samples_per_epoch/batch_size
# `nb_val_samples`->`validation_steps` and `val_samples`->`steps`
batch_size=32
print(len(samples),len(train_samples),len(train_samples[0]), batch_size, len(train_samples)/batch_size)
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch = (len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size, epochs=3)

model.save('model.h5')
"""
If the above code throw exceptions, try
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
"""
