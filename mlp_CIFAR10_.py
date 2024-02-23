# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:00:52 2024

@author: Jungyu Lee, 301236221

Assignment 2 Exercise 2
"""

from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

class_names = ['airplane', 'automobile', 'bird', 
               'cat', 'deer', 'dog', 'frog', 
               'horse', 'ship', 'truck']

# batch normalization, dropout
""" early stop 75/100 epochs sgd"""
""" | 69 / 54 / 54 """
model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(512, activation='relu'), # kernel_regularizer 
    layers.Dropout(0.5),  
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])

""" early stop 97/100 epochs sgd"""
""" | 62 / 57 / 57 """
# model = models.Sequential([
#     layers.Flatten(input_shape=(32, 32, 3)),
#     layers.Dense(512, activation='relu', kernel_initializer="lecun_normal"), # kernel_regularizer 
#     layers.Dropout(0.5),  
#     layers.Dense(256, activation='relu', kernel_initializer="lecun_normal"),
#     layers.Dense(10, activation='softmax')
# ])

""" 50/100 adamax """
""" 62 / 56 / 57 """
# model = models.Sequential([
#     layers.Flatten(input_shape=(32, 32, 3)),
#     layers.Dense(512, activation='elu', kernel_initializer="lecun_normal"), # kernel_regularizer 
#     layers.Dropout(0.3),  
#     layers.Dense(256, activation='elu', kernel_initializer="lecun_normal"),
#     layers.Dropout(0.3),
#     layers.Dense(10, activation='softmax')
# ])

# compile the model
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',   
    min_delta=0.001,      
    patience=10,          
    restore_best_weights=True 
)

# train the model / validation 0.8
history = model.fit(x_train, y_train, epochs=100, batch_size=32
                    , validation_split=0.08, callbacks=[early_stopping])  

import pandas as pd

# gap will be higher if it overfits the train data
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 3) # set the vertical range to [0-3]
plt.show()

model.evaluate(x_test, y_test)

# make predictions
predictions = model.predict(x_test)

# plot predictions with their labels
plt.figure(figsize=(10, 10))
for i in range(10):  
    plt.subplot(5, 2, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    actual = y_test[i][0]
    predicted = np.argmax(predictions[i])
    color = 'green' if actual == predicted else 'red'
    plt.xlabel(f'Actual: {class_names[actual]}, Predicted: {class_names[predicted]}', color=color)
plt.tight_layout()
plt.show()