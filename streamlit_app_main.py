# make streamlit app for practice.ipynb

import streamlit as st
import pandas as pd
import numpy as np
# demo build CNN

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
import pandas as pd
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from tensorflow.keras.models import load_model
from PIL import Image


import warnings
warnings.filterwarnings('ignore')



# GUI

st.title('Data Science Project')
st.write('## Classification images Project')
st.write('### i used CNN algo to solve this problem.')

st.image('streamlit_1_pic.png', width = 500)

# initialising the cnn
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(filters = 32, kernel_size = 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPool2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
classifier.add(MaxPool2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Reading and Preprocessing images

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                      shear_range = 0.2,
                                        zoom_range = 0.2,
                                        horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)  

########################

# assign data to a variable
train_data = 'training_data_tiger_lion'
test_data = 'test_data_tiger_lion'

training_set = train_datagen.flow_from_directory(train_data,
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

test_set = test_datagen.flow_from_directory(test_data,
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# Part 3 - Training the CNN

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [EarlyStopping(monitor='val_loss', patience=5),
                ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]


# history = classifier.fit_generator(training_set,
#                                     epochs = 100,
#                                     validation_data = test_set,
#                                     callbacks = callbacks)

##############GUI###################

# load model
def load_model():
    model = tf.keras.models.load_model('model_tiger_lion_85%.h5')
    return model

# fuction predict

def predict(model, image):
    img = image.resize((64, 64))
    img = np.expand_dims(img, axis=0)
    img = img/255.0
    prediction = model.predict(img)
    return prediction

menu = ['Classification images details', 'make a prediction']

choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Classification images details':
    st.subheader('Classification images:')
    st.write('### i used CNN algo to solve this problem.')
    st.write('#### Image Classification is a fundamental task that attempts to comprehend an entire image as a whole. The goal is to classify the image by assigning it to a specific label. ')
    st.image('streamlit_1_pic.png', width = 500)





elif choice == 'make a prediction':

    st.subheader('make a prediction')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Result: ")
        model = load_model()
        prediction = predict(model, image)
        if prediction[0][0] > 0.5:
            st.write("This is a tiger")
        else:
            st.write("This is a lion")










    # real uploaded_file_1 


        # training_set = train_datagen.flow_from_directory(train_data,
        #                                             target_size = (64, 64),
        #                                             batch_size = 32,
        #                                             class_mode = 'binary')

        # test_set = test_datagen.flow_from_directory(test_data,
        #                                             target_size = (64, 64),
        #                                             batch_size = 32,
        #                                             class_mode = 'binary')




        # # Part 3 - Training the CNN

        # from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

        # callbacks = [EarlyStopping(monitor='val_loss', patience=5),
        #                 ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

        # history = classifier.fit_generator(training_set,
        #                                     epochs = 50,
        #                                     validation_data = test_set,
        #                                     callbacks = callbacks)





