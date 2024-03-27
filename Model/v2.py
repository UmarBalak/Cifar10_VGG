from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, DepthwiseConv2D, SeparableConv2D, GlobalAveragePooling2D
from keras import optimizers
from keras.regularizers import l2
import numpy as np
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt

class OptimizedCIFAR10CNN:
    def __init__(self, train=True):
        self.num_classes = 10
        self.weight_decay = 1e-4
        self.x_shape = [32, 32, 3]

        self.model_1 = self.build_model()
        if train:
            self.model_1 = self.train(self.model_1)
        else:
            self.model_1.load_model('optimized_cifar10_test_2.keras')

    def store_training_history(self, epoch, history, history_df=None):
      if history_df is None:
          history_df = pd.DataFrame(columns=['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr', 'time_per_epoch', 'time_per_step'])

      history_df = history_df.append({
          'epoch': epoch,
          'loss': history.history['loss'][0],
          'accuracy': history.history['accuracy'][0],
          'val_loss': history.history['val_loss'][0],
          'val_accuracy': history.history['val_accuracy'][0],
          'lr': history.history['lr'][0],
      }, ignore_index=True)

      return history_df

    def plot_training_history(self, history):
        """
        Plot the training history including training loss, validation loss,
        training accuracy, and validation accuracy over epochs.

        Parameters:
            history (dict): History object returned by model.fit().
        """
        plt.figure(figsize=(12, 6))

        # Plot training & validation loss values
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper right')

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='lower right')

        plt.tight_layout()
        plt.show()


    def build_model(self):
        model_1 = Sequential()
        weight_decay = self.weight_decay

        model_1.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                         input_shape=self.x_shape))
        model_1.add(Activation('relu'))
        model_1.add(BatchNormalization())
        model_1.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model_1.add(Activation('relu'))
        model_1.add(BatchNormalization())
        model_1.add(MaxPooling2D(pool_size=(2, 2)))
        model_1.add(Dropout(0.25))

        model_1.add(Conv2D(128, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model_1.add(Activation('relu'))
        model_1.add(BatchNormalization())
        model_1.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model_1.add(Activation('relu'))
        model_1.add(BatchNormalization())
        model_1.add(Conv2D(256, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model_1.add(Activation('relu'))
        model_1.add(BatchNormalization())
        model_1.add(MaxPooling2D(pool_size=(2, 2)))
        model_1.add(Dropout(0.25))

        model_1.add(Flatten())
        model_1.add(Dense(1024, kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model_1.add(Activation('relu'))
        model_1.add(BatchNormalization())
        model_1.add(Dropout(0.5))
        model_1.add(Dense(self.num_classes))
        model_1.add(Activation('softmax'))

        return model_1


    # Normalize and other functions remain the same as in the initial code
    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model_1.predict(x,batch_size)


    def train(self, model_1):
        model_name = 'optimized_cifar10_v1'
        batch_size = 128 # original 64
        max_epochs = 75 # original 100
        learning_rate = 0.001 # original 0.01 , good - 0.005
        lr_decay = 1e-6 # 1e-6
        lr_drop = 20

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                              patience=2, min_lr=0.0001)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   start_from_epoch=5,
                                                   patience=3)



        datagen = ImageDataGenerator(
            brightness_range=[0.4,1.8], # extra
            fill_mode='nearest', # extra
            rotation_range=10,
            width_shift_range=0.1, # original 0.1
            height_shift_range=0.1, # original 0.1
            horizontal_flip=True
        )
        datagen.fit(x_train)

        # sgd = optimizers.legacy.SGD(learning_rate=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

        model_1.compile(loss='categorical_crossentropy', optimizer=optimizers.legacy.Adam(learning_rate = learning_rate), metrics=['accuracy'])

        history = model_1.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                                      steps_per_epoch=x_train.shape[0] // batch_size,
                                      epochs=max_epochs,
                                      validation_data=(x_test, y_test),
                                      callbacks=[reduce_lr], # + early stoping
                                      verbose=2)

        # Plot training history
        self.plot_training_history(history)

        # Store training history
        history_df = self.store_training_history(max_epochs, history)

        # Save training history to a CSV file
        history_df.to_csv(f"{model_name}.csv", index=False)

        model_1.save(f"{model_name}.keras")
        return model_1, history_df

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Train the model
    model_1, hist_df = OptimizedCIFAR10CNN()

    predicted_x = model_1.predict(x_test)
    residuals = np.argmax(predicted_x, 1) != np.argmax(y_test, 1)

    loss = sum(residuals) / len(residuals)
    print("The validation 0/1 loss is: ", loss)