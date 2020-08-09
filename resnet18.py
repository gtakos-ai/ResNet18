import keras
from keras.models import Model
from keras.layers import Input, merge
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def resnetLayer(inputs,
            out_filters=64,
            in_filters=64,
            kernel_size=3,
            strides=1,
            activation='relu',
            batchNormalization=True,
            residual=True):

    conv1 = Conv2D(out_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))
    conv2 = Conv2D(out_filters,
                  kernel_size=kernel_size,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if batchNormalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    x = conv1(x)
    if batchNormalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    x = conv2(x)
    if residual:
        x = keras.layers.add([x, inputs])
    return x

def resnetBlock(inputs,
            in_filters=64,
            out_filters=64,
            kernel_size=3,
            strides=1,
            activation='relu',
            batchNormalization=True):

    x = inputs
    for i in range(3):
        if (i==0):
            res=False
        else:
            res=True
            strides=1
        x = resnetLayer(inputs=x,
            in_filters=in_filters,
            out_filters=out_filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
            batchNormalization=batchNormalization,
            residual=res)
    return x

def firstLayer(inputs, output_filters=64):
    conv = Conv2D(output_filters, (3,3), padding='same')
    x = conv(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def resNet18(input_shape, numClasses=10):
    output_filters = 16
    depth = 4 # 4*depth + 2 layers
    inputs = Input(shape=input_shape)
    x = firstLayer(inputs, output_filters=output_filters)
    for i in range(depth):
        if (i>0):
            strides=2
        else:
            strides=1
        in_filters = output_filters*strides
        x = resnetBlock(inputs=x, 
            in_filters=in_filters, 
            out_filters=output_filters,
            kernel_size=3,
            strides=strides,
            activation='relu',
            batchNormalization=True)
        output_filters*=2
        
    poolSize = int(input_shape[1]/(2**(depth-1)))   
    x = AveragePooling2D(pool_size=(poolSize,poolSize))(x)
    
    y = Flatten()(x)
    outputs = Dense(numClasses,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

# credit to https://www.kaggle.com/purshipurshi2005 for cifar10 data processing
def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(data_dir, negatives=False):
    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
        cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names
  
            
#data directory           
cifar_10_dir = 'datasets/cifar10/cifar-10-batches-py'
#training and test data
x_train, train_filenames, y_train, x_test, test_filenames, y_test, label_names = load_cifar_10_data(cifar_10_dir)
# Input image dims
input_shape = x_train.shape[1:]
# Data normalization.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255            

CLASSES_NUM = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 50
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, CLASSES_NUM)
y_test = keras.utils.to_categorical(y_test, CLASSES_NUM)
 
callbacks = [
    EarlyStopping(patience=30, verbose=1),
    ReduceLROnPlateau(factor=0.3, patience=5, min_lr=0.000001, verbose=1)]  

model = resNet18(input_shape=input_shape, numClasses=CLASSES_NUM)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=LEARNING_RATE),
              metrics=['accuracy'])
model.summary()

model_history = model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(x_test, y_test),
              callbacks=callbacks,
              shuffle=True)

# plot loss curves
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#plot accuracy curves
acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# plot validation set accuracy only
val_acc = model_history.history['val_accuracy']
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# confusion matrix
y_true = np.argmax(y_test, axis=1)
predicted = model.predict(x_test)
y_pred = np.argmax(predicted, axis=1)
cm = confusion_matrix(y_true, y_pred)
print(cm)
sns.heatmap(np.round(100*cm/np.sum(cm, axis=0)), annot=True)
plt.title("Confusion Matrix (percentages)")
plt.xlabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

# histogram of predicted and actual data
plt.hist(y_pred, bins = CLASSES_NUM)
plt.title("Prediction Histogram")
plt.show()

plt.hist(y_true, bins = CLASSES_NUM)
plt.title("Validation Histogram")
plt.show()
