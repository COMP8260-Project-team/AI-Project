
# save the final model to file
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow import keras
from tensorflow.keras import layers
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
 
scores, histories = list(), list()

#load fashion dataset
def fashion_load_dataset():
    # load dataset
    (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
    return train_X, train_y, test_X, test_y
 

# load train and test dataset
def mnist_load_dataset():
    # load dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    return train_images, train_labels, test_images, test_labels
 
# scale pixels
def prep_pixels(train, test):
    train_images = train.reshape((60000, 28, 28, 1))
    test_images = test.reshape((10000, 28, 28, 1))
    
     # convert from integers to floats
    # normalize to range 0-1
    train_norm = train_images.astype("float32") / 255
    test_norm = test_images.astype("float32") / 255

    # return normalized images
    return train_norm, test_norm
 
# define cnn model
def define_model(n):
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(16,3,activation='relu')(inputs)
    x = layers.Conv2D(32,3,activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    if(n==2 or n==4):
        x = layers.Conv2D(64,3,activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    
    if(n==3 or n==4):
        first = layers.Dense(126, activation='relu')(x)
        outputs = layers.Dense(10,activation='softmax')(first)
    else:
        outputs = layers.Dense(10,activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    #print(model.summary())

    # compile model
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model


def evaluate_model(dataX, dataY, tdataX, tdataY,n):
    global scores,histories
    # define model
    model = define_model(n)
    # fit model
    history = model.fit(dataX, dataY, epochs=2, batch_size=32)
    # evaluate model
    _, acc = model.evaluate(tdataX, tdataY)
    print('> %.3f' % (acc * 100.0))
    # append scores
    scores.append(acc)
    histories.append(history)
#     return scores, histories
 
# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
#         pyplot.subplot(211)
#         pyplot.title('Cross Entropy Loss')
#         pyplot.plot(histories[i].history['loss'], color='blue', label='train')
#         pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(2,1,1)
        pyplot.title('Classification Training Accuracy on model ')
        print("accuracy")
        print(histories[i].history['accuracy'])
        pyplot.plot(histories[i].history['accuracy'])
#         pyplot.legend()
        pyplot.show()
    
# evaluate a model using k-fold cross-validation
# def evaluate_model(dataX, dataY, n_folds=5):
#     scores, histories = list(), list()
#     # prepare cross validation
#     kfold = KFold(n_folds, shuffle=True, random_state=1)
#     # enumerate splits
#     for train_ix, test_ix in kfold.split(dataX):
        
#         # define model
#         model = define_model()
#         # select rows for train and test
#         trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
#         # fit model
#         history = model.fit(trainX, trainY, epochs=5, batch_size=32, validation_data=(testX, testY))
#         # evaluate model
#         _, acc = model.evaluate(testX, testY)
#         print('> %.3f' % (acc * 100.0))
#         # append scores
#         scores.append(acc)
#         histories.append(history)
#     return scores, histories
 
# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # box and whisker plots of results
    pyplot.plot(scores)
    pyplot.show()
 
# run the test harness for evaluating a model
def run_test_harness():
    for i in range(1,2):
        # load dataset
        #trainX, trainY, testX, testY = mnist_load_dataset()
        trainX, trainY, testX, testY = fashion_load_dataset()
        # prepare pixel data
        trainX, testX = prep_pixels(trainX, testX)
        # evaluate model
        evaluate_model(trainX, trainY,testX,testY,i)
    # learning curves
#     summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)
 
# entry point, run the test harness
run_test_harness()


