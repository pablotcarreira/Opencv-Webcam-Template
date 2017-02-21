
import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import mnist


# the data, shuffled and split between train and test sets
(X_train_orig, y_train), (X_test_orig, y_test) = mnist.load_data()

k = nb_classes = 10
(n,height,width) = X_train_orig.shape  # deve ser 60000 amostras de 28 x 28 pixels
m = input_dim = height * width
X_train = X_train_orig.reshape(n, input_dim)/255.
X_test = X_test_orig.reshape(10000, input_dim)/255.

X_train = X_train[:500]
y_train = y_train[:500]
X_test  = X_test[:100]
y_test  = y_test[:100]

from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
output_dim = nb_classes = 10
model = Sequential()
model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
batch_size = 500
nb_epoch = 1000

sgd = SGD(lr= 0.5, decay=0e-6, momentum=0., nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=0, validation_data=(X_test, Y_test))
loss = model.evaluate(X_test, Y_test, verbose=0)
print('loss:', loss)

y_hat = model.predict_classes(X_test)

print(y_hat.shape)
print('y_hat: ', y_hat[:20])
print('y_test:', y_test[:20])
print(y_test.shape)
pd.crosstab(y_hat, y_test)

model.save("model_mnist")


