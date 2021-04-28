# -*- coding: utf-8 -*-

#from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout

from parse_data import preprocess_data


# Multilayer perceptron (MLP) for binary classification:
def create_model():
    # First we initialize the model. "Sequential" means there are no loops. 
    model = Sequential()

    # Add layers one at the time. Each with 64 nodes. 
    model.add(Dense(64, input_dim=500, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation = 'relu')) 
    model.add(Dropout(0.25))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation = 'sigmoid'))

    # Compile the keras model using 'accuracy' to measure model performance
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    x_train, y_train, x_test = preprocess_data()

    #KerasClassifier(build_fn=create_model, verbose=10)
    model = create_model()

    # Train the model, takes a few minutes
    model.fit(x_train, y_train, epochs=50, batch_size=20)

    _, accuracy = model.evaluate(x_train, y_train)
    pred_prob = model.predict(x_test)

    print('Accuracy: %.2f' % (accuracy*100))

