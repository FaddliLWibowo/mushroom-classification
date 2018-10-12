from keras.models import Sequential
from keras.layers import Dense

def build(input_dim):

    model = Sequential()
    model.add(Dense(200, input_dim=input_dim,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(100, activation='relu',kernel_initializer='uniform'))
    model.add(Dense(1, activation='sigmoid',kernel_initializer='uniform'))

    return model