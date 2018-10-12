import pandas as pd
import NN
import matplotlib.pyplot as plt
import os
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df = pd.read_csv("mushrooms.csv")

headers = list(df)
headers.remove('class')
new_list = headers

df['class']=df['class'].astype('category')
df['class'] = df['class'].cat.codes

df = pd.get_dummies(df, columns=new_list)

X = df.iloc[:, df.columns != 'class'].values
y = df['class'].values

input_dim=X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)  

EPOCHS = 32
LR = 0.001
BS = len(X_test) // 100

model = NN.build(input_dim)

model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])

history=model.fit(x=X_train,y=y_train,verbose=1,epochs=EPOCHS,batch_size=BS,validation_split=0.2)

y_pred = model.predict_classes(X_test)

cm = confusion_matrix(y_test, y_pred)

accuracy = model.evaluate(X_test,y_test)[1] * 100


print("The accuracy of this model is: " +'{}'.format(round(accuracy,2)))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
