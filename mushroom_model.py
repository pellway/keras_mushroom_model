# Uses mushroom data to create a model. Uses binary classification to determine 
# whether a mushroom is edible or not. Uses neural networks using Keras library.
# Dataset used: https://archive.ics.uci.edu/ml/datasets/Mushroom
# Note: Added CSV lables to dataset just for readability.

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Load mushroom dataset
dataset = pd.read_csv('dataset/agaricus-lepiota.csv')
df = pd.DataFrame(dataset)

# Convert string data into dummy integer
for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name]= df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes

# # split into input (X) and output (y) variables
X = df.drop(columns=['edible'])
y = df[['edible']].copy()

# Create Keras model using layers
model = Sequential()
model.add(Dense(18, input_dim=22, activation='relu'))
model.add(Dense(6, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compile Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit Keras model on dataset
history = model.fit(X, y, validation_split=0.10, epochs=30, batch_size=10, verbose=0)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("accuracy.png")
plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("loss.png")

# Evaluate Keras model
accuracy = model.evaluate(X, y)
score = accuracy[1]
print('Program completed with Accuracy: %.2f' % (score*100) + '%')
