# Uses mushroom data to create a model. Uses binary classification to determine 
# whether a mushroom is edible or not. Uses neural networks using Keras library.
# Dataset used: https://archive.ics.uci.edu/ml/datasets/Mushroom
# Note: Added CSV lables to dataset just for readability.

# Import libraries
import pandas as pd
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
# print(X)
# print(y)

# Create Keras model using layers
model = Sequential()
model.add(Dense(14, input_dim=22, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit Keras model on dataset
epochsVar = 20
batchSizeVar = 20
model.fit(X, y, epochs=epochsVar, batch_size=batchSizeVar)

# Evaluate Keras model
accuracy = model.evaluate(X, y)
score = accuracy[1]
print('Program completed with Accuracy: %.2f' % (score*100) + '%')
