# Implementing a Recurrent Neural Network

# Task 1 - Data Preprocessing

# importing the libraries
# numpy arrays are only allowed as inputs to neural networks in keras and not the pandas data-frames
import numpy as np
# to visualize the results
import matplotlib.pyplot as plt
# for importing and managing the dataset easily
import pandas as pd

# importing the training set
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
# now we select the right column from the "dataset_train" dataset for making the "training_set"
training_set = dataset_train.iloc[:, 1:2].values
# building the the training set numpy array on which the RNN will be trained; we only choose the "Open" column i.e. column 1
# note: if in the above line, we would not have added ".values", then it would just have been a data-frame
# note that iloc method is used to get the right index of the desired columns
# the first parameter in the iloc is the row selector parameter and the second parameter is the column selector parameter

# feature scaling
# feature scaling can be done through standardisation or normalization
# here we use normalization
# note that it is recommended that whenever the sigmoid activation function is involved, use normalization
from sklearn.preprocessing import MinMaxScaler
# create the MinMaxScaler object
sc = MinMaxScaler(feature_range = (0, 1))
# (0, 1) because we want the scaling to be done within this limit
training_set_scaled = sc.fit_transform(training_set)
# fit_transform fits and scales/transforms the data
# note: fit means it will calculate the min() and max() of the data

# creating a data structure with 60 timesteps and 1 output
# this means, the RNN will take into account the previous 60 records to predict one output at a particular time t
# this may be called as the Backpropagation Through Time (BPTT)
# we need to use this concept to avoid over-fitting or wrong predictions
# we are now going to form two entities: X_train and y_train
# X_train will be the input to the neural network
# y_train will be the output to the neural network
# for any financial day X_train will have the 60 records of the previous 60 stock prices
# y_train will have the stock price of the next financial day
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i - 60 : i, 0])
    # 0 means the column 0
    # note that RNN is memorizing at this stage
    y_train.append(training_set_scaled[i : i + 1, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# reshaping the data - the last step in data preprocessing
# we will be adding a new dimension to the data set
# this will help in better predictions
# for adding a new dimension to the numpy array, always use the reshape function
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# the second parameter takes in the batch size (rows), timesteps (columns), and the number of indicatiors/predictors
# in current set (1) - Open stock price column

# Task 2 - Building the RNN

# importing the Keras libraires and packages
from keras.models import Sequential # building the sequence of layers
from keras.layers import Dense # building the hidden layes
from keras.layers import LSTM # building the LSTM layers
from keras.layers import Dropout # for Dropout Regularization - avoiding over-fitting

# initializing the RNN
regressor = Sequential() # regressor because we are predicting a continuous output

# adding the first LSTM layer and Droupout Regularizaiton
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# first parameter is the number of LSTM memory units/cells or neurons, second parameter is the return sequence
# which will be True as we are making a stacked LSTM, the third argument is the input shape
# in third parameter, we will add the last two shapes from X_train as 0th will be added automatically
regressor.add(Dropout(0.20)) # mention the dropout rate for neurons...10 neurons will dropout at each iteration...0.20 * 50 = 10

# adding the second LSTM layer and Droupout Regularizaiton
regressor.add(LSTM(units = 50, return_sequences = True))
# first parameter is the number of LSTM memory units/cells or neurons, second parameter is the return sequence
# which will be True as we are making a stacked LSTM, the third argument is the input shape
# in third parameter, we will add the last two shapes from X_train as 0th will be added automatically
regressor.add(Dropout(0.20)) # mention the dropout rate for neurons...10 neurons will dropout at each iteration...0.20 * 50 = 10

# adding the third LSTM layer and Droupout Regularizaiton
regressor.add(LSTM(units = 50, return_sequences = True)) # first parameter is the number of LSTM memory
# units/cells or neurons, second parameter is the return sequence which will be True as we are making a stacked LSTM,
# the third argument is the input shape
# in third parameter, we will add the last two shapes from X_train as 0th will be added automatically
regressor.add(Dropout(0.20)) # mention the dropout rate for neurons...10 neurons will dropout at each iteration...0.20 * 50 = 10

# adding the fourth LSTM layer and Droupout Regularizaiton
regressor.add(LSTM(units = 50)) # first parameter is the number of LSTM memory units/cells or neurons,
# second parameter is the return sequence which will be True as we are making a stacked LSTM,
# the third argument is the input shape
# in third parameter, we will add the last two shapes from X_train as 0th will be added automatically
regressor.add(Dropout(0.20)) # mention the dropout rate for neurons...10 neurons will dropout at each iteration...0.20 * 50 = 10

# adding the last LSTM layer and Droupout Regularizaiton
regressor.add(Dense(units = 1)) # the parameter is for specifying the output neurons

# compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# rmsprop optimizer is generally used in RNN but adam is too strong
# for classification problem, loss should be say binary_crossentropy

# fitting the RNN to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
# epochs mean how many times the data should be forward propagated and back propagated or say
# show many times the neural network should be trained
# the last parameter specifies the batch size for training - how many records should be trained at a single time
# the training will take some time - around 15 minutes

# Task 3 - predicting and generating visualizaitons

dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_test_set = dataset_test.iloc[:, 1:2].values
# note: for predicting the Jan 2017 stock prices, we need previous 60 days record
dataset_merged = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# for vertical concatenation, we use axis = 0 and gor horizontal, we use axis = 1
inputs = dataset_merged[len(dataset_train) - len(dataset_test) - 60 : ].values
inputs = inputs.reshape(-1, 1)

# scaling the inputs; since fitting has already been done, we will go for just transform
inputs = sc.transform(inputs)
# making the 3D structure for prediction

# forming memory for test set
X_test = []
for i in range(60, 80): # we go till 80 because the number of records in test are 20
    X_test.append(inputs[i - 60 : i, 0]) # 0 means the column 0
X_test= np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# making predictions
pred_test_set = regressor.predict((X_test))
# inversing the scaling to get the actual format
pred_test_set = sc.inverse_transform(pred_test_set)

# plotting the visualizations
plt.plot(real_test_set, color = 'red', label = 'Real Stock Prices from Jan 2017')
plt.plot(pred_test_set, color = 'blue', label = 'Predicted Stock Prices from Jan 2017')
plt.title("Stock Price Prediction")
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

