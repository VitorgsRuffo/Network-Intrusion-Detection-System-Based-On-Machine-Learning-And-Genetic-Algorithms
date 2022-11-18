import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


#loading data:
dataset = pd.read_csv('./CICDDoS2019.reduced.csv')
#print(dataset)

#binarizing labels:

dataset.loc[dataset[' Label'] != 'BENIGN', ' Label'] = 1
dataset.loc[dataset[' Label'] == 'BENIGN', ' Label'] = 0
labels = dataset[' Label']
del dataset[' Label']


#fixing wrong values;

del dataset[' Source IP'] #wrong column
dataset.replace('N','0')


#converting to array.
numpy_dataset = dataset.values
numpy_dataset = np.asarray(numpy_dataset).astype(np.float32)

numpy_labels = labels.values
numpy_labels = np.asarray(numpy_labels).astype(np.float32)



#fixing more wrong values:
numpy_dataset=np.nan_to_num(numpy_dataset, nan=0,posinf=0, neginf=0)


#scaling
scaler = MinMaxScaler()
scaler.fit(numpy_dataset)
scaled_numpy_dataset = scaler.transform(numpy_dataset)

train_x, test_x, train_y, test_y = train_test_split(scaled_numpy_dataset, numpy_labels, train_size = 0.66, random_state=77)


#saving data
train_x = np.array(train_x)
np.savetxt("train_x.txt", train_x)

test_x = np.array(test_x)
np.savetxt("test_x.txt", test_x)

train_y = np.array(train_y)
np.savetxt("train_y.txt", train_y)

test_y = np.array(test_y)
np.savetxt("test_y.txt", test_y)
