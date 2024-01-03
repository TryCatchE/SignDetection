import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


dataDir = pickle.load(open('./data.pickle', 'rb'))
data = dataDir['data']

for i in range(len(data)):
    if len(data[i]) != 42:
        # Trim the list to the first 42 elements
        data[i] = data[i][:42]

# Now try converting to a NumPy array again
try:
    data_array = np.array(data)
    print("Conversion successful.")
except ValueError as e:
    print("Error during conversion:", e)
# Print out the first few lengths for debugging
# exit()

# data = np.asarray(dataDir['data'])
labels = np.asarray(dataDir['labels'])



xTrain, xTest, yTrain, yTest = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(xTrain, yTrain)

y_predict = model.predict(xTest)

score = accuracy_score(y_predict, yTest)

print('{}% of data classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
