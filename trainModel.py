import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


dataDir = pickle.load(open('./data.pickle', 'rb'))


data = np.asarray(dataDir['data'])
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
