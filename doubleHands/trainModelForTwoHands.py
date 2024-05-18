import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve

# Load data from pickle file
dataDir = pickle.load(open('./data2.pickle', 'rb'))

# Extract data and labels
data = np.asarray(dataDir['data'])
labels = np.asarray(dataDir['labels'])

# Split data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(xTrain, yTrain)

# Predict on the test set
y_predict = model.predict(xTest)

# Calculate accuracy
score = accuracy_score(y_predict, yTest)

# Print accuracy
print('{}% of data classified correctly !'.format(score * 100))

# Save the trained model
# Change the model name 
f = open('model2.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

# Countplot to visualize the distribution of labels in the dataset
plt.figure(figsize=(8, 6))
sns.countplot(x=labels)
plt.title('Distribution of Labels')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Learning curve to visualize the model's learning progress
train_sizes, train_scores, test_scores = learning_curve(model, xTrain, yTrain, cv=5, scoring='accuracy')
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='Training Accuracy', color='blue')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
plt.plot(train_sizes, test_mean, label='Cross-validation Accuracy', color='green')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='green', alpha=0.2)
plt.title('Learning Curve')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(yTest, y_predict))

# Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(yTest, y_predict), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
