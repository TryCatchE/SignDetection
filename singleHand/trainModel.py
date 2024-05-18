import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the pickle file
dataDir = pickle.load(open('./data.pickle', 'rb'))
data = dataDir['data']

# Ensure that each data point has the same length
for i in range(len(data)):
    if len(data[i]) != 42:
        # Trim the list to the first 42 elements
        data[i] = data[i][:42]

# Convert the data to a NumPy array
try:
    data_array = np.array(data)
    print("Conversion successful.")
except ValueError as e:
    print("Error during conversion:", e)

# Convert labels to NumPy array
labels = np.asarray(dataDir['labels'])

# Split data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize Random Forest Classifier
model = RandomForestClassifier()

# Train the model
model.fit(xTrain, yTrain)

# Predict labels for test data
y_predict = model.predict(xTest)

# Calculate accuracy
score = accuracy_score(y_predict, yTest)
print('{}% of data classified correctly !'.format(score * 100))

# Save the trained model
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

# Visualize confusion matrix
cm = confusion_matrix(yTest, y_predict)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Visualize class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=labels)
plt.title('Class Distribution')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.show()

# Generate learning curve
train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), xTrain, yTrain, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

# Calculate mean and standard deviation of training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(yTest, y_predict))
