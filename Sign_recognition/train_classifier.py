import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
import time

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

filtered_data = [x for x in data_dict['data'] if len(x) == 42]
filtered_labels = [label for i, label in enumerate(data_dict['labels']) if len(data_dict['data'][i]) == 42]

data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the model
model = RandomForestClassifier()

# Measure training time
start_time = time.time()
model.fit(x_train, y_train)
training_time = time.time() - start_time

# Predict on the test set
y_predict = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predict)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict)

# Calculate classification report (includes precision, recall, F1 score)
class_report = classification_report(y_test, y_predict)

# Calculate F1 score
f1 = f1_score(y_test, y_predict, average='weighted')

# Calculate precision
precision = precision_score(y_test, y_predict, average='weighted')

# Calculate recall
recall = recall_score(y_test, y_predict, average='weighted')

# Print the results
print('{}% of samples were classified correctly!'.format(accuracy * 100))
print('Training time: {:.4f} seconds'.format(training_time))
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)
print('F1 Score: {:.4f}'.format(f1))
print('Precision: {:.4f}'.format(precision))
print('Recall: {:.4f}'.format(recall))

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()

# Save the model as model1.p
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
