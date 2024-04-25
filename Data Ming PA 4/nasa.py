import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset from a CSV file directly
df = pd.read_csv('nasa.csv')

features = df.drop(['Hazardous'], axis=1)
target = df['Hazardous']

# Convert categorical data to numeric using get_dummies
features = pd.get_dummies(features)

# Normalize the feature data since KNN is sensitive to the magnitude of data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Splitting the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Initialize and train the KNeighborsClassifier with the optimal k value (found previously or assume a value)
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Compute the confusion matrix and print it
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Generate and print the classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

# Extract and print detailed evaluation metrics
tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives (Correct Non-Hazardous): {tn}")
print(f"False Positives (Incorrectly Hazardous): {fp}")
print(f"False Negatives (Hazardous Missed): {fn}")
print(f"True Positives (Correct Hazardous): {tp}")

# Parsing the classification report (simplified for direct lines)
lines = report.split('\n')
print("\nDetailed Metrics:")
for line in lines[2:4]:
    print(line)
