import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('nasa.csv')

# Data Analysis
print("Dataset Overview:")
print(df.info())
print("\nDataset Description:")
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Data Cleaning
columns_to_drop = ['Neo Reference ID', 'Name', 'Close Approach Date', 'Orbit Determination Date', 'Orbiting Body', 'Equinox']
df_clean = df.drop(columns=columns_to_drop)

# Convert categorical columns to numeric using get_dummies
df_clean = pd.get_dummies(df_clean)

# Prepare data for classification
target = df_clean['Hazardous']
features = df_clean.drop('Hazardous', axis=1)

# Normalize the feature data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)

# KNN Model Evaluation
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

# Calculate and print accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy}")
