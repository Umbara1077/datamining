# Collaborated with Dante Corso, Gino Costanzo

import pandas as pd
import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)

    def calculate_likelihood(self, x, mean, stdev):
        exponent = np.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent

    def calculate_class_probabilities(self, input_data):
        probabilities = {}
        for class_value in self.classes:
            class_indices = np.where(self.y_train == class_value)[0]
            class_data = self.X_train[class_indices]

            # Calculate mean and standard deviation for each feature
            class_means = np.mean(class_data, axis=0)
            class_stdevs = np.std(class_data, axis=0)

            # Calculate likelihood for each feature
            likelihood = np.prod(self.calculate_likelihood(input_data, class_means, class_stdevs))

            # Calculate prior probability
            prior = len(class_indices) / len(self.y_train)

            # Calculate class probability using Bayes' theorem
            probabilities[class_value] = prior * likelihood

        return probabilities

    # Use it for further predicting model
    def predict(self, X):
        predictions = []
        for sample in X:
            probabilities = self.calculate_class_probabilities(sample)
            predicted_class = max(probabilities, key=probabilities.get)
            predictions.append(predicted_class)
        return predictions

# Data Frame example

data = [['Blue','Blue','Brown','Yes'],['Red','Blue','Brown','Yes'],
        ['Red','Brown','Black','No'],['Red','Green','Brown','No'],
        ['Green','Green','Blonde','Yes'],['Blue','Brown','Blonde','No']]


df = pd.DataFrame( data, columns = ['Favorite Color', 'Eye Color', 'Hair Color', 'Happy'] )

thing1 = sum(map(lambda x: 1 if 'Red' in x else 0, df[['Favorite Color']].values)) 
thing2 = sum(map(lambda x: 1 if 'Blue' in x else 0, df[['Eye Color']].values)) 
thing3 = sum(map(lambda x: 1 if 'Brown' in x else 0, df[['Hair Color']].values))
use = sum(map(lambda x: 1 if 'Yes' in x else 0, df[['Yes']].values))

features = thing1, thing2, thing3 # df[['Favorite Color','Eye Color','Hair Color']].values 
target = use # df['Happy'].values

# Initialize the Naive Bayes class
nb = NaiveBayes()

# Use methods in class
nb.fit(features, target)

y_pred = nb.predict(features)

accuracy = np.mean(y_pred == target)
print("Accuracy:", accuracy)
