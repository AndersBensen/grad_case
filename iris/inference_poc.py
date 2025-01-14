import pickle

# Constants
FILENAME = "./iris_model.pkl"
TARGET_NAMES = ['setosa', 'versicolor', 'virginica']

# Load the model
with open(FILENAME, 'rb') as file:
    iris_model = pickle.load(file)

# Input features - should correspond to a setosa
sepal_length = 5.1
sepal_width = 3.5
petal_length = 3.1
petal_width = 0.2

# Prepare input sample
input_sample = [[sepal_length, sepal_width, petal_length, petal_width]]

# Make a prediction
prediction = iris_model.predict(input_sample)
predicted_target = TARGET_NAMES[prediction[0]]

# Output the prediction
print("Predicted flower:", predicted_target)