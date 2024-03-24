import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate sample data
np.random.seed(42)
num_machines = 500
num_years = 3
num_records = num_machines * num_years * 365
machine_ids = np.random.randint(1, num_machines + 1, size=num_records)
maintenance_data = pd.DataFrame({
    'machine_id': machine_ids,
    'maintenance_date': pd.date_range(start='2019-01-01', periods=num_records),
    'sensor_data': np.random.randn(num_records),
    'failure': np.random.randint(0, 2, size=num_records)  # Simulated binary failure (0: no failure, 1: failure)
})

# Split data into features and target
X = maintenance_data[['sensor_data']]
y = maintenance_data['failure']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"Random Forest Classifier Accuracy: {rf_accuracy:.2f}")

# Train a simple neural network using TensorFlow
model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
nn_loss, nn_accuracy = model.evaluate(X_test, y_test)

print(f"Neural Network Classifier Accuracy: {nn_accuracy:.2f}")

# Calculate cost and uptime improvements
maintenance_cost_reduction = 0.3
uptime_increase = 0.2
maintenance_cost_savings = maintenance_cost_reduction * total_maintenance_costs
uptime_increase_percentage = uptime_increase * 100

print(f"Maintenance Cost Reduction: {maintenance_cost_savings:.2f}")
print(f"Uptime Increase Percentage: {uptime_increase_percentage:.2f}%")
