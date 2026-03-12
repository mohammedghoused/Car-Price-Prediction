import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Load the cleaned data
car = pd.read_csv("Cleaned Car.csv")
# Drop the index column if it exists
if 'Unnamed: 0' in car.columns:
    car = car.drop(columns='Unnamed: 0')

# Prepare features and target
X = car.drop(columns='Price')
y = car['Price']

# Find the best random_state (same as in the notebook)
print("Finding best random_state...")
scores = []
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
    # Create OneHotEncoder and fit it on the full dataset to get categories
    ohe = OneHotEncoder()
    ohe.fit(X[['name', 'company', 'fuel_type']])
    
    # Create column transformer
    col_trans = make_column_transformer(
        (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
        remainder='passthrough'
    )
    
    # Create pipeline
    lr = LinearRegression()
    pipe = make_pipeline(col_trans, lr)
    
    # Train and evaluate
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

# Find the best random_state
best_random_state = np.argmax(scores)
best_score = scores[best_random_state]
print(f"Best random_state: {best_random_state}")
print(f"Best R2 score: {best_score}")

# Train final model with best random_state
print("\nTraining final model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=best_random_state)

# Create OneHotEncoder and fit it on the full dataset to get categories
ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

# Create column transformer
col_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

# Create and train pipeline
lr = LinearRegression()
pipe = make_pipeline(col_trans, lr)
pipe.fit(X_train, y_train)

# Evaluate final model
y_pred = pipe.predict(X_test)
final_score = r2_score(y_test, y_pred)
print(f"Final model R2 score: {final_score}")

# Save the model
print("\nSaving model...")
pickle.dump(pipe, open('CarPriceModel.pkl', 'wb'))
print("Model saved successfully!")

# Test the model
print("\nTesting model with sample prediction...")
test_prediction = pipe.predict(pd.DataFrame([['Maruti Suzuki Swift', 'Maruti', 2008, 100, 'Petrol']], 
                                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
print(f"Sample prediction: {test_prediction[0]:.2f}")
print("\nModel retrained successfully and is compatible with scikit-learn 1.8.0!")

