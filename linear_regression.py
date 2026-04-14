# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load Dataset
df = pd.read_csv(r"C:\Users\HP\Documents\archive\Housing.csv")

# Step 3: Basic Exploration
print(df.head())
print(df.info())
print(df.isnull().sum())

# Step 4: Handle Missing Values 
# Fill missing values with mean of each column) 
df = df.fillna(df.mean(numeric_only=True))

# Step 5: Feature Selection
X = df[['area']]   # simple regression 
y = df['price']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Prediction
y_pred = model.predict(X_test)

# Step 9: Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
plt.scatter(X_test['area'], y_test)
print("R2 Score:", r2)

# Step 10: Plot Regression Line
plt.scatter(X_test, y_test)

# Sort X_test and y_pred 
sorted_indices = X_test['area'].argsort()
X_test_sorted = X_test.iloc[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]
# Plot the regression line
plt.plot(X_test_sorted['area'], y_pred_sorted, color='red')
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])  
plt.title("Linear Regression")
plt.show()