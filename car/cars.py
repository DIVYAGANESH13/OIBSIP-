import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
filepath=(r"C:\Users\Jayap\OneDrive\Documents\Desktop\car\car data.csv")
data = pd.read_csv(filepath)
features = data[['Year', 'Present_Price', 'Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner']]
target = data['Selling_Price']
label_encoder = LabelEncoder()
features.loc[:, 'Fuel_Type'] = label_encoder.fit_transform(features['Fuel_Type'])
features.loc[:, 'Selling_type'] = label_encoder.fit_transform(features['Selling_type'])
features.loc[:, 'Transmission'] = label_encoder.fit_transform(features['Transmission'])
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
new_car_features = [2022, 15000, 50000, 'Petrol', 'Dealer', 'Manual', 1]
new_car_features[3] = label_encoder.transform([new_car_features[3]])[0] if 'Petrol' in label_encoder.classes_ else 0  
new_car_features[4] = label_encoder.transform([new_car_features[4]])[0] if 'Dealer' in label_encoder.classes_ else 0  
new_car_features[5] = label_encoder.transform([new_car_features[5]])[0]  

scaled_features = scaler.transform([new_car_features])
predicted_price = model.predict(scaled_features)

print(f'Predicted Price for the New Car: {predicted_price[0]}')
import matplotlib.pyplot as plt

plt.scatter(y_test, predictions)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Prices (Linear Regression)")
plt.show()
coefficients = model.coef_
feature_names = X_train.columns

plt.barh(feature_names, coefficients)
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Feature Coefficients (Linear Regression)")
plt.show()
