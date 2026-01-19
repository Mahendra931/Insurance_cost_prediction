import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("data/insurance.csv")

df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df = pd.get_dummies(df, columns=['region'], drop_first=True)

X = df.drop('charges', axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# model = LinearRegression()
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# joblib.dump(model, "model/linear_regression.pkl")
# print("Model saved successfully!")

joblib.dump(model, "model/random_forest.pkl")

print("Random Forest model saved successfully!")



from sklearn.metrics import r2_score, mean_absolute_error

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# row = X.iloc[[0]]
# print("Actual:", y.iloc[0])
# print("Predicted:", model.predict(row)[0])

