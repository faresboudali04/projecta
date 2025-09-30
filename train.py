import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.DataFrame({"x":[1,2,3,4,5],"y":[2,4,6,8,10]})
X = data[["x"]]
y = data["y"]

model = LinearRegression().fit(X, y)
joblib.dump(model, "model.pkl")
print("Model trained and saved!")
