# =========================================================
# HOUSE PRICE PREDICTION (INDIA)
# Linear Regression + Matplotlib Graphs
# Prices in Indian Rupees (₹)
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. HELPER FUNCTIONS
# =========================================================

def inr(value):
    """Format price in Indian Rupees."""
    sign = "-" if value < 0 else ""
    value = abs(value)

    if value >= 1e7:
        return f"{sign}₹{value / 1e7:.2f} Cr"
    elif value >= 1e5:
        return f"{sign}₹{value / 1e5:.2f} Lakh"
    else:
        return f"{sign}₹{value:,.0f}"


def lakh_axis(x, pos):
    """Format graph axis values in Lakh."""
    return f"₹{x / 1e5:.0f}L"


# =========================================================
# 2. LOAD DATASET
# =========================================================

df = pd.read_csv("Housing.csv")


# =========================================================
# 3. RENAME COLUMNS
# =========================================================

df = df.rename(columns={
    "area": "area_sqft",
    "bedrooms": "bhk",
    "stories": "floors",
    "parking": "car_parking",
    "mainroad": "main_road_facing",
    "guestroom": "guest_room",
    "hotwaterheating": "geyser",
    "airconditioning": "ac",
    "prefarea": "prime_locality",
    "furnishingstatus": "furnishing"
})


# =========================================================
# 4. PRICE DISTRIBUTION GRAPH
# =========================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["price"],
    bins=30,
    alpha=0.75,
    edgecolor="black"
)

plt.axvline(
    df["price"].mean(),
    linestyle="--",
    linewidth=2,
    color="red",
    label=f"Average Price: {inr(df['price'].mean())}"
)

plt.gca().xaxis.set_major_formatter(
    FuncFormatter(lakh_axis)
)

plt.xlabel("House Price (₹ Lakh)")
plt.ylabel("Number of Houses")
plt.title("House Price Distribution (India)")
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()


# =========================================================
# 5. PRICE VS AREA GRAPH
# =========================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["area_sqft"],
    df["price"],
    alpha=0.6,
    edgecolors="black"
)

plt.gca().yaxis.set_major_formatter(
    FuncFormatter(lakh_axis)
)

plt.xlabel("Area (sq ft)")
plt.ylabel("Price (₹ Lakh)")
plt.title("Price vs Area")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# =========================================================
# 6. FEATURES AND TARGET
# =========================================================

X = df.drop("price", axis=1)
y = df["price"]

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


# =========================================================
# 7. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# =========================================================
# 8. CREATE LINEAR REGRESSION MODEL
# =========================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# =========================================================
# 9. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# 10. TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)

print("Model training completed!")


# =========================================================
# 11. MAKE PREDICTIONS
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# 12. MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n===================================")
print("        MODEL EVALUATION")
print("===================================")

print(f"MAE      : {inr(mae)}")
print(f"RMSE     : {inr(rmse)}")
print(f"R² Score : {r2:.4f}")


# =========================================================
# 13. ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.65,
    s=60,
    edgecolors="black"
)

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    linewidth=2,
    color="red",
    label="Perfect Prediction"
)

ax = plt.gca()

ax.xaxis.set_major_formatter(
    FuncFormatter(lakh_axis)
)

ax.yaxis.set_major_formatter(
    FuncFormatter(lakh_axis)
)

plt.xlabel("Actual Price (₹ Lakh)")
plt.ylabel("Predicted Price (₹ Lakh)")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# =========================================================
# 14. PREDICT A NEW HOUSE PRICE
# =========================================================

new_house = pd.DataFrame({
    "area_sqft": [1800],
    "bhk": [3],
    "bathrooms": [2],
    "floors": [2],
    "main_road_facing": ["yes"],
    "guest_room": ["no"],
    "basement": ["no"],
    "geyser": ["yes"],
    "ac": ["yes"],
    "car_parking": [1],
    "prime_locality": ["yes"],
    "furnishing": ["semi-furnished"]
})


# =========================================================
# 15. PREDICT PRICE
# =========================================================

predicted_price = model.predict(new_house)[0]


# =========================================================
# 16. DISPLAY NEW HOUSE DETAILS
# =========================================================

print("\n===================================")
print("       NEW HOUSE DETAILS")
print("===================================")

print(new_house.T.to_string(
    header=False,
    index=True
))


# =========================================================
# 17. FINAL RESULT
# =========================================================

print("\n===================================")
print("   HOUSE PRICE PREDICTION (INDIA)")
print("===================================")

print(f"MAE             : {inr(mae)}")
print(f"RMSE            : {inr(rmse)}")
print(f"R² Score        : {r2:.4f}")

print("-----------------------------------")

print(f"Predicted Price : {inr(predicted_price)}")
print(f"Exact Price     : ₹{predicted_price:,.0f}")

print("===================================")