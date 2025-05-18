import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from google.colab import drive

# Mount Google Drive to use as storage
drive.mount('/content/drive')

# Create project directory
project_path = '/content/drive/MyDrive/TeslaStockPrediction'
!mkdir -p {project_path}

# Download TSLA stock data
ticker = "TSLA"
end_date = datetime.now()
start_date = end_date - timedelta(days=365*2)  # 2 years of data
print(f"Downloading {ticker} stock data...")
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Basic exploration
print(f"\nData shape: {stock_data.shape}")
print("\nFirst 5 rows:")
print(stock_data.head())
print("\nMissing values:")
print(stock_data.isnull().sum())
print("\nStatistics:")
print(stock_data.describe())

# Visualize closing price
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'])
plt.title(f'{ticker} Stock Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.savefig(f"{project_path}/{ticker}_price.png")
plt.show()

# Save data
stock_data.to_csv(f"{project_path}/{ticker}_stock_data.csv")
print(f"\nData saved to {project_path}/{ticker}_stock_data.csv")

# Data Preparation and featuring
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
from sklearn.preprocessing import StandardScaler
import joblib
drive.mount('/content/drive')
project_path = '/content/drive/MyDrive/TeslaStockPrediction'
ticker = "TSLA"

# Load
stock_data = pd.read_csv(f"{project_path}/{ticker}_stock_data.csv", index_col=0, parse_dates=True)
print(f"Data loaded: {stock_data.shape[0]} rows")

# Ensure 'Close' and 'Volume' are numeric
stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
stock_data['Volume'] = pd.to_numeric(stock_data['Volume'], errors='coerce')

# Create lag features
def create_lag_features(df, n_lags=5):
    df_copy = df.copy()

    # Lag closing prices
    for i in range(1, n_lags + 1):
        df_copy[f'Close_Lag_{i}'] = df_copy['Close'].shift(i)

    # Create features
    # 1. Volatility (5-day standard deviation)
    df_copy['Volatility'] = df_copy['Close'].rolling(window=5).std()

    # 2. 5-day price momentum
    df_copy['Momentum'] = df_copy['Close'].pct_change(periods=5)

    # 3. 10-day moving average ratio
    df_copy['MA10_Ratio'] = df_copy['Close'] / df_copy['Close'].rolling(window=10).mean()

    # 4. Volume change
    df_copy['Volume_Change'] = df_copy['Volume'].pct_change()

    # Drop rows with NaN values
    df_copy = df_copy.dropna()
    return df_copy

processed_data = create_lag_features(stock_data, n_lags=5)
print(f"Processed data shape: {processed_data.shape}")
print("Features created:")
print(processed_data.columns.tolist())

# Prepare features and target
features = ['Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5','Volatility', 'Momentum', 'MA10_Ratio', 'Volume_Change']
X = processed_data[features]
y = processed_data['Close']

# Train-test split (80/20)
train_size = int(len(processed_data) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data
processed_data.to_csv(f"{project_path}/{ticker}_processed_data.csv")
np.save(f"{project_path}/X_train.npy", X_train_scaled)
np.save(f"{project_path}/X_test.npy", X_test_scaled)
np.save(f"{project_path}/y_train.npy", y_train.values)
np.save(f"{project_path}/y_test.npy", y_test.values)
joblib.dump(scaler, f"{project_path}/scaler.pkl")

print("\nData preparation complete. Files saved to Google Drive.")

# Training Code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from google.colab import drive

drive.mount("/content/drive", force_remount=True)
project_path = '/content/drive/MyDrive/TeslaStockPrediction'
ticker = "TSLA"

# Load
X_train = np.load(f"{project_path}/X_train.npy")
X_test = np.load(f"{project_path}/X_test.npy")
y_train = np.load(f"{project_path}/y_train.npy")
y_test = np.load(f"{project_path}/y_test.npy")

print(f"Loaded training data: {X_train.shape[0]} samples")
print(f"Loaded testing data: {X_test.shape[0]} samples")

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calc metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{model_name} Evaluation:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    return model, y_pred, mse

# Model 1: Linear Regression
print("\nTraining Linear Regression...")
lr_model, lr_pred, lr_mse = evaluate_model(LinearRegression(), X_train, X_test, y_train, y_test, "Linear Regression")

# Model 2: Random Forest
print("\nTraining Random Forest...")
rf_model, rf_pred, rf_mse = evaluate_model(RandomForestRegressor(n_estimators=100, random_state=42),X_train, X_test, y_train, y_test, "Random Forest")

# Model 3: Gradient Boosting
print("\nTraining Gradient Boosting...")
gbr_model, gbr_pred, gbr_mse = evaluate_model(GradientBoostingRegressor(n_estimators=100, random_state=42),X_train, X_test, y_train, y_test, "Gradient Boosting")

# Model 4: XGBoost
print("\nTraining XGBoost...")
xgb_model, xgb_pred, xgb_mse = evaluate_model(XGBRegressor(n_estimators=100, random_state=42),X_train, X_test, y_train, y_test, "XGBoost")

# Model 5: LightGBM
print("\nTraining LightGBM...")
lgbm_model, lgbm_pred, lgbm_mse = evaluate_model(LGBMRegressor(n_estimators=100, random_state=42),X_train, X_test, y_train, y_test, "LightGBM")

# Model 6: SVR
print("\nTraining SVR...")
svr_model, svr_pred, svr_mse = evaluate_model(SVR(),X_train, X_test, y_train, y_test, "SVR")

# Model list with scores
models = {
    "Linear Regression": (lr_model, lr_pred, lr_mse),
    "Random Forest": (rf_model, rf_pred, rf_mse),
    "Gradient Boosting": (gbr_model, gbr_pred, gbr_mse),
    "XGBoost": (xgb_model, xgb_pred, xgb_mse),
    "LightGBM": (lgbm_model, lgbm_pred, lgbm_mse),
    "SVR": (svr_model, svr_pred, svr_mse)
}
# Select best model with lowest MSE
best_model_name = min(models, key=lambda name: models[name][2])
best_model, best_pred, best_mse = models[best_model_name]
print(f"\nBest model: {best_model_name} with Mean Squared Error: {best_mse:.2f}")

# Visualize with plt
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(best_pred, label='Predicted', alpha=0.7)
plt.title(f'Tesla Stock Price: Actual vs Predicted ({best_model_name})')
plt.xlabel('Sample Index')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid(True)
plt.savefig(f"{project_path}/prediction_results.png")
plt.show()

# Save
joblib.dump(best_model, f"{project_path}/tesla_stock_model.pkl")

"""week4: deployyment"""

# Prediction Code
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from google.colab import drive

drive.mount('/content/drive')
project_path = '/content/drive/MyDrive/TSLA-Model'
ticker = "TSLA"

model = joblib.load(f"{project_path}/tesla_stock_model.pkl")
scaler = joblib.load(f"{project_path}/scaler.pkl")

# Get latest 30 day data from yahoo
end_date = datetime.now()
start_date = end_date - timedelta(days=30)  # fetch enough data for lag/rolling features
latest_data = yf.download(ticker, start=start_date, end=end_date)

def create_lag_features(df, n_lags=5):
    df_copy = df.copy()
    for i in range(1, n_lags + 1):
        df_copy[f'Close_Lag_{i}'] = df_copy['Close'].shift(i)
    df_copy['Volatility'] = df_copy['Close'].rolling(window=5).std()
    df_copy['Momentum'] = df_copy['Close'].pct_change(periods=5)
    df_copy['MA10_Ratio'] = df_copy['Close'] / df_copy['Close'].rolling(window=10).mean()
    df_copy['Volume_Change'] = df_copy['Volume'].pct_change()
    df_copy = df_copy.dropna()
    return df_copy

latest_features = create_lag_features(latest_data, n_lags=5)
feature_cols = ['Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5','Volatility', 'Momentum', 'MA10_Ratio', 'Volume_Change']
X_latest = latest_features[feature_cols]
X_latest_scaled = pd.DataFrame(scaler.transform(X_latest),columns=X_latest.columns,index=X_latest.index)
# Predict with current data
latest_prediction = model.predict(X_latest_scaled.iloc[[-1]])[0]
latest_actual = latest_features['Close'].iloc[-1].item()

# Calc Accuracy
abs_error = abs(latest_actual - latest_prediction)
percentage_error = (abs_error / latest_actual) * 100

print(f"\n Actual closing price:  ${latest_actual:.2f}")
print(f" Predicted price:       ${latest_prediction:.2f}")
print(f" Absolute Error :       ${abs_error:.2f}")
print(f" Accuracy of Model:     {100 - percentage_error:.2f}%")