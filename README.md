# Stock Price Prediction with Machine Learning

This project predicts the closing price of a stock using historical data from Yahoo Finance. It implements various machine learning models, such as Linear Regression, Random Forest, XGBoost, and LightGBM, to forecast the next day's closing price. The best-performing model is saved and used for real-time predictions through a Flask web application.

## Features

- Predict stock closing price for the next day using historical stock data.
- Select the stock ticker symbol and source (currently only Yahoo Finance is supported).
- View predicted price on a clean and interactive web interface.
- The model uses advanced machine learning algorithms such as:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor
  - LightGBM Regressor
  - Support Vector Regressor (SVR)

## Technologies Used

- **Flask**: Web framework for building the application.
- **YFinance**: Fetch stock data from Yahoo Finance.
- **Scikit-learn**: For machine learning algorithms and data preprocessing.
- **XGBoost**: High-performance gradient boosting algorithm.
- **LightGBM**: Another high-performance gradient boosting algorithm.
- **Joblib**: For saving and loading machine learning models.
- **Pandas**: For data manipulation and cleaning.
- **NumPy**: For numerical operations.

## Installation

Follow these steps to get your local development environment set up:

1. Clone the repository:
    ```bash
    git clone https://github.com/Nikhil-Emmanuel/stock.git
    cd stock-price-prediction
    ```

2. Create and activate a virtual environment:
    - For Windows:
      ```bash
      python -m venv venv
      venv\Scripts\activate
      ```
    - For macOS/Linux:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Make sure you have the trained model (`best_model.pkl`) and scaler (`scaler.pkl`) in the root directory, or train the model by running the `model.py` file.

## Running the Application

To run the Flask application, use the following command:

```bash
python app.py
```

## Open the Application
Once the command is executed , click on `127.0.0.1:5000` to open the application in your browser.
