import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def build_enhanced_synergy_model():
    """
    Builds an enhanced linear regression model with four features
    (including PV share) to predict average synergy and prints the
    R-squared value.
    """
    # 1. Load the dataset with the new 'pv_share' feature included
    data = {
        'country': ['AT', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI',
                    'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'NL', 'NO',
                    'PL', 'PT', 'RO', 'SI', 'SK', 'SE', 'UK', 'GR'],
        'drop': [0.020, 0.020, 0.012, 0.007, 0.003, 0.008, 0.002, 0.002, 0.045,
                 0.002, 0.030, 0.128, 0.051, 0.000, 0.008, 0.023, 0.009, 0.006,
                 0.038, 0.005, 0.001, 0.047, 0.011, 0.004, 0.000, 0.001, 0.008,
                 0.016],
        'reduce var': [0.146, 0.037, 0.016, 0.006, 0.003, 0.020, 0.020, 0.045,
                       0.164, 0.057, 0.084, 0.587, 0.405, 0.000, 0.009, 0.118,
                       0.012, 0.068, 0.176, 0.206, 0.020, 0.258, 0.036, 0.003,
                       0.000, 0.024, 0.031, 0.022],
        'average corr': [-0.249, -0.358, -0.318, -0.347, -0.302, --0.358, -0.290,
                         -0.320, -0.343, -0.345, -0.369, -0.307, -0.175, -0.325,
                         -0.306, -0.283, -0.263, -0.300, -0.367, -0.475, -0.302,
                         -0.240, -0.308, -0.260, -0.233, -0.328, -0.363, -0.225],
        'pv_share': [0.088, 0.418, 0.439, 0.095, 0.211, 0.289, 0.077, 0.006,
                     0.170, 0.003, 0.251, 0.081, 0.069, 0.000, 0.311, 0.091,
                     0.374, 0.016, 0.144, 0.003, 0.007, 0.078, 0.235, 0.012,
                     0.007, 0.008, 0.173, 0.471],
        'average synergy 1': [1.56, 1.485, 1.495, 1.11, 1.24, 1.35, 1.44, 1.135,
                              1.22, 1.05, 1.325, 1.395, 1.38, 1.0, 1.18, 1.68,
                              1.4, 1.17, 1.685, 1.04, 1.115, 1.265, 1.57, 1.03,
                              1.015, 1.175, 1.415, 1.255]
    }
    df = pd.DataFrame(data).set_index('country')

    # 2. Define features (X) and target (y), now including 'pv_share'
    features = ['drop', 'reduce var', 'average corr', 'pv_share']
    target = 'average synergy 1'
    X = df[features]
    y = df[target]

    # 3. Create and train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # 4. Make predictions to evaluate the model
    y_pred = model.predict(X)

    # 5. Calculate and print the R-squared value
    r2 = r2_score(y, y_pred)
    
    print("📈 Enhanced Regression Model Results")
    print("--------------------------------------")
    print(f"R-squared (R^2) value: {r2:.4f}")
    
    # Optional: Display the model's learned coefficients
    print("\nModel Coefficients:")
    print(f"Intercept: {model.intercept_:.4f}")
    for feature, coef in zip(features, model.coef_):
        print(f"- {feature}: {coef:.4f}")

# Run the function
build_enhanced_synergy_model()