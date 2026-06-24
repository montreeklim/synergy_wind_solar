import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree, DecisionTreeClassifier
import matplotlib.pyplot as plt

def analyze_synergy_with_hourly_correlation():
    """
    Builds and visualizes a Decision Tree using hourly correlation data
    to extract interpretable rules for high synergy ratios.
    """
    # =========================================================================
    # 1. CREATE HOURLY CORRELATION FEATURES (df_corrs)
    # This section is from the code you provided.
    # NOTE: Ensure the CSV files are in the correct path (e.g., a 'data' subfolder).
    # =========================================================================
    try:
        df_pv = pd.read_csv("../data/EMHIRES_PV_2015.csv")
        df_wind = pd.read_csv("../data/EMHIRES_wind_2015.csv")
    except FileNotFoundError:
        print("Error: Ensure 'EMHIRES_PV_2015.csv' and 'EMHIRES_wind_2015.csv' are in a subfolder named 'data'.")
        return

    caps = {
        "AT": {"wind":1981, "solar":404}, "BE": {"wind":2172, "solar":3068},
        "BG": {"wind":701, "solar":1041}, "CH": {"wind":60, "solar":756},
        "CZ": {"wind":277, "solar":2067}, "DE": {"wind":43429, "solar":38411},
        "DK": {"wind":5082, "solar":781}, "EE": {"wind":301, "solar":6},
        "ES": {"wind":23003, "solar":6967}, "FI": {"wind":1082, "solar":11},
        "FR": {"wind":10312, "solar":6192}, "EL": {"wind":1775, "solar":2444}, # Changed EL to GR for consistency
        "HR": {"wind":384, "solar":44}, "HU": {"wind":328, "solar":29},
        "IE": {"wind":2400, "solar":1}, "IT": {"wind":8750, "solar":19100},
        "LT": {"wind":290, "solar":69}, "LU": {"wind":60, "solar":116},
        "LV": {"wind":70, "solar":2}, "NL": {"wind":3641, "solar":1429},
        "NO": {"wind":860, "solar":14}, "PL": {"wind":5186, "solar":87},
        "PT": {"wind":4826, "solar":429}, "RO": {"wind":2923, "solar":1249},
        "SI": {"wind":3, "solar":263}, "SK": {"wind":3, "solar":532},
        "SE": {"wind":3029, "solar":104}, "UK": {"wind":13563, "solar":9000},
    }
    countries = list(caps.keys())
    pv_mult = pd.Series({c: caps[c]['solar'] for c in countries})
    wind_mult = pd.Series({c: caps[c]['wind'] for c in countries})
    df_pv[countries] = df_pv[countries].mul(pv_mult, axis=1)
    df_wind[countries] = df_wind[countries].mul(wind_mult, axis=1)
    valid = [c for c in countries if caps[c]['wind']>0 and caps[c]['solar']>0]
    corrs = []
    for h in range(5, 18):
        p = df_pv[df_pv['Hour']==h][valid]
        w = df_wind[df_wind['Hour']==h][valid]
        corrs.append(p.corrwith(w))
    df_corrs = pd.concat(corrs, axis=1)
    # Rename columns for clarity in the tree visualization
    df_corrs.columns = [f'corr_hour_{h}' for h in range(5, 18)]

    # =========================================================================
    # 2. LOAD OTHER FEATURES AND MERGE
    # =========================================================================
    data = {
        'country': ['AT', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI',
                    'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'NL', 'NO',
                    'PL', 'PT', 'RO', 'SI', 'SK', 'SE', 'UK', 'EL'],
        'drop': [0.020, 0.020, 0.012, 0.007, 0.003, 0.008, 0.002, 0.002, 0.045,
                 0.002, 0.030, 0.128, 0.051, 0.000, 0.008, 0.023, 0.009, 0.006,
                 0.038, 0.005, 0.001, 0.047, 0.011, 0.004, 0.000, 0.001, 0.008,
                 0.016],
        'reduce var': [0.146, 0.037, 0.016, 0.006, 0.003, 0.020, 0.020, 0.045,
                       0.164, 0.057, 0.084, 0.587, 0.405, 0.000, 0.009, 0.118,
                       0.012, 0.068, 0.176, 0.206, 0.020, 0.258, 0.036, 0.003,
                       0.000, 0.024, 0.031, 0.022],
        'pv_share': [0.088, 0.418, 0.439, 0.095, 0.211, 0.289, 0.077, 0.006,
                     0.170, 0.003, 0.251, 0.081, 0.069, 0.000, 0.311, 0.091,
                     0.374, 0.016, 0.144, 0.003, 0.007, 0.078, 0.235, 0.012,
                     0.007, 0.008, 0.173, 0.471],
        'average synergy 1': [1.56, 1.485, 1.495, 1.11, 1.24, 1.35, 1.44, 1.135,
                              1.22, 1.05, 1.325, 1.395, 1.38, 1.0, 1.18, 1.68,
                              1.4, 1.17, 1.685, 1.04, 1.115, 1.265, 1.57, 1.03,
                              1.015, 1.175, 1.415, 1.255]
    }
    df_main = pd.DataFrame(data).set_index('country')

    # Merge the main features with the hourly correlation features
    df_full = df_main.merge(df_corrs, left_index=True, right_index=True)
    
    # =========================================================================
    # 3. DEFINE NEW FEATURE SET AND RUN THE MODEL
    # =========================================================================
    
    # Define the new, expanded feature set
    base_features = ['drop', 'reduce var', 'pv_share']
    hourly_corr_features = list(df_corrs.columns)
    features = base_features + hourly_corr_features
    
    target = 'average synergy 1'
    X = df_full[features]
    y = df_full[target]

    # Create and train the PRUNED Decision Tree model
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X, y)

    # Visualize the Decision Tree
    print("🌳 Decision Tree with Hourly Correlation Features")
    plt.figure(figsize=(22, 14))
    plot_tree(
        model,
        feature_names=features,
        filled=True,
        rounded=True,
        precision=3,
        fontsize=9
    )
    plt.title("Decision Tree for Synergy Ratio (with Hourly Correlation Features)", fontsize=16)
    plt.show()
    return X

def analyze_synergy_groups_with_classifier():
    """
    Categorizes the synergy ratio into groups and builds a Decision Tree
    Classifier to find the rules that define each group.
    """
    # 1. Load the dataset (using the updated pv_share list)
    data = {
        'country': ['AT', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI',
                    'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'NL', 'NO',
                    'PL', 'PT', 'RO', 'SI', 'SK', 'SE', 'UK', 'EL'],
        'drop': [0.020, 0.020, 0.012, 0.007, 0.003, 0.008, 0.002, 0.002, 0.045,
                 0.002, 0.030, 0.128, 0.051, 0.000, 0.008, 0.023, 0.009, 0.006,
                 0.038, 0.005, 0.001, 0.047, 0.011, 0.004, 0.000, 0.001, 0.008,
                 0.016],
        'reduce var': [0.146, 0.037, 0.016, 0.006, 0.003, 0.020, 0.020, 0.045,
                       0.164, 0.057, 0.084, 0.587, 0.405, 0.000, 0.009, 0.118,
                       0.012, 0.068, 0.176, 0.206, 0.020, 0.258, 0.036, 0.003,
                       0.000, 0.024, 0.031, 0.022],
        'average corr': [-0.249, -0.358, -0.318, -0.347, -0.302, -0.358, -0.290,
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
    df = pd.DataFrame(data)

    # 2. --- NEW: Create Synergy Categories ---
    # Define the intervals (bins) and labels for our groups
    bins = [0, 1.2, 1.4, float('inf')]
    labels = ['Low Synergy', 'Medium Synergy', 'High Synergy']
    df['synergy_group'] = pd.cut(df['average synergy 1'], bins=bins, labels=labels, right=False)
    
    # 3. Define features (X) and the NEW categorical target (y)
    features = ['drop', 'reduce var', 'average corr', 'pv_share']
    target = 'synergy_group'
    X = df[features]
    y = df[target]

    # 4. Create and train a Decision Tree CLASSIFIER
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)

    # 5. Visualize the new classification tree
    print("🌳 Decision Tree Classifier for Synergy Groups")
    plt.figure(figsize=(18, 10))
    plot_tree(
        model,
        feature_names=features,
        class_names=labels, # Use the group labels in the nodes
        filled=True,
        rounded=True,
        precision=3,
        fontsize=9
    )
    plt.title("Decision Tree for Synergy Group Classification", fontsize=16)
    plt.show()
    return X

def analyze_synergy_groups_with_hourly_correlation():
    """
    Builds a Decision Tree Classifier using hourly correlation data to
    find the rules that define synergy groups (Low, Medium, High).
    """
    # =========================================================================
    # 1. CREATE HOURLY CORRELATION FEATURES (df_corrs)
    # NOTE: Ensure the CSV files are in the correct path (e.g., a 'data' subfolder).
    # =========================================================================
    try:
        df_pv = pd.read_csv("../data/EMHIRES_PV_2015.csv")
        df_wind = pd.read_csv("../data/EMHIRES_wind_2015.csv")
    except FileNotFoundError:
        print("Error: Ensure 'EMHIRES_PV_2015.csv' and 'EMHIRES_wind_2015.csv' are in a subfolder named 'data'.")
        return

    # Using 'GR' for Greece to ensure consistency for merging
    caps = {
        "AT": {"wind":1981, "solar":404}, "BE": {"wind":2172, "solar":3068},
        "BG": {"wind":701, "solar":1041}, "CH": {"wind":60, "solar":756},
        "CZ": {"wind":277, "solar":2067}, "DE": {"wind":43429, "solar":38411},
        "DK": {"wind":5082, "solar":781}, "EE": {"wind":301, "solar":6},
        "ES": {"wind":23003, "solar":6967}, "FI": {"wind":1082, "solar":11},
        "FR": {"wind":10312, "solar":6192}, "EL": {"wind":1775, "solar":2444},
        "HR": {"wind":384, "solar":44}, "HU": {"wind":328, "solar":29},
        "IE": {"wind":2400, "solar":1}, "IT": {"wind":8750, "solar":19100},
        "LT": {"wind":290, "solar":69}, "LU": {"wind":60, "solar":116},
        "LV": {"wind":70, "solar":2}, "NL": {"wind":3641, "solar":1429},
        "NO": {"wind":860, "solar":14}, "PL": {"wind":5186, "solar":87},
        "PT": {"wind":4826, "solar":429}, "RO": {"wind":2923, "solar":1249},
        "SI": {"wind":3, "solar":263}, "SK": {"wind":3, "solar":532},
        "SE": {"wind":3029, "solar":104}, "UK": {"wind":13563, "solar":9000},
    }
    countries = list(caps.keys())
    pv_mult = pd.Series({c: caps[c]['solar'] for c in countries})
    wind_mult = pd.Series({c: caps[c]['wind'] for c in countries})
    df_pv[countries] = df_pv[countries].mul(pv_mult, axis=1)
    df_wind[countries] = df_wind[countries].mul(wind_mult, axis=1)
    valid = [c for c in countries if caps[c]['wind']>0 and caps[c]['solar']>0]
    corrs = []
    for h in range(5, 18):
        p = df_pv[df_pv['Hour']==h][valid]
        w = df_wind[df_wind['Hour']==h][valid]
        corrs.append(p.corrwith(w))
    df_corrs = pd.concat(corrs, axis=1)
    df_corrs.columns = [f'corr_hour_{h}' for h in range(5, 18)]

    # =========================================================================
    # 2. LOAD OTHER FEATURES AND MERGE
    # =========================================================================
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
        'pv_share': [0.088, 0.418, 0.439, 0.095, 0.211, 0.289, 0.077, 0.006,
                     0.170, 0.003, 0.251, 0.081, 0.069, 0.000, 0.311, 0.091,
                     0.374, 0.016, 0.144, 0.003, 0.007, 0.078, 0.235, 0.012,
                     0.007, 0.008, 0.173, 0.471],
        'average synergy 1': [1.56, 1.485, 1.495, 1.11, 1.24, 1.35, 1.44, 1.135,
                              1.22, 1.05, 1.325, 1.395, 1.38, 1.0, 1.18, 1.68,
                              1.4, 1.17, 1.685, 1.04, 1.115, 1.265, 1.57, 1.03,
                              1.015, 1.175, 1.415, 1.255]
    }
    df_main = pd.DataFrame(data).set_index('country')
    df_full = df_main.merge(df_corrs, left_index=True, right_index=True)

    # =========================================================================
    # 3. CREATE CATEGORICAL TARGET AND RUN THE CLASSIFIER
    # =========================================================================
    bins = [0, 1.1, 1.4, float('inf')]
    labels = ['Low Synergy', 'Medium Synergy', 'High Synergy']
    df_full['synergy_group'] = pd.cut(df_full['average synergy 1'], bins=bins, labels=labels, right=False)
    
    base_features = ['drop', 'reduce var', 'pv_share']
    hourly_corr_features = list(df_corrs.columns)
    features = base_features + hourly_corr_features
    target = 'synergy_group'
    X = df_full[features]
    y = df_full[target]

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)

    # Visualize the new classification tree
    print("🌳 Decision Tree Classifier with Hourly Correlation Features")
    plt.figure(figsize=(20, 12))
    plot_tree(
        model,
        feature_names=features,
        class_names=labels,
        filled=True,
        rounded=True,
        precision=3,
        fontsize=9
    )
    plt.title("Decision Tree for Synergy Group Classification (with Hourly Features)", fontsize=16)
    plt.show()
    return X

# Run the function
X = analyze_synergy_groups_with_hourly_correlation()

# Run the function
# X = analyze_synergy_groups_with_classifier()

# Run the function
# X = analyze_synergy_with_hourly_correlation()