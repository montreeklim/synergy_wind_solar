import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

def create_synergy_classification_tree(file_path):
    """
    Loads synergy data, classifies the average_ratio into custom bins,
    and builds a decision tree to visualize the classification rules.

    Args:
        file_path (str): The path to the input CSV file.
    """
    try:
        # --- THIS IS THE CORRECTED LINE ---
        df = pd.read_excel(file_path, index_col=0)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it is in the correct directory.")
        return

    # 1. Create the categorical target variable by binning 'average_ratio'
    bins = [1, 1.1, 1.2, float('inf')]
    labels = ['1.0-1.1', '1.1-1.2', '>1.2']
    df['synergy_group'] = pd.cut(df['average_synergy_5'], bins=bins, labels=labels, right=False)

    # 2. Define features (X) and the new categorical target (y)
    # Exclude the original continuous ratio and the country column
    # features = df.columns.drop(['country', 'average_synergy_1', 'average_synergy_5', 'synergy_group'])
    features = df.columns.drop(['average_synergy_1', 'average_synergy_5', 'synergy_group', 'drop', 'var_reduction', 
                                'corr_hour_5', 'corr_hour_6', 'corr_hour_7', 'corr_hour_8', 'corr_hour_9', 'corr_hour_10', 'corr_hour_11',
                                'corr_hour_12', 'corr_hour_13', 'corr_hour_14', 'corr_hour_15', 'corr_hour_16', 'corr_hour_17',])
    target = 'synergy_group'

    X = df[features]
    y = df[target]

    # Handle cases where some bins might be empty
    if y.isnull().any():
        print("Warning: Some rows could not be assigned to a bin and will be dropped.")
        not_null_indices = y.notna()
        X = X[not_null_indices]
        y = y[not_null_indices]


    # 3. Create and train a pruned Decision Tree Classifier
    # We limit max_depth to make the tree readable and prevent overfitting.
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X, y)

    # 4. Visualize the Decision Tree
    print("🌳 Decision Tree for Synergy Group Classification")
    plt.figure(figsize=(25, 15))
    plot_tree(
        model,
        feature_names=list(features),
        class_names=labels,
        filled=True,
        rounded=True,
        precision=3,
        fontsize=10
    )
    plt.title("Decision Tree for Synergy Group Classification", fontsize=18)
    plt.show()
    return model, df[df.index.isin(X.index)], X, y


def print_country_splits(model, df, X):
    """
    Analyzes a trained decision tree and prints the countries belonging to each split.
    """
    print("\n" + "="*60)
    print("🌳 Country Breakdown for Each Decision in the Tree")
    print("="*60)

    tree = model.tree_
    decision_path = model.decision_path(X)

    # Iterate through each node of the tree
    for node_id in range(tree.node_count):
        # Check if it's a split node (not a leaf)
        if tree.children_left[node_id] != tree.children_right[node_id]:
            # Get the feature and threshold for the split
            feature_idx = tree.feature[node_id]
            feature_name = X.columns[feature_idx]
            threshold = tree.threshold[node_id]

            # Find countries that go LEFT (condition is TRUE)
            left_child_node_id = tree.children_left[node_id]
            left_indices = decision_path[:, left_child_node_id].nonzero()[0]
            left_countries = df.index[left_indices].tolist()

            # Find countries that go RIGHT (condition is FALSE)
            right_child_node_id = tree.children_right[node_id]
            right_indices = decision_path[:, right_child_node_id].nonzero()[0]
            right_countries = df.index[right_indices].tolist()

            print(f"\n--- Node {node_id} ---")
            print(f"Decision Rule: {feature_name} <= {threshold:.3f}")
            print(f"  ✅ TRUE (Go Left): {left_countries}")
            print(f"  ❌ FALSE (Go Right): {right_countries}")

# --- Run the analysis ---
# Use the name of the file you uploaded
file_name = 'data_decision_tree.xlsx'
model, df_for_analysis, X, y = create_synergy_classification_tree(file_name)

if model:
    print_country_splits(model, df_for_analysis, X)

df_selected = df_for_analysis[(df_for_analysis['PV share'] > 0.05) & (df_for_analysis['PV share'] <= 0.95)]

# --- 2. Generate and Save Each Plot Separately ---

# --- Plot 1: Morning Correlation ---
plt.figure(figsize=(6, 5)) # Create a new figure for the first plot
ax1 = plt.gca() # Get current axes

# Calculate regression
slope, intercept, r_value, p_value, std_err = stats.linregress(df_selected['morning_corr'], df_selected['average_synergy_1'])
sns.regplot(ax=ax1, x='morning_corr', y='average_synergy_1', data=df_selected)

# Add statistics text (R^2 only, black color)
stats_text = f'$R^2 = {r_value**2:.2f}$'
ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='left', color='black') # Changed color to 'black' and adjusted position

# Set labels and title
ax1.set_xlabel('Early Morning Correlation')
ax1.set_ylabel('Average Synergy')

# Save and close the plot
plt.tight_layout()
plt.savefig('morning_corr_1.png', dpi=300)
plt.close()

# --- Plot 2: Day Correlation ---
plt.figure(figsize=(6, 5)) # Create a new figure for the second plot
ax2 = plt.gca()

# Calculate regression
slope, intercept, r_value, p_value, std_err = stats.linregress(df_selected['day_corr'], df_selected['average_synergy_1'])
sns.regplot(ax=ax2, x='day_corr', y='average_synergy_1', data=df_selected)

# Add statistics text (R^2 only, black color)
stats_text = f'$R^2 = {r_value**2:.2f}$'
ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='left', color='black') # Changed color to 'black'

# Set labels and title
ax2.set_xlabel('Daytime Correlation')
ax2.set_ylabel('Average Synergy') # Added y-label as it's a separate plot

# Save and close the plot
plt.tight_layout()
plt.savefig('day_corr_1.png', dpi=300)
plt.close()

# --- Plot 3: Evening Correlation ---
plt.figure(figsize=(6, 5)) # Create a new figure for the third plot
ax3 = plt.gca()

# Calculate regression
slope, intercept, r_value, p_value, std_err = stats.linregress(df_selected['evening_corr'], df_selected['average_synergy_1'])
sns.regplot(ax=ax3, x='evening_corr', y='average_synergy_1', data=df_selected)

# Add statistics text (R^2 only, black color)
stats_text = f'$R^2 = {r_value**2:.2f}$'
ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='left', color='black') # Changed color to 'black'

# Set labels and title
ax3.set_xlabel('Late Afternoon Correlation')
ax3.set_ylabel('Average Synergy') # Added y-label as it's a separate plot

# Save and close the plot
plt.tight_layout()
plt.savefig('evening_corr_1.png', dpi=300)
plt.close()
