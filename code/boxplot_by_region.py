import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# --- 1. Common Data and Mappings ---

# Climatological groups (used by both plots)
climatological_groups = {
    "Atlantic Maritime": ['Ireland', 'United Kingdom', 'France', 'Belgium', 'Netherlands'],
    "Continental": ['Germany', 'Poland', 'Czech Republic', 'Austria', 'Hungary', 'Switzerland', 'Luxembourg', 'Slovakia'],
    "Mediterranean": ['Spain', 'Portugal', 'Italy', 'Greece', 'Croatia', 'Slovenia', 'Bulgaria', 'Romania'],
    "Nordic & Baltic": ['Norway', 'Sweden', 'Finland', 'Denmark', 'Estonia', 'Latvia', 'Lithuania']
}

# Create a reverse mapping from country name to region
country_to_region = {country: region for region, countries in climatological_groups.items() for country in countries}

# --- 2. Generate First Plot (Battery) ---

# Data for the first plot
country_data_string_1 = """country	epsilon_0.01	epsilon_0.05
AT	(1.30, 1.39)	(1.18, 1.29)
BE	(1.30, 1.38)	(1.30, 1.38)
BG	(1.25, 1.36)	(1.29, 1.43)
CH	(1.06, 1.09)	(1.08, 1.10)
CZ	(1.14, 1.21)	(1.15, 1.18)
DE	(1.19, 1.28)	(1.27, 1.36)
DK	(1.21, 1.28)	(1.24, 1.30)
EE	(1.05, 1.08)	(1.04, 1.07)
ES	(1.14, 1.19)	(1.14, 1.20)
FI	(1.02, 1.03)	(1.02, 1.03)
FR	(1.22, 1.27)	(1.20, 1.27)
GR	(1.18, 1.23)	(1.20, 1.24)
HR	(1.17, 1.21)	(1.23, 1.27)
HU	(1.16, 1.20)	(1.18, 1.26)
IE	(1.00, 1.00)	(1.00, 1.00)
IT	(1.12, 1.16)	(1.12, 1.16)
LT	(1.36, 1.46)	(1.29, 1.42)
LU	(1.23, 1.35)	(1.27, 1.35)
LV	(1.07, 1.10)	(1.07, 1.10)
NL	(1.39, 1.51)	(1.29, 1.39)
NO	(1.01, 1.04)	(1.01, 1.01)
PL	(1.05, 1.06)	(1.03, 1.05)
PT	(1.12, 1.20)	(1.05, 1.09)
RO	(1.31, 1.42)	(1.31, 1.44)
SI	(1.01, 1.02)	(1.01, 1.02)
SK	(1.00, 1.01)	(1.01, 1.01)
SE	(1.07, 1.10)	(1.06, 1.09)
GB	(1.25, 1.31)	(1.21, 1.28)
"""

# Mapping for the first plot's country codes
country_code_map_1 = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CH': 'Switzerland',
    'CZ': 'Czech Republic', 'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia',
    'ES': 'Spain', 'FI': 'Finland', 'FR': 'France', 'GB': 'United Kingdom',
    'GR': 'Greece', 'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland',
    'IT': 'Italy', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia',
    'NL': 'Netherlands', 'NO': 'Norway', 'PL': 'Poland', 'PT': 'Portugal',
    'RO': 'Romania', 'SE': 'Sweden', 'SI': 'Slovenia', 'SK': 'Slovakia'
}

# Process data for the first plot
df1 = pd.read_csv(io.StringIO(country_data_string_1), sep='\t')
def get_average_from_interval(interval_str):
    parts = interval_str.strip('()').split(',')
    return (float(parts[0]) + float(parts[1])) / 2
df1['avg_0.01'] = df1['epsilon_0.01'].apply(get_average_from_interval)
df1['avg_0.05'] = df1['epsilon_0.05'].apply(get_average_from_interval)
df1['country_name'] = df1['country'].map(country_code_map_1)
df1['Region'] = df1['country_name'].map(country_to_region)
df_long_1 = pd.melt(df1, id_vars=['Region'], value_vars=['avg_0.01', 'avg_0.05'],
                    var_name='Reliability threshold', value_name='Average Synergy Ratio')
df_long_1['Reliability threshold'] = df_long_1['Reliability threshold'].replace({'avg_0.01': '0.01', 'avg_0.05': '0.05'})

# Create the first plot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 8))
ax1 = sns.boxplot(data=df_long_1, x='Region', y='Average Synergy Ratio', hue='Reliability threshold',
                  palette=['#F5E8DD', '#1E1E1E'], order=["Continental", "Atlantic Maritime", "Mediterranean", "Nordic & Baltic"])
plt.setp(ax1.get_legend().get_title(), fontsize=16)
plt.setp(ax1.get_legend().get_texts(), fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Region', fontsize=14)
plt.ylabel('Average Synergy Ratio', fontsize=14)
plt.tight_layout()
plt.savefig("boxplot_battery_new.png", dpi=300)
plt.close()

# --- 3. Generate Second Plot (Naive) ---

# Data for the second plot
data_string_2 = """country	tol	ratio_mean
AT	0.01	5.34007
AT	0.05	3.71853
BE	0.01	5.54167
BE	0.05	3.53313
BG	0.01	2.62056
BG	0.05	1.77724
CH	0.01	2.40881
CH	0.05	1.33104
CZ	0.01	0.23154
CZ	0.05	1.59162
DE	0.01	1.74737
DE	0.05	1.78051
DK	0.01	3.38077
DK	0.05	3.13566
EE	0.05	3.85596
EL	0.01	1.38922
EL	0.05	1.35861
ES	0.01	1.31874
ES	0.05	1.36114
FI	0.05	12.06112
FR	0.01	1.8898
FR	0.05	1.85034
HR	0.01	11.69279
HR	0.05	5.21189
HU	0.01	5.7113
HU	0.05	3.04718
IE	0.01	52.43378
IE	0.05	2.99957
IT	0.01	1.32299
IT	0.05	1.27998
LT	0.05	19.69641
LU	0.01	4.10997
LU	0.05	2.76452
LV	0.01	0.25366
LV	0.05	1.25073
NL	0.01	3.71242
NL	0.05	3.49606
NO	0.01	5.30685
NO	0.05	1.10654
PL	0.01	22.63879
PL	0.05	9.12974
PT	0.01	1.49482
PT	0.05	1.47107
RO	0.01	3.81406
RO	0.05	3.02638
SE	0.01	3.64687
SE	0.05	3.18852
SI	0.01	0.23206
SI	0.05	1.03668
SK	0.01	1.04623
SK	0.05	1.01556
UK	0.01	2.60203
UK	0.05	2.20583
"""

# Mapping for the second plot's country codes
country_code_map_2 = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CH': 'Switzerland',
    'CZ': 'Czech Republic', 'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia',
    'ES': 'Spain', 'FI': 'Finland', 'FR': 'France', 'UK': 'United Kingdom',
    'EL': 'Greece', 'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland',
    'IT': 'Italy', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia',
    'NL': 'Netherlands', 'NO': 'Norway', 'PL': 'Poland', 'PT': 'Portugal',
    'RO': 'Romania', 'SE': 'Sweden', 'SI': 'Slovenia', 'SK': 'Slovakia'
}

# Process data for the second plot
df2 = pd.read_csv(io.StringIO(data_string_2), sep='\t')
df2['country_name'] = df2['country'].map(country_code_map_2)
df2['Region'] = df2['country_name'].map(country_to_region)
df_plot_2 = df2.rename(columns={'tol': 'Reliability threshold', 'ratio_mean': 'Average Synergy Ratio'})
df_plot_2['Reliability threshold'] = df_plot_2['Reliability threshold'].astype(str)

# Create the second plot
plt.figure(figsize=(8, 8))
ax2 = sns.boxplot(data=df_plot_2, x='Region', y='Average Synergy Ratio', hue='Reliability threshold',
                  palette=['#F5E8DD', '#1E1E1E'], order=["Continental", "Atlantic Maritime", "Mediterranean", "Nordic & Baltic"])
ax2.set_ylim(0, 15)
plt.setp(ax2.get_legend().get_title(), fontsize=16)
plt.setp(ax2.get_legend().get_texts(), fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=16)
plt.xlabel('Region', fontsize=14)
plt.ylabel('Average Synergy Ratio', fontsize=14)
plt.tight_layout()
plt.savefig("boxplot_naive_new.png", dpi=300)
plt.close()