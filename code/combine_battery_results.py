import glob
import re
import pandas as pd

def load_folder(folder_tag):
    pat = re.compile(r'battery_([A-Z]+)_day_(\d+)_tol_([\d.]+)_' + re.escape(folder_tag))
    dfs = []
    for f in glob.glob(f'battery_results/{folder_tag}/*.csv'):
        m = pat.search(f)
        if not m:
            continue
        tmp = pd.read_csv(f)
        tmp['country']      = m.group(1)
        tmp['forecast_day'] = int(m.group(2))
        tmp['tol']          = float(m.group(3))
        tmp['battery_size'] = folder_tag
        dfs.append(tmp)
    return pd.concat(dfs, ignore_index=True)

df = load_folder('100_installed_capacity')

df = df[['forecast_day', 'tol', 'country', 'set_number',
         'wind objective', 'solar objective', 'combine objective', 'ratio']]
df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')
df = df.sort_values(
    ['tol', 'forecast_day', 'country', 'set_number']
).reset_index(drop=True)

out = 'results/all_countries_battery_results.xlsx'
df.to_excel(out, index=False)
print(f'Written {out}  —  {df.shape[0]} rows x {df.shape[1]} cols')
print(f'  tols={sorted(df.tol.unique())}  days={sorted(df.forecast_day.unique())}  countries={len(df.country.unique())}')
