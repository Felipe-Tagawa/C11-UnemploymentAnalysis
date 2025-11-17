import os
import kagglehub as kh
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Baixar datasets
path = kh.dataset_download("marcelomizuno/unemployment-rates")
path2 = kh.dataset_download("sazidthe1/global-inflation-data")
csv_path = os.path.join(path, 'unemployment_rates.csv')
csv_path2 = os.path.join(path2, 'global_inflation_data.csv')

# Ler CSVs
df_unemp = pd.read_csv(csv_path)
df_inflation = pd.read_csv(csv_path2)

print(df_unemp.head())
print()
print(df_inflation.head())

output_dir = 'data'
os.makedirs(output_dir, exist_ok=True) # Criar diretório

df_unemp.to_csv(os.path.join(output_dir, 'unemployment_rates.csv'), index=False)
df_inflation.to_csv(os.path.join(output_dir, 'global_inflation_data.csv'), index=False)

