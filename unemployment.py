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
df_unemp = pd.read_csv(csv_path, delimiter=',', index_col='TIME_PERIOD', parse_dates=True)
df_unemp.index.name = 'DATE'
df_inflation = pd.read_csv(csv_path2)

output_dir = 'data'
os.makedirs(output_dir, exist_ok=True) # Criar diretório

df_unemp.to_csv(os.path.join(output_dir, 'unemployment_rates.csv'), index=False)
df_inflation.to_csv(os.path.join(output_dir, 'global_inflation_data.csv'), index=False)

# Ajeitar o Dataset de Inflação para Time Series:

# Filtrar o indicador para remover
indicator = df_inflation['indicator_name'].unique()[0]
df_inflation = df_inflation[df_inflation['indicator_name'] == indicator]

# Remover o indicador
df_inflation = df_inflation.drop(columns=['indicator_name'])

# Conversão - Todos as colunas vinham em anos

df_inflation_long = df_inflation.melt(
    id_vars=['country_name'],
    var_name='year',
    value_name='inflation'
)

# Converter year para datetime:

df_inflation_long['year'] = pd.to_datetime(df_inflation_long['year'], format='%Y')

# Conversão 2 - cada país virar coluna
df_inflation_util = df_inflation_long.pivot(
    index='year',
    columns='country_name',
    values='inflation'
)

df_inflation_util.index.name = 'DATE' # Nome da coluna de anos

# Fazer o MERGE dos dois datasets:

df_general = df_unemp.merge(
    df_inflation_util,
    left_index=True,
    right_index=True,
    how='inner', # Funciona exatamente como um Inner Join
    suffixes=('_unemp', '_infl') # diferenciar colunas dos dois datasets
)

df_general.to_csv(os.path.join(output_dir, 'general_util.csv'), index=False)

# Plot Time Series

# Unemployment Rate
df_unemp['Brazil'].plot(figsize=(8,6), title='Índice de Desemprego no Brasil', xlabel = 'Data', ylabel = 'Unemployment Rate', x_compat = True)

# Selecionar apenas colunas de desemprego
unemp_cols = [col for col in df_general.columns if col.endswith('_unemp')]

# Plotar todas juntas
df_general[unemp_cols].plot(
    figsize=(12, 6),
    title="Taxa de Desemprego — 5 Países",
    xlabel="Ano",
    ylabel="Taxa de Desemprego"
)

# Selecionar apenas colunas de inflação
infl_cols = [col for col in df_general.columns if col.endswith('_infl')]

# Plotar todas juntas
df_general[infl_cols].plot(
    figsize=(12,6),
    title="Inflação — 5 Países",
    xlabel="Ano",
    ylabel="Inflação (%)",
    logy=True # Necessária pois no Brasil foi muito alta nos anos 80 e 90
)
plt.show()

# Decompose Time Series



