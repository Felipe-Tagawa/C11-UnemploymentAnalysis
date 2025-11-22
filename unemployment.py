import os
import kagglehub as kh
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

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

#print(df_general[infl_cols])
#print(df_general[unemp_cols])

for country_infl, country_unemp in zip(infl_cols, unemp_cols):
    serie1 = df_general[country_infl].copy()
    serie2 = df_general[country_unemp].copy()

    # Muitos NaNs, por isso é necessario preencher os NaNs com interpolação temporal
    serie1 = serie1.interpolate(method='time').ffill().bfill()
    serie2 = serie2.interpolate(method='time').ffill().bfill()

    # Transformação log para inflação alta (Brasil principal caso)
    log_applied_1 = False
    if serie1.max() > 20:
        serie1 = np.log1p(serie1)
        log_applied_1 = True

    # Definindo o período baseado na frequência dos dados
    freq = pd.infer_freq(df_general.index)

    if freq == 'A' or freq == 'Y':
        period = 2
    elif freq == 'M':
        period = 12
    else:
        period = 2

    #  Decomposição com extrapolate_trend pra evitar NaNs
    decomp_infl = seasonal_decompose(
        serie1,
        model='additive',
        period=period,
        extrapolate_trend='freq'  # Vai evitar o Nan
    )

    fig1 = decomp_infl.plot()
    if log_applied_1: log_note = "(escala log aplicada)"
    else: log_note = ""

    fig1.suptitle(f"Decomposição Inflação — {country_infl}{log_note}")
    plt.tight_layout()

    decomp_unemp = seasonal_decompose(
        serie2,
        model='additive',
        period=period,
        extrapolate_trend='freq'  # Vai evitar o Nan
    )

    fig2 = decomp_unemp.plot()
    fig2.suptitle(f"Decomposição Desemprego — {country_unemp}")
    plt.tight_layout()
    plt.show()

    # Print de infos de geral
    print(f"\nPaís: {country_infl.replace('_infl', '')}")
    print(f"Período usado: {period}")

'''
Perguntas:
a. A série possui Tendência? Se sim, que tipo?
b. A série possui Sazonalidade? Se sim, qual o período que ela acontece?
c. A série apresenta um Ciclo? Se sim, por qual razão?

Análise série 1:
a. 
b. 
c. 

Análise série 2:
a. 
b. 
c. 
'''