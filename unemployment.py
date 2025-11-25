import os
import kagglehub as kh
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

# ==========================================
# 1. DOWNLOAD E LEITURA DOS DADOS
# ==========================================

# Dataset de Desemprego (Anual, Global)
path_unemp_global = kh.dataset_download("pantanjali/unemployment-dataset")
csv_unemp_global_path = os.path.join(path_unemp_global, 'unemployment analysis.csv')

# Dataset de Inflação (Anual, Global)
path_infl_global = kh.dataset_download("sazidthe1/global-inflation-data")
csv_infl_global_path = os.path.join(path_infl_global, 'global_inflation_data.csv')

# Ler CSV de Desemprego (Dados Anuais por País)
df_unemp_global = pd.read_csv(csv_unemp_global_path)

# Ler CSV de Inflação (Dados Anuais por País)
df_inflation_global = pd.read_csv(csv_infl_global_path)

# Criar diretório de saída
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 2. TRATAMENTO E MERGE DOS DADOS (Anual)
# ==========================================

# --- Tratamento do Desemprego ---

# Melt: Transformar anos em linhas para análise de séries temporais
df_unemp_long = df_unemp_global.melt(
    id_vars=['Country Name', 'Country Code'],
    var_name='year',
    value_name='unemployment'
)
df_unemp_long['year'] = pd.to_datetime(df_unemp_long['year'], format='%Y')
df_unemp_long = df_unemp_long.set_index('year').sort_index()

# Filtrar Desemprego do Brasil
df_unemp_br = df_unemp_long[df_unemp_long['Country Name'] == 'Brazil']['unemployment']
df_unemp_br.index.name = 'DATE'

# --- Tratamento da Inflação ---

# Filtrar indicador e fazer o Melt como no código anterior
indicator = df_inflation_global['indicator_name'].unique()[0]
df_inflation_global = df_inflation_global[df_inflation_global['indicator_name'] == indicator].drop(columns=['indicator_name'])

df_inflation_long = df_inflation_global.melt(
    id_vars=['country_name'],
    var_name='year',
    value_name='inflation'
)
df_inflation_long['year'] = pd.to_datetime(df_inflation_long['year'], format='%Y')
df_inflation_long = df_inflation_long.set_index('year').sort_index()

# Filtrar Inflação do Brasil
df_infl_br = df_inflation_long[df_inflation_long['country_name'] == 'Brazil']['inflation']
df_infl_br.index.name = 'DATE'

# --- Merge dos Datasets (Anual) ---

# Juntar as duas séries anuais do Brasil
df_br_general = pd.merge(
    df_unemp_br,
    df_infl_br,
    left_index=True,
    right_index=True,
    how='inner' # Como 'Inner Join'
)
df_br_general.columns = ['Brazil_unemp', 'Brazil_infl']

df_br_general.to_csv(os.path.join(output_dir, 'brazil_annual_series.csv'))

# ==========================================
# 3. VISUALIZAÇÃO DAS TIME SERIES (Brasil)
# ==========================================

country_name = 'Brazil'
infl_col = 'Brazil_infl'
unemp_col = 'Brazil_unemp'

# Desemprego Anual
plt.figure(figsize=(12, 6))
df_br_general[unemp_col].plot(
    title=f"Taxa de Desemprego Anual no {country_name}",
    xlabel="Ano",
    ylabel="Taxa de Desemprego (%)",
    marker='o'
)
plt.tight_layout()
plt.show()

# Inflação Anual no país
brazil_infl = df_br_general[infl_col].copy()

# Hipervolatilidade (Anos iniciais de 1970 - 1999)
infl_hyper_range = brazil_infl.loc[:'1999-12-31']
plt.figure(figsize=(12, 6))
infl_hyper_range.plot(
    title=f"Inflação Anual no {country_name} (Hipervolatilidade)",
    xlabel="Ano",
    ylabel="Inflação (%)",
    marker='o'
)
plt.tight_layout()
plt.show()

# Pós-Estabilização (2000 em diante)
infl_stable_range = brazil_infl.loc['2000-01-01':]
plt.figure(figsize=(12, 6))
infl_stable_range.plot(
    title=f"Inflação Anual no {country_name} (Pós-Estabilização)",
    xlabel="Ano",
    ylabel="Inflação (%)",
    marker='o'
)
plt.tight_layout()
plt.show()

# ==========================================
# 4. DECOMPOSIÇÃO DAS TIME SERIES
# ==========================================

# A decomposição sazonal em séries anuais (período=1)
PERIOD = 1 # Período de 1 ano

# Preencher gaps - valores incorretos
serie_infl = df_br_general[infl_col].copy().ffill().bfill()
serie_unemp = df_br_general[unemp_col].copy().ffill().bfill()

# --- Decomposição da INFLAÇÃO ---

log_note = ""
# Se a série tiver valores muito altos, como é o caso do Brasil para Inflação
if serie_infl.max() > 50:
    serie_infl = np.log1p(serie_infl)
    log_note = "(Escala Log)"

decomp_infl = seasonal_decompose(
    serie_infl,
    model='additive',
    period=PERIOD,
    extrapolate_trend='freq'
)

fig1 = decomp_infl.plot()
fig1.set_size_inches(10, 8)
fig1.suptitle(f"Decomposição Inflação {country_name} {log_note} (Anual)", fontsize=14)
plt.tight_layout()
plt.show()

# --- Decomposição do DESEMPREGO ---
decomp_unemp = seasonal_decompose(
    serie_unemp,
    model='additive',
    period=PERIOD,
    extrapolate_trend='freq'
)

fig2 = decomp_unemp.plot()
fig2.set_size_inches(10, 8)
fig2.suptitle(f"Decomposição Desemprego {country_name} (Anual)", fontsize=14)
plt.tight_layout()
plt.show()