import os
import kagglehub as kh
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

# ==========================================
# 1. DOWNLOAD E LEITURA DOS DADOS
# ==========================================

path = kh.dataset_download("marcelomizuno/unemployment-rates")
path2 = kh.dataset_download("sazidthe1/global-inflation-data")

csv_path = os.path.join(path, 'unemployment_rates.csv')
csv_path2 = os.path.join(path2, 'global_inflation_data.csv')

# Ler CSV de Desemprego (Dados Mensais)
df_unemp = pd.read_csv(csv_path, delimiter=',', index_col='TIME_PERIOD', parse_dates=True)
df_unemp.index.name = 'DATE'
# Definir frequência explícita de Início de Mês -- MS - Month Start
df_unemp = df_unemp.sort_index().asfreq('MS')

# Ler CSV de Inflação (Dados Anuais)
df_inflation = pd.read_csv(csv_path2)

# Criar diretório de saída - importante para nossa análise dos datasets
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)

# Salvar CSVs
df_unemp.to_csv(os.path.join(output_dir, 'unemployment_rates.csv'), index=False)
df_inflation.to_csv(os.path.join(output_dir, 'global_inflation_data.csv'), index=False)

# ==========================================
# 2. TRATAMENTO DE DADOS
# ==========================================

# Filtrar indicador principal e remover coluna inútil - não usaremos indicadores
indicator = df_inflation['indicator_name'].unique()[0]
df_inflation = df_inflation[df_inflation['indicator_name'] == indicator].drop(columns=['indicator_name'])

# Melt: Transformar colunas de anos em linhas - alongar
df_inflation_long = df_inflation.melt(
    id_vars=['country_name'],
    var_name='year',
    value_name='inflation'
)
df_inflation_long['year'] = pd.to_datetime(df_inflation_long['year'], format='%Y')

# Pivot: Transformar países em colunas
df_inflation_util = df_inflation_long.pivot(
    index='year',
    columns='country_name',
    values='inflation'
)
df_inflation_util.index.name = 'DATE'

# Transformar a inflação em mensal via interpolação para o merge não descartar dados.
df_inflation_monthly = df_inflation_util.resample('MS').interpolate(method='linear')

# --- Merge dos Datasets ---
df_general = df_unemp.merge(
    df_inflation_monthly,
    left_index=True,
    right_index=True,
    how='inner',  # Igual JOIN
    suffixes=('_unemp', '_infl')
)

# Dataset Geral (Merge dos outros dois):
df_general.to_csv(os.path.join(output_dir, 'general_util_corrected.csv'))

# ==========================================
# 3. VISUALIZAÇÃO DAS TIME SERIES
# ==========================================

unemp_cols = [col for col in df_general.columns if col.endswith('_unemp')]
if unemp_cols:
    df_general[unemp_cols].plot(
        figsize=(12, 6),
        title="Taxa de Desemprego — Comparativo",
        xlabel="Ano",
        ylabel="Taxa de Desemprego (%)"
    )
    plt.tight_layout()
    plt.show()

infl_cols = [col for col in df_general.columns if col.endswith('_infl')]
if infl_cols:
    # PRÉ-2000 (Hiperinflação)
    df_pre_2000 = df_general.loc[:'1999-12-01', infl_cols]

    if 'Brazil_infl' in df_pre_2000.columns and df_pre_2000['Brazil_infl'].max() > 1000:
        # Apenas fins de visualização do Brasil
        df_pre_2000_log = df_pre_2000.copy()
        df_pre_2000_log['Brazil_infl'] = np.log1p(df_pre_2000_log['Brazil_infl'])

        df_pre_2000_log.plot(
            figsize=(12, 6),
            title="Inflação — Comparativo (Pré-2000). Brasil em Escala Log.",
            xlabel="Ano",
            ylabel="Inflação (%) (Outros) / Log(1+Inflação) (Brasil)"
        )
    else:
        df_pre_2000.plot(
            figsize=(12, 6),
            title="Inflação — Comparativo (Pré-2000)",
            xlabel="Ano",
            ylabel="Inflação (%)"
        )

    plt.tight_layout()
    plt.show()

    # PÓS-2000 (Inflação Controlada)
    df_post_2000 = df_general.loc['2000-01-01':, infl_cols]
    df_post_2000.plot(
        figsize=(12, 6),
        title="Inflação — Comparativo (Pós-2000)",
        xlabel="Ano",
        ylabel="Inflação (%)"
    )
    plt.tight_layout()
    plt.show()

# INFLAÇÃO NO BRASIL (ANTES E DEPOIS DE 2000)
if 'Brazil_infl' in df_general.columns:
    brazil_infl = df_general['Brazil_infl'].copy()

    # Período 1: Hiperinflação (1970 até 1999)
    start_year = '1970'
    end_year = '1999'
    infl_hyper = brazil_infl.loc[start_year:end_year]

    # Plot 1: Hiperinflação
    plt.figure(figsize=(12, 6))
    infl_hyper.plot(
        title=f"Inflação Mensal no Brasil ({start_year}-{end_year}) - Hipervolatilidade",
        xlabel="Ano",
        ylabel="Inflação (%)"
    )
    plt.tight_layout()
    plt.show()

    # Período 2: Pós-Estabilização (2000 em diante)
    start_year_stab = '2000'
    infl_stable = brazil_infl.loc[start_year_stab:]

    # Plot 2: Pós-Estabilização
    plt.figure(figsize=(12, 6))
    infl_stable.plot(
        title=f"Inflação Mensal no Brasil (A partir de {start_year_stab}) - Estabilizada",
        xlabel="Ano",
        ylabel="Inflação (%)"
    )
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. DECOMPOSIÇÃO DAS TIME SERIES
# ==========================================

# Extrair lista de países únicos
countries = list(set([col.replace('_infl', '').replace('_unemp', '')
                      for col in df_general.columns]))

for country_name in countries:
    country_infl = f'{country_name}_infl'
    country_unemp = f'{country_name}_unemp'

    # Verificar se ambas as colunas existem
    if country_infl not in df_general.columns or country_unemp not in df_general.columns:
        continue

    serie_infl = df_general[country_infl].copy()
    serie_unemp = df_general[country_unemp].copy()

    # Renomear séries para limpar a legenda no gráfico de decomposição
    serie_infl.name = country_name
    serie_unemp.name = country_name

    # Preencher pequenos gaps(vazios) se existirem (ffill/bfill)
    serie_infl = serie_infl.ffill().bfill()
    serie_unemp = serie_unemp.ffill().bfill()

    # --- Decomposição da INFLAÇÃO ---

    # Verificar necessidade de Log (ex: Brasil anos 80/90)
    log_note = ""
    if serie_infl.max() > 50:
        serie_infl = np.log1p(serie_infl)
        log_note = "(Escala Log)"

    # Período: 12 meses
    decomp_infl = seasonal_decompose(
        serie_infl,
        model='additive',
        period=12,
        extrapolate_trend='freq'
    )

    fig1 = decomp_infl.plot()
    fig1.set_size_inches(10, 8)
    fig1.suptitle(f"Decomposição Inflação {log_note}", fontsize=14)
    plt.tight_layout()
    plt.show()

    # --- Decomposição do DESEMPREGO ---

    # Período: 12 meses
    decomp_unemp = seasonal_decompose(
        serie_unemp,
        model='additive',
        period=12,
        extrapolate_trend='freq'  # Evitar NaN
    )

    fig2 = decomp_unemp.plot()
    fig2.set_size_inches(10, 8)
    fig2.suptitle(f"Decomposição Desemprego", fontsize=14)
    plt.tight_layout()
    plt.show()

'''

1. INFLAÇÃO:
   a. Tendência: 
   b. Sazonalidade: Não há sazonalidade real observável pois os dados originais são anuais.
   c. Ciclo: Sim.

2. DESEMPREGO:
   a. Tendência: 
   b. Sazonalidade: Geralmente sim, contratações de fim de ano, demissões em início de ano, etc.
   c. Ciclo: Sim.

'''