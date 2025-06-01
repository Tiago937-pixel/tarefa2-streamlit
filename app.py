# <<< INÍCIO DO CÓDIGO PARA COLAR EM UM ARQUIVO CHAMADO app.py >>>
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# --- Configuração da Página ---
st.set_page_config(layout="wide", page_title="Análise de Preços Imobiliários (Ames Housing)")

# --- Função de carregamento usando CSV local ---
@st.cache_data  # Cache para não recarregar os dados a cada interação
def load_data():
    """
    Carrega o dataset Ames Housing lendo um arquivo CSV local em 'AmesHousing.csv'
    que esteja na raiz do repositório. Se o arquivo não existir, retorna DataFrame vazio
    e exibe mensagem de erro.
    """
    # 1) Caminho relativo para o CSV na raiz do repositório
    local_csv = "AmesHousing.csv"

    # 2) Verifica se o arquivo existe
    if not os.path.isfile(local_csv):
        st.error(f"Arquivo não encontrado: {local_csv}")
        st.info("Coloque o arquivo 'AmesHousing.csv' na raiz do projeto (junto ao app.py).")
        return pd.DataFrame()

    try:
        # 3) Lê o CSV inteiro para um pandas DataFrame
        df = pd.read_csv(local_csv)
        st.success("Dataset Ames Housing carregado com sucesso (arquivo local)!")
    except Exception as e:
        st.error(f"Erro ao ler o arquivo local CSV: {e}")
        return pd.DataFrame()

    # 4) Tratamento básico de valores ausentes
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            df[col].fillna('Missing', inplace=True)

    # 5) Renomeação opcional de colunas, se desejar manter o padrão antigo
    df.rename(columns={
        'Overall Qual': 'OverallQual',
        'Gr Liv Area': 'GrLivArea',
        'Total Bsmt SF': 'TotalBsmtSF',
        '1st Flr SF': 'FirstFlrSF',
        'Bsmt Qual': 'BsmtQual',
        'Kitchen Qual': 'KitchenQual',
        'Fireplace Qu': 'FireplaceQu',
        'Garage Type': 'GarageType',
        'Garage Cars': 'GarageCars',
        'Garage Area': 'GarageArea',
        'Central Air': 'CentralAir',
        'MS Zoning': 'MSZoning',
        'House Style': 'HouseStyle'
    }, inplace=True, errors='ignore')

    return df

# Carrega o DataFrame (será lido do CSV local)
df_ames = load_data()

# --- Título e aviso caso não carregue dados ---
st.title("🏠 Dashboard de Análise de Preços Imobiliários")
st.markdown("Dataset: Ames Housing")

if df_ames.empty:
    st.warning("Dataset não pôde ser carregado. Funcionalidades limitadas.")
else:
    # --- Barra Lateral para Entradas do Usuário ---
    st.sidebar.header("Opções de Análise")

    analysis_type = st.sidebar.selectbox(
        "Escolha o Tipo de Análise:",
        ["Visão Geral dos Dados", "Análise ANOVA", "Análise de Regressão Linear"]
    )

    # --- Visão Geral dos Dados ---
    if analysis_type == "Visão Geral dos Dados":
        st.header("Visão Geral dos Dados")
        st.markdown(f"O dataset possui {df_ames.shape[0]} observações e {df_ames.shape[1]} variáveis.")
        
        st.subheader("Primeiras Linhas do Dataset:")
        st.dataframe(df_ames.head())

        st.subheader("Estatísticas Descritivas (Variáveis Numéricas):")
        st.dataframe(df_ames.describe(include=np.number))

        if 'SalePrice' in df_ames.columns:
            st.subheader("Distribuição do Preço de Venda (SalePrice)")
            fig_price_hist = px.histogram(
                df_ames, x='SalePrice', nbins=70, 
                title="Distribuição do Preço de Venda"
            )
            st.plotly_chart(fig_price_hist, use_container_width=True)
        else:
            st.warning("Coluna 'SalePrice' não encontrada no dataset.")

        st.subheader("Heatmap de Correlação (Variáveis Numéricas)")
        numeric_cols = df_ames.select_dtypes(include=np.number).columns
        if not df_ames[numeric_cols].empty:
            corr_matrix = df_ames[numeric_cols].corr()
            fig_corr = px.imshow(
                corr_matrix, title="Heatmap de Correlação",
                text_auto=".2f", aspect="auto", 
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.write("Nenhuma coluna numérica encontrada para calcular a correlação.")

    # --- Análise ANOVA ---
    elif analysis_type == "Análise ANOVA":
        st.header("I. Análise Exploratória e Comparativa com ANOVA")
        st.markdown("""
        Aqui investigamos se existem diferenças significativas no preço médio de venda (`SalePrice`) 
        entre as categorias de variáveis selecionadas.
        """)

        if 'SalePrice' not in df_ames.columns:
            st.error("A coluna 'SalePrice' é necessária para a análise ANOVA e não foi encontrada.")
        else:
            # Seleção de variáveis categóricas para ANOVA
            all_cat_vars = [
                col for col in df_ames.select_dtypes(include='object').columns 
                if df_ames[col].nunique() < 30 and df_ames[col].nunique() > 1
            ]
            
            default_anova_vars = []
            common_vars_for_anova = [
                'OverallQual', 'Neighborhood', 'CentralAir', 
                'KitchenQual', 'MSZoning', 'HouseStyle', 'ExterQual', 'BsmtQual'
            ]
            # Caso 'OverallQual' seja numérico mas com poucas categorias, trata como string
            if 'OverallQual' in df_ames.columns and df_ames['OverallQual'].dtype != 'object' and df_ames['OverallQual'].nunique() < 15:
                if 'OverallQual' not in all_cat_vars:
                    all_cat_vars.insert(0, 'OverallQual')

            for var_name in common_vars_for_anova:
                if var_name in all_cat_vars or (var_name == 'OverallQual' and var_name in df_ames.columns):
                    default_anova_vars.append(var_name)
                if len(default_anova_vars) >= 3:
                    break
            if not default_anova_vars and all_cat_vars:
                default_anova_vars = all_cat_vars[:min(3, len(all_cat_vars))]

            selected_cat_vars_anova = st.sidebar.multiselect(
                "Escolha 2-3 variáveis para ANOVA:",
                options=all_cat_vars,
                default=default_anova_vars
            )

            if not selected_cat_vars_anova:
                st.warning("Por favor, selecione pelo menos uma variável categórica para a análise ANOVA.")
            else:
                for cat_var in selected_cat_vars_anova:
                    st.subheader(f"ANOVA para: {cat_var} vs. SalePrice")

                    anova_data = df_ames[['SalePrice', cat_var]].dropna()
                    
                    if anova_data[cat_var].dtype != 'object':
                        anova_data[cat_var] = anova_data[cat_var].astype(str)

                    fig_boxplot = px.box(
                        anova_data,
                        x=cat_var,
                        y='SalePrice',
                        title=f'Preço de Venda por {cat_var}',
                        color=cat_var
                    )
                    st.plotly_chart(fig_boxplot, use_container_width=True)

                    # Agrupa valores para cada categoria
                    groups = [
                        anova_data['SalePrice'][anova_data[cat_var] == val] 
                        for val in anova_data[cat_var].unique()
                    ]
                    groups_for_test = [g for g in groups if len(g) >= 2]

                    if len(groups_for_test) < 2:
                        st.warning(f"Não há grupos suficientes (mínimo 2 com >=2 obs.) para ANOVA na variável '{cat_var}'. Pulando.")
                        continue

                    # Teste F de ANOVA
                    f_statistic, p_value_anova = stats.f_oneway(*groups_for_test)
                    st.write(f"**Resultado ANOVA:** F-Statistic = {f_statistic:.4f}, p-valor = {p_value_anova:.4g}")

                    alpha = 0.05
                    if p_value_anova < alpha:
                        st.success(
                            f"O p-valor ({p_value_anova:.4g}) é menor que {alpha}. "
                            f"Há uma diferença estatisticamente significativa no preço médio de venda "
                            f"entre as diferentes categorias de '{cat_var}'."
                        )
                    else:
                        st.info(
                            f"O p-valor ({p_value_anova:.4g}) é maior ou igual a {alpha}. "
                            f"Não há evidência de uma diferença estatisticamente significativa no preço médio de venda "
                            f"entre as diferentes categorias de '{cat_var}'."
                        )

                    st.markdown("**Verificação das premissas da ANOVA:**")
                    model_ols = ols(f'SalePrice ~ C({cat_var})', data=anova_data).fit()
                    residuals = model_ols.resid

                    # 1. Normalidade dos Resíduos (Shapiro-Wilk ou Kolmogorov-Smirnov)
                    if len(residuals) >= 3 and len(residuals) <= 5000:
                        shapiro_stat, shapiro_p = stats.shapiro(residuals)
                        st.write(f"*Normalidade dos Resíduos (Shapiro-Wilk):* Estatística={shapiro_stat:.4f}, p-valor={shapiro_p:.4g}")
                        if shapiro_p > alpha:
                            st.write("Resíduos parecem ser normalmente distribuídos (p > 0.05).")
                        else:
                            st.write("Resíduos NÃO parecem ser normalmente distribuídos (p <= 0.05). Premissa violada.")
                    elif len(residuals) > 5000:
                        ks_stat, ks_p = stats.kstest(
                            residuals,
                            'norm',
                            args=(np.mean(residuals), np.std(residuals))
                        )
                        st.write(f"*Normalidade dos Resíduos (Kolmogorov-Smirnov):* Estatística={ks_stat:.4f}, p-valor={ks_p:.4g}")
                        if ks_p > alpha:
                            st.write("Resíduos parecem ser normalmente distribuídos (p > 0.05).")
                        else:
                            st.write("Resíduos NÃO parecem ser normalmente distribuídos (p <= 0.05). Premissa violada.")
                    else:
                        st.write("*Normalidade dos Resíduos:* Dados insuficientes para o teste.")

                    # Q-Q Plot dos Resíduos
                    fig_qq, ax_qq = plt.subplots(figsize=(6,4))
                    sm.qqplot(residuals, line='s', ax=ax_qq, fit=True)
                    ax_qq.set_title(f'Q-Q Plot dos Resíduos para {cat_var}')
                    st.pyplot(fig_qq)
                    plt.close(fig_qq)

                    # 2. Homocedasticidade (Levene Test)
                    levene_stat, levene_p = stats.levene(*groups_for_test)
                    st.write(f"*Homocedasticidade das Variâncias (Levene Test):* Estatística={levene_stat:.4f}, p-valor={levene_p:.4g}")
                    if levene_p > alpha:
                        st.write("Variâncias parecem ser homogêneas (p > 0.05).")
                    else:
                        st.write("Variâncias NÃO parecem ser homogêneas (p <= 0.05). Premissa violada (heterocedasticidade).")

                    prem_normalidade_ok = (
                        (len(residuals) >= 3 and 'shapiro_p' in locals() and shapiro_p > alpha)
                        if (len(residuals) >= 3 and len(residuals) <= 5000)
                        else (len(residuals) > 5000 and 'ks_p' in locals() and ks_p > alpha
                              if len(residuals) > 5000 else True)
                    )
                    prem_homocedasticidade_ok = levene_p > alpha

                    # 3. Alternativa Robusta (Kruskal-Wallis) se premissas violadas
                    if not prem_normalidade_ok or not prem_homocedasticidade_ok:
                        st.markdown("**Alternativa Robusta (Kruskal-Wallis Test):** Como as premissas da ANOVA podem ter sido violadas.")
                        kruskal_stat, kruskal_p = stats.kruskal(*groups_for_test)
                        st.write(f"*Kruskal-Wallis Test:* H-Statistic={kruskal_stat:.4f}, p-valor={kruskal_p:.4g}")
                        if kruskal_p < alpha:
                            st.success(
                                f"Kruskal-Wallis indica uma diferença significativa (p < {alpha}) "
                                f"na mediana do preço de venda entre os grupos de '{cat_var}'."
                            )
                        else:
                            st.info(
                                f"Kruskal-Wallis NÃO indica uma diferença significativa (p >= {alpha}) "
                                f"na mediana do preço de venda entre os grupos de '{cat_var}'."
                            )

                    # 4. Post‐hoc (Tukey HSD) se ANOVA significativa e premissas ok
                    if p_value_anova < alpha and prem_normalidade_ok and prem_homocedasticidade_ok:
                        st.markdown("**Teste Post-hoc (Tukey HSD):**")
                        tukey_results = pairwise_tukeyhsd(anova_data['SalePrice'], anova_data[cat_var], alpha=alpha)
                        st.text(str(tukey_results))
                        st.caption(f"""
                        Interpretação do Tukey HSD para '{cat_var}':
                        A tabela acima mostra comparações par a par. 'reject = True' indica uma diferença significativa 
                        no preço médio de venda entre os dois grupos comparados.
                        """)
                    st.markdown("---")  # Separador

    # --- Análise de Regressão Linear ---
    elif analysis_type == "Análise de Regressão Linear":
        st.header("II. Análise de Regressão Linear para Previsão de Preços")
        st.markdown("""
        Construímos um modelo de regressão linear para prever o preço de venda (`SalePrice`) 
        com base em variáveis independentes selecionadas.
        """)

        if 'SalePrice' not in df_ames.columns:
            st.error("A coluna 'SalePrice' é necessária para a análise de Regressão e não foi encontrada.")
        else:
            # Seleção de variáveis para Regressão
            all_potential_predictors = [
                col for col in df_ames.columns 
                if col not in ['SalePrice', 'Order', 'PID']
            ]
            
            numerical_predictors = df_ames[all_potential_predictors].select_dtypes(include=np.number).columns.tolist()
            categorical_predictors_reg = df_ames[all_potential_predictors].select_dtypes(include='object').columns.tolist()

            st.sidebar.markdown("**Configurações da Regressão:**")
            
            default_num_reg_vars = []
            common_num_vars_reg = [
                'GrLivArea', 'OverallQual', 'TotalBsmtSF', 
                'YearBuilt', 'FirstFlrSF', 'GarageCars', 'GarageArea'
            ]
            for var in common_num_vars_reg:
                if var in numerical_predictors:
                    default_num_reg_vars.append(var)

            selected_numerical_vars_reg = st.sidebar.multiselect(
                "Selecione preditores numéricos:",
                options=numerical_predictors,
                default=default_num_reg_vars[:min(5, len(default_num_reg_vars))]
            )
            
            default_cat_reg_vars = []
            common_cat_vars_reg = ['Neighborhood', 'MSZoning', 'CentralAir', 'KitchenQual']
            for var in common_cat_vars_reg:
                if var in categorical_predictors_reg and df_ames[var].nunique() < 15:
                    default_cat_reg_vars.append(var)

            selected_categorical_vars_reg = st.sidebar.multiselect(
                "Selecione preditores categóricos (serão 'dummificados'):",
                options=[var for var in categorical_predictors_reg if df_ames[var].nunique() < 20],
                default=default_cat_reg_vars[:min(2, len(default_cat_reg_vars))]
            )

            if not selected_numerical_vars_reg and not selected_categorical_vars_reg:
                st.warning("Por favor, selecione pelo menos uma variável preditora.")
            else:
                X_vars = selected_numerical_vars_reg + selected_categorical_vars_reg
                reg_df = df_ames[['SalePrice'] + X_vars].copy().dropna()

                if reg_df.shape[0] < len(X_vars) + 2:
                    st.error("Dados insuficientes após remover NaNs para as variáveis selecionadas.")
                else:
                    X = reg_df[X_vars]
                    y = reg_df['SalePrice']

                    if selected_categorical_vars_reg:
                        X = pd.get_dummies(
                            X, 
                            columns=selected_categorical_vars_reg, 
                            drop_first=True, 
                            dtype=float
                        )

                    X = sm.add_constant(X.astype(float))  # Adicionar intercepto e garantir float

                    try:
                        model = sm.OLS(y, X).fit()
                        st.subheader("Sumário do Modelo de Regressão (OLS)")
                        st.text(model.summary())

                        st.markdown("**Interpretação do Sumário:**")
                        st.markdown(f"""
                        - **R-squared (R²):** {model.rsquared:.3f}. Indica que ~{model.rsquared*100:.1f}% da variância em `SalePrice` é explicada pelos preditores.  
                        - **Adj. R-squared:** {model.rsquared_adj:.3f}. R² ajustado pela quantidade de preditores.  
                        - **P>|t| (p-valor dos coeficientes):** Se < 0.05, o preditor é estatisticamente significativo.  
                        - **Coeficientes (coef):** Mostra a mudança esperada em `SalePrice` para um aumento unitário no preditor, mantendo outros constantes.  
                        """)

                        st.subheader("Diagnóstico dos Resíduos da Regressão")
                        residuals_reg = model.resid
                        fitted_values_reg = model.fittedvalues

                        # 1. Resíduos vs. Ajustados
                        fig_res_fit, ax_res_fit = plt.subplots(figsize=(8,5))
                        sns.scatterplot(x=fitted_values_reg, y=residuals_reg, ax=ax_res_fit, alpha=0.6)
                        ax_res_fit.axhline(0, color='red', linestyle='--')
                        ax_res_fit.set_xlabel("Valores Ajustados")
                        ax_res_fit.set_ylabel("Resíduos")
                        ax_res_fit.set_title("Resíduos vs. Valores Ajustados")
                        st.pyplot(fig_res_fit)
                        plt.close(fig_res_fit)
                        st.caption("Ideal: pontos aleatórios em torno da linha zero, sem padrões (como funil ou curva).")

                        # 2. Q-Q Plot dos Resíduos
                        fig_qq_reg, ax_qq_reg = plt.subplots(figsize=(6,4))
                        sm.qqplot(residuals_reg, line='s', ax=ax_qq_reg, fit=True)
                        ax_qq_reg.set_title("Q-Q Plot dos Resíduos (Regressão)")
                        st.pyplot(fig_qq_reg)
                        plt.close(fig_qq_reg)
                        st.caption("Ideal: pontos próximos à linha diagonal, indicando normalidade dos resíduos.")

                        # 3. Teste de Homocedasticidade (Breusch-Pagan)
                        try:
                            X_bp_test = X.select_dtypes(include=np.number)
                            bp_test = sm.stats.het_breuschpagan(residuals_reg, X_bp_test)
                            labels_bp = ['Estatística LM', 'p-valor LM', 'Estatística F', 'p-valor F']
                            st.write("**Teste de Homocedasticidade (Breusch-Pagan):**")
                            for name, val in zip(labels_bp, bp_test):
                                st.write(f"- {name}: {val:.4f}")
                            if bp_test[1] < 0.05:
                                st.write("Evidência de Heterocedasticidade (p < 0.05). Variância dos erros não é constante.")
                            else:
                                st.write("Não há evidência significativa de Heterocedasticidade (p >= 0.05).")
                        except Exception as e_bp:
                            st.warning(f"Não foi possível executar o teste de Breusch-Pagan: {e_bp}")

                    except Exception as e:
                        st.error(f"Erro ao ajustar o modelo de regressão: {e}")
                        st.info("Verifique se há multicolinearidade perfeita ou variáveis problemáticas.")

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Sobre este App:**
    Dashboard para análise do dataset Ames Housing usando ANOVA e Regressão Linear. 
    Criado para o Desafio da Tarefa 2.
    """)
# <<< FIM DO CÓDIGO PARA COLAR EM UM ARQUIVO CHAMADO app.py >>>

