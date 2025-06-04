# Dashboard de Análise de Preços Imobiliários - Ames Housing Dataset
# Tarefa 2: Precificação Imobiliária com ANOVA e Regressão Linear
# Professor: João Gabriel de Moraes Souza
# UnB - Departamento de Engenharia de Produção

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Configuração da Página ---
st.set_page_config(
    layout="wide", 
    page_title="Análise Imobiliária - Ames Housing",
    page_icon="🏠"
)

# CSS personalizado inspirado nos dashboards do professor
st.markdown("""
<style>
.stSlider > div > div > div > div > div > div {
    background-color: #4CAF50 !important;
}
.main-header {
    text-align: center;
    color: #003366;
    padding: 1rem 0;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# --- Cabeçalho Principal ---
st.markdown("<h1 class='main-header'>🏠 Dashboard de Análise de Preços Imobiliários</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='main-header'>Ames Housing Dataset - ANOVA e Regressão Linear</h3>", unsafe_allow_html=True)
st.markdown("<p class='main-header'>Professor: João Gabriel de Moraes Souza</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Função de carregamento de dados ---
@st.cache_data
def load_data():
    """
    Carrega e trata o dataset Ames Housing
    """
    local_csv = "AmesHousing.csv"
    
    if not os.path.isfile(local_csv):
        st.error(f"❌ Arquivo não encontrado: {local_csv}")
        st.info("💡 Coloque o arquivo 'AmesHousing.csv' na raiz do projeto.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(local_csv)
        
        # Tratamento de valores ausentes
        for col in df.select_dtypes(include=np.number).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        for col in df.select_dtypes(include='object').columns:
            if df[col].isnull().any():
                df[col].fillna('Missing', inplace=True)

        # Renomeação de colunas para facilitar uso
        column_mapping = {
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
            'House Style': 'HouseStyle',
            'Exter Qual': 'ExterQual'
        }
        df.rename(columns=column_mapping, inplace=True, errors='ignore')
        
        st.success("✅ Dataset Ames Housing carregado com sucesso!")
        return df
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        return pd.DataFrame()

# Carregamento dos dados
df_ames = load_data()

if df_ames.empty:
    st.warning("⚠️ Dataset não carregado. Funcionalidades limitadas.")
    st.stop()

# --- Barra Lateral ---
st.sidebar.title("🛠️ Configurações de Análise")
st.sidebar.markdown("### Selecione o tipo de análise:")

analysis_type = st.sidebar.selectbox(
    "Escolha a Análise:",
    ["📊 Visão Geral dos Dados", "🔬 Análise ANOVA", "📈 Regressão Linear"]
)

# Parâmetros globais
alpha_global = st.sidebar.slider("Nível de significância (α)", 0.01, 0.10, 0.05, 0.01)

# --- Seção 1: Visão Geral dos Dados ---
if analysis_type == "📊 Visão Geral dos Dados":
    st.header("📊 Visão Geral dos Dados")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📋 Observações", f"{df_ames.shape[0]:,}")
    with col2:
        st.metric("📊 Variáveis", f"{df_ames.shape[1]}")
    with col3:
        if 'SalePrice' in df_ames.columns:
            st.metric("💰 Preço Médio", f"${df_ames['SalePrice'].mean():,.0f}")
    
    # Abas para organizar melhor
    tab1, tab2, tab3 = st.tabs(["📋 Dados", "📊 Estatísticas", "📈 Visualizações"])
    
    with tab1:
        st.subheader("Primeiras Linhas do Dataset")
        st.dataframe(df_ames.head(10), use_container_width=True)
        
        # Info sobre tipos de dados
        st.subheader("Informações sobre Variáveis")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Variáveis Numéricas:**")
            numeric_cols = df_ames.select_dtypes(include=np.number).columns.tolist()
            st.write(f"Total: {len(numeric_cols)}")
            st.write(numeric_cols[:10])  # Primeiras 10
            
        with col2:
            st.write("**Variáveis Categóricas:**")
            cat_cols = df_ames.select_dtypes(include='object').columns.tolist()
            st.write(f"Total: {len(cat_cols)}")
            st.write(cat_cols[:10])  # Primeiras 10
    
    with tab2:
        st.subheader("Estatísticas Descritivas - Variáveis Numéricas")
        st.dataframe(df_ames.describe(include=np.number), use_container_width=True)
        
        # Estatísticas das categóricas
        st.subheader("Estatísticas - Variáveis Categóricas (Top 5 categorias)")
        cat_cols = df_ames.select_dtypes(include='object').columns
        for col in cat_cols[:5]:  # Primeiras 5 categóricas
            st.write(f"**{col}:**")
            st.write(df_ames[col].value_counts().head())
    
    with tab3:
        if 'SalePrice' in df_ames.columns:
            st.subheader("Distribuição do Preço de Venda (SalePrice)")
            
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(
                    df_ames, x='SalePrice', nbins=50,
                    title="Distribuição Normal do Preço",
                    labels={'SalePrice': 'Preço de Venda ($)', 'count': 'Frequência'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Log do preço para melhor visualização
                df_temp = df_ames.copy()
                df_temp['LogSalePrice'] = np.log(df_temp['SalePrice'])
                fig_log = px.histogram(
                    df_temp, x='LogSalePrice', nbins=50,
                    title="Distribuição Log do Preço",
                    labels={'LogSalePrice': 'Log(Preço)', 'count': 'Frequência'}
                )
                st.plotly_chart(fig_log, use_container_width=True)
        
        # Heatmap de correlação
        st.subheader("Matriz de Correlação (Top 15 Variáveis Numéricas)")
        numeric_cols = df_ames.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            # Selecionar as 15 variáveis mais correlacionadas com SalePrice
            if 'SalePrice' in numeric_cols:
                corr_with_price = df_ames[numeric_cols].corr()['SalePrice'].abs().sort_values(ascending=False)
                top_cols = corr_with_price.head(15).index.tolist()
            else:
                top_cols = numeric_cols[:15].tolist()
            
            corr_matrix = df_ames[top_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title="Heatmap de Correlação",
                color_continuous_scale="RdBu",
                aspect="auto",
                text_auto=".2f"
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)

# --- Seção 2: Análise ANOVA ---
elif analysis_type == "🔬 Análise ANOVA":
    st.header("🔬 I. Análise Exploratória e Comparativa com ANOVA")
    st.markdown("""
    **Objetivo:** Investigar se existem diferenças significativas no preço médio de venda (`SalePrice`) 
    entre categorias de variáveis selecionadas, seguindo a metodologia da Tarefa 2.
    """)

    if 'SalePrice' not in df_ames.columns:
        st.error("❌ A coluna 'SalePrice' é necessária para ANOVA.")
        st.stop()

    # Configurações da ANOVA na sidebar
    st.sidebar.markdown("### 🔬 Configurações ANOVA")
    
    # Seleção de variáveis categóricas
    all_cat_vars = []
    for col in df_ames.columns:
        if col != 'SalePrice':
            if df_ames[col].dtype == 'object' and df_ames[col].nunique() < 15 and df_ames[col].nunique() > 1:
                all_cat_vars.append(col)
            elif col in ['OverallQual', 'OverallCond'] and df_ames[col].nunique() < 15:
                all_cat_vars.append(col)

    # Variáveis sugeridas baseadas na tarefa
    suggested_vars = ['OverallQual', 'Neighborhood', 'CentralAir', 'KitchenQual', 'ExterQual', 'BsmtQual']
    default_vars = [var for var in suggested_vars if var in all_cat_vars][:3]

    selected_cat_vars = st.sidebar.multiselect(
        "Escolha 2-3 variáveis categóricas:",
        options=all_cat_vars,
        default=default_vars,
        help="Selecione variáveis como tipo de bairro, qualidade do acabamento, etc."
    )

    if not selected_cat_vars:
        st.warning("⚠️ Selecione pelo menos uma variável categórica para análise.")
        st.stop()

    # Opções de testes
    st.sidebar.markdown("### 🧪 Opções de Testes")
    show_assumptions = st.sidebar.checkbox("Verificar pressupostos ANOVA", value=True)
    show_robust_tests = st.sidebar.checkbox("Executar testes robustos", value=True)
    show_posthoc = st.sidebar.checkbox("Teste Post-hoc (Tukey)", value=True)

    # Análise para cada variável selecionada
    for i, cat_var in enumerate(selected_cat_vars):
        st.markdown(f"## {i+1}. Análise: {cat_var} vs. SalePrice")
        
        # Preparação dos dados
        anova_data = df_ames[['SalePrice', cat_var]].dropna()
        
        # Converter para string se necessário
        if anova_data[cat_var].dtype != 'object':
            anova_data[cat_var] = anova_data[cat_var].astype(str)

        # Filtrar grupos com pelo menos 2 observações
        group_counts = anova_data[cat_var].value_counts()
        valid_groups = group_counts[group_counts >= 2].index
        anova_data = anova_data[anova_data[cat_var].isin(valid_groups)]

        if len(valid_groups) < 2:
            st.warning(f"⚠️ Grupos insuficientes para '{cat_var}'. Pulando análise.")
            continue

        # Estatísticas descritivas por grupo
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Boxplot interativo
            fig_box = px.box(
                anova_data, x=cat_var, y='SalePrice',
                title=f'Distribuição do Preço por {cat_var}',
                color=cat_var,
                labels={'SalePrice': 'Preço de Venda ($)'}
            )
            fig_box.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_box, use_container_width=True)

        with col2:
            # Estatísticas por grupo
            st.markdown("**Estatísticas por Grupo:**")
            grupo_stats = anova_data.groupby(cat_var)['SalePrice'].agg(['count', 'mean', 'std']).round(2)
            st.dataframe(grupo_stats, use_container_width=True)

        # Teste ANOVA F
        st.markdown("### 📊 Teste ANOVA (F-Test)")
        groups = [anova_data['SalePrice'][anova_data[cat_var] == val] for val in valid_groups]
        f_statistic, p_value_anova = stats.f_oneway(*groups)
        
        # Resultados em métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("F-Statistic", f"{f_statistic:.4f}")
        with col2:
            st.metric("p-valor", f"{p_value_anova:.4g}")
        with col3:
            is_significant = p_value_anova < alpha_global
            st.metric("Resultado", "✅ Significativo" if is_significant else "❌ Não Significativo")

        # Interpretação
        if is_significant:
            st.success(f"""
            **✅ Resultado Significativo (p = {p_value_anova:.4g} < {alpha_global})**
            
            Há diferença estatisticamente significativa no preço médio de venda entre as categorias de '{cat_var}'.
            Essa variável impacta significativamente na precificação dos imóveis.
            """)
        else:
            st.info(f"""
            **ℹ️ Resultado Não Significativo (p = {p_value_anova:.4g} ≥ {alpha_global})**
            
            Não há evidência de diferença significativa no preço médio entre as categorias de '{cat_var}'.
            """)

        # Verificação de pressupostos
        if show_assumptions:
            st.markdown("### 🔍 Verificação dos Pressupostos da ANOVA")
            
            # Ajustar modelo para análise de resíduos
            model_ols = ols(f'SalePrice ~ C({cat_var})', data=anova_data).fit()
            residuals = model_ols.resid
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 1. Teste de Normalidade
                st.markdown("**1. Normalidade dos Resíduos**")
                if len(residuals) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(residuals)
                    st.write(f"Shapiro-Wilk: W = {shapiro_stat:.4f}, p = {shapiro_p:.4g}")
                    if shapiro_p > alpha_global:
                        st.success("✅ Resíduos são normais (p > α)")
                    else:
                        st.warning("⚠️ Resíduos NÃO são normais (p ≤ α)")
                else:
                    ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
                    st.write(f"Kolmogorov-Smirnov: D = {ks_stat:.4f}, p = {ks_p:.4g}")
                    if ks_p > alpha_global:
                        st.success("✅ Resíduos são normais (p > α)")
                    else:
                        st.warning("⚠️ Resíduos NÃO são normais (p ≤ α)")

            with col2:
                # 2. Teste de Homocedasticidade
                st.markdown("**2. Homocedasticidade (Levene)**")
                levene_stat, levene_p = stats.levene(*groups)
                st.write(f"Levene: W = {levene_stat:.4f}, p = {levene_p:.4g}")
                if levene_p > alpha_global:
                    st.success("✅ Variâncias homogêneas (p > α)")
                else:
                    st.warning("⚠️ Heterocedasticidade detectada (p ≤ α)")

            # Q-Q Plot
            fig_qq, ax_qq = plt.subplots(figsize=(8, 5))
            sm.qqplot(residuals, line='s', ax=ax_qq, fit=True)
            ax_qq.set_title(f'Q-Q Plot dos Resíduos - {cat_var}')
            st.pyplot(fig_qq)
            plt.close(fig_qq)

        # Testes Robustos
        if show_robust_tests:
            st.markdown("### 🛡️ Testes Não-Paramétricos (Robustos)")
            
            # Kruskal-Wallis
            kruskal_stat, kruskal_p = stats.kruskal(*groups)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Kruskal-Wallis H", f"{kruskal_stat:.4f}")
            with col2:
                st.metric("p-valor KW", f"{kruskal_p:.4g}")
            
            if kruskal_p < alpha_global:
                st.success(f"✅ Kruskal-Wallis: Diferença significativa nas medianas (p = {kruskal_p:.4g})")
            else:
                st.info(f"ℹ️ Kruskal-Wallis: Sem diferença significativa (p = {kruskal_p:.4g})")

        # Post-hoc Tukey
        if show_posthoc and is_significant:
            st.markdown("### 🎯 Teste Post-hoc (Tukey HSD)")
            st.info("Executado apenas quando ANOVA é significativa")
            
            try:
                tukey_results = pairwise_tukeyhsd(anova_data['SalePrice'], anova_data[cat_var], alpha=alpha_global)
                st.text(str(tukey_results))
                
                # Interpretação do Tukey
                tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])
                significant_pairs = tukey_df[tukey_df['reject'] == True]
                
                if len(significant_pairs) > 0:
                    st.success(f"✅ {len(significant_pairs)} comparações par a par são significativas:")
                    for _, row in significant_pairs.iterrows():
                        st.write(f"• **{row['group1']} vs {row['group2']}**: diferença = {row['meandiff']:.0f}")
                else:
                    st.info("ℹ️ Nenhuma comparação par a par é significativa após correção.")
                    
            except Exception as e:
                st.error(f"Erro no teste Tukey: {e}")

        st.markdown("---")  # Separador entre variáveis

# --- Seção 3: Regressão Linear ---
elif analysis_type == "📈 Regressão Linear":
    st.header("📈 II. Modelagem Preditiva com Regressão Linear")
    st.markdown("""
    **Objetivo:** Construir um modelo de regressão linear múltipla para prever o preço de venda (`SalePrice`) 
    seguindo as especificações da Tarefa 2.
    """)

    if 'SalePrice' not in df_ames.columns:
        st.error("❌ A coluna 'SalePrice' é necessária para Regressão.")
        st.stop()

    # Configurações da Regressão na sidebar
    st.sidebar.markdown("### 📈 Configurações da Regressão")
    
    # Seleção de variáveis
    numerical_predictors = df_ames.select_dtypes(include=np.number).columns.tolist()
    numerical_predictors = [col for col in numerical_predictors if col not in ['SalePrice', 'Order', 'PID']]
    
    categorical_predictors = df_ames.select_dtypes(include='object').columns.tolist()
    categorical_predictors = [col for col in categorical_predictors if df_ames[col].nunique() < 20]

    # Variáveis sugeridas baseadas na importância para preços imobiliários
    suggested_num = ['GrLivArea', 'OverallQual', 'TotalBsmtSF', 'YearBuilt', 'GarageCars']
    suggested_cat = ['Neighborhood', 'CentralAir', 'KitchenQual', 'ExterQual']
    
    default_num = [var for var in suggested_num if var in numerical_predictors][:4]
    default_cat = [var for var in suggested_cat if var in categorical_predictors][:2]

    selected_numerical = st.sidebar.multiselect(
        "Variáveis Numéricas (4-6 total):",
        options=numerical_predictors,
        default=default_num,
        help="Escolha variáveis contínuas como área, ano de construção, etc."
    )

    selected_categorical = st.sidebar.multiselect(
        "Variáveis Categóricas (1-2):",
        options=categorical_predictors,
        default=default_cat,
        help="Escolha variáveis categóricas (serão convertidas em dummies)"
    )

    # Opções de modelagem
    st.sidebar.markdown("### 🔧 Opções de Modelagem")
    use_log_transform = st.sidebar.checkbox("Transformação Log-Log", value=True, 
                                          help="Aplica log na variável dependente e nas contínuas")
    include_constant = st.sidebar.checkbox("Incluir Intercepto", value=True)
    use_robust = st.sidebar.checkbox("Modelo Robusto (HC0)", value=False,
                                   help="Correção para heterocedasticidade")

    # Validação da seleção
    total_vars = len(selected_numerical) + len(selected_categorical)
    if total_vars < 4 or total_vars > 6:
        st.warning(f"⚠️ Selecione entre 4-6 variáveis (atual: {total_vars})")
        st.stop()
    
    if len(selected_numerical) == 0:
        st.warning("⚠️ Selecione pelo menos uma variável contínua")
        st.stop()
    
    if len(selected_categorical) == 0:
        st.warning("⚠️ Selecione pelo menos uma variável categórica")
        st.stop()

    # Preparação dos dados
    X_vars = selected_numerical + selected_categorical
    reg_df = df_ames[['SalePrice'] + X_vars].copy().dropna()
    
    if reg_df.shape[0] < 50:
        st.error("❌ Dados insuficientes após limpeza")
        st.stop()

    st.success(f"✅ Dataset preparado: {reg_df.shape[0]} observações, {len(X_vars)} preditores")

    # Abas para organizar a análise
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Preparação", "📊 Modelo", "🔍 Diagnósticos", "💡 Interpretação"])

    with tab1:
        st.subheader("🔧 Preparação dos Dados")
        
        # Mostrar correlações com SalePrice
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Correlações com SalePrice:**")
            correlations = reg_df[selected_numerical + ['SalePrice']].corr()['SalePrice'].sort_values(ascending=False)
            correlations = correlations.drop('SalePrice')  # Remove auto-correlação
            st.dataframe(correlations.to_frame('Correlação'), use_container_width=True)

        with col2:
            st.markdown("**Estatísticas das Variáveis Categóricas:**")
            for cat_var in selected_categorical:
                st.write(f"**{cat_var}:**")
                st.write(f"Categorias únicas: {reg_df[cat_var].nunique()}")
                st.write(reg_df[cat_var].value_counts().head(3))

        # Aplicar transformações
        y = reg_df['SalePrice'].copy()
        X = reg_df[X_vars].copy()

        if use_log_transform:
            st.markdown("**🔄 Aplicando Transformação Log-Log**")
            y = np.log(y)
            
            # Log nas variáveis numéricas (evitando zeros)
            for col in selected_numerical:
                X[col] = np.log(X[col].replace(0, 1))
            
            st.info("✅ Transformação logarítmica aplicada em SalePrice e variáveis numéricas")

        # Criar variáveis dummy
        if selected_categorical:
            X_dummies = pd.get_dummies(X[selected_categorical], drop_first=True, dtype=float)
            X = pd.concat([X[selected_numerical], X_dummies], axis=1)
            st.info(f"✅ Variáveis dummy criadas: {X_dummies.shape[1]} novas variáveis")

        # Adicionar constante
        if include_constant:
            X = sm.add_constant(X.astype(float))

        # Mostrar dados finais
        st.markdown("**📋 Dados Finais para Modelagem:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Variável dependente: {y.name} (transformada: {'Log' if use_log_transform else 'Original'})")
            st.write(f"Observações: {len(y)}")
        with col2:
            st.write(f"Preditores: {X.shape[1]}")
            st.write(f"Nomes: {list(X.columns)}")

    with tab2:
        st.subheader("📊 Ajuste do Modelo")
        
        try:
            # Ajustar modelo
            if use_robust:
                model = sm.OLS(y, X).fit(cov_type='HC0')
                st.info("🛡️ Modelo robusto (HC0) para heterocedasticidade")
            else:
                model = sm.OLS(y, X).fit()
                st.info("📊 Modelo OLS padrão")

            # Métricas do modelo
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R²", f"{model.rsquared:.4f}")
            with col2:
                st.metric("R² Ajustado", f"{model.rsquared_adj:.4f}")
            with col3:
                # RMSE
                y_pred = model.predict(X)
                if use_log_transform:
                    # Converter de volta para escala original
                    y_actual_orig = np.exp(y)
                    y_pred_orig = np.exp(y_pred)
                    rmse = np.sqrt(mean_squared_error(y_actual_orig, y_pred_orig))
                    st.metric("RMSE", f"${rmse:,.0f}")
                else:
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    st.metric("RMSE", f"${rmse:,.0f}")
            with col4:
                # MAE
                if use_log_transform:
                    mae = mean_absolute_error(y_actual_orig, y_pred_orig)
                    st.metric("MAE", f"${mae:,.0f}")
                else:
                    mae = mean_absolute_error(y, y_pred)
                    st.metric("MAE", f"${mae:,.0f}")

            # Sumário completo
            st.markdown("**📋 Sumário Completo do Modelo:**")
            st.text(str(model.summary()))

            # Significância das variáveis
            st.markdown("**🎯 Significância das Variáveis:**")
            coef_df = pd.DataFrame({
                'Variável': model.params.index,
                'Coeficiente': model.params.values,
                'Std Error': model.bse.values,
                't-statistic': model.tvalues.values,
                'p-valor': model.pvalues.values,
                'Significativo': model.pvalues.values < alpha_global
            })
            
            # Colorir significativas
            def highlight_significant(val):
                return 'background-color: lightgreen' if val else 'background-color: lightcoral'
            
            styled_df = coef_df.style.applymap(highlight_significant, subset=['Significativo'])
            st.dataframe(styled_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Erro no ajuste do modelo: {e}")
            st.stop()

    with tab3:
        st.subheader("🔍 Diagnósticos dos Pressupostos")
        
        residuals = model.resid
        fitted_values = model.fittedvalues
        
        # 1. Linearidade e Homocedasticidade
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**1. Resíduos vs Valores Ajustados**")
            fig_res, ax_res = plt.subplots(figsize=(8, 6))
            ax_res.scatter(fitted_values, residuals, alpha=0.6)
            ax_res.axhline(0, color='red', linestyle='--')
            ax_res.set_xlabel('Valores Ajustados')
            ax_res.set_ylabel('Resíduos')
            ax_res.set_title('Resíduos vs Ajustados')
            st.pyplot(fig_res)
            plt.close(fig_res)
            st.caption("Ideal: pontos aleatórios em torno de zero")

        with col2:
            st.markdown("**2. Q-Q Plot (Normalidade)**")
            fig_qq, ax_qq = plt.subplots(figsize=(8, 6))
            sm.qqplot(residuals, line='s', ax=ax_qq, fit=True)
            ax_qq.set_title('Q-Q Plot dos Resíduos')
            st.pyplot(fig_qq)
            plt.close(fig_qq)
            st.caption("Ideal: pontos próximos à linha diagonal")

        # 2. Testes estatísticos
        st.markdown("**🧪 Testes dos Pressupostos:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Normalidade (Shapiro-Wilk ou Kolmogorov-Smirnov)
            st.markdown("**Normalidade dos Resíduos:**")
            if len(residuals) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                st.write(f"Shapiro-Wilk: {shapiro_p:.4g}")
                if shapiro_p > alpha_global:
                    st.success("✅ Normal")
                else:
                    st.warning("⚠️ Não Normal")
            else:
                ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
                st.write(f"K-S Test: {ks_p:.4g}")
                if ks_p > alpha_global:
                    st.success("✅ Normal")
                else:
                    st.warning("⚠️ Não Normal")

        with col2:
            # Homocedasticidade (Breusch-Pagan)
            st.markdown("**Homocedasticidade:**")
            try:
                X_bp = X.select_dtypes(include=np.number)
                bp_test = het_breuschpagan(residuals, X_bp)
                bp_p = bp_test[1]
                st.write(f"Breusch-Pagan: {bp_p:.4g}")
                if bp_p > alpha_global:
                    st.success("✅ Homocedástico")
                else:
                    st.warning("⚠️ Heterocedástico")
            except:
                st.info("Teste BP não disponível")

        with col3:
            # Multicolinearidade (VIF)
            st.markdown("**Multicolinearidade (VIF):**")
            try:
                X_vif = X.drop(columns=['const']) if 'const' in X.columns else X
                X_vif = X_vif.select_dtypes(include=np.number)
                
                if X_vif.shape[1] > 1:
                    vif_values = []
                    for i in range(X_vif.shape[1]):
                        vif = variance_inflation_factor(X_vif.values, i)
                        vif_values.append(vif)
                    
                    max_vif = max(vif_values)
                    st.write(f"VIF Máximo: {max_vif:.2f}")
                    if max_vif < 10:
                        st.success("✅ VIF < 10")
                    elif max_vif < 20:
                        st.warning("⚠️ VIF Moderado")
                    else:
                        st.error("❌ VIF Alto")
                        
                    # Mostrar VIF detalhado
                    vif_df = pd.DataFrame({
                        'Variável': X_vif.columns,
                        'VIF': vif_values
                    })
                    st.dataframe(vif_df, use_container_width=True)
                else:
                    st.info("VIF requer múltiplas variáveis")
            except Exception as e:
                st.error(f"Erro VIF: {e}")

    with tab4:
        st.subheader("💡 Interpretação e Recomendações")
        
        # Interpretação dos coeficientes
        st.markdown("**📊 Interpretação dos Coeficientes:**")
        
        interpretation_data = []
        for var in model.params.index:
            coef = model.params[var]
            p_val = model.pvalues[var]
            is_sig = p_val < alpha_global
            
            if var == 'const':
                if use_log_transform:
                    interp = f"Intercepto: exp({coef:.4f}) = {np.exp(coef):,.0f} (preço base)"
                else:
                    interp = f"Intercepto: ${coef:,.0f} (preço base)"
            else:
                if use_log_transform and var in selected_numerical:
                    # Interpretação elasticidade
                    elasticity = coef * 100
                    interp = f"Elasticidade: +1% em {var} → {elasticity:+.2f}% no preço"
                elif var.startswith(tuple(selected_categorical)):
                    # Variável dummy
                    if use_log_transform:
                        pct_change = (np.exp(coef) - 1) * 100
                        interp = f"Dummy {var}: {pct_change:+.1f}% vs categoria base"
                    else:
                        interp = f"Dummy {var}: ${coef:+,.0f} vs categoria base"
                else:
                    # Variável numérica sem log
                    interp = f"+1 unidade em {var} → ${coef:+,.0f} no preço"
            
            interpretation_data.append({
                'Variável': var,
                'Coeficiente': f"{coef:.4f}",
                'p-valor': f"{p_val:.4g}",
                'Significativo': "✅" if is_sig else "❌",
                'Interpretação': interp
            })
        
        interp_df = pd.DataFrame(interpretation_data)
        st.dataframe(interp_df, use_container_width=True)
        
        # Recomendações práticas
        st.markdown("**🎯 Recomendações Práticas para Precificação:**")
        
        significant_vars = model.params[model.pvalues < alpha_global]
        significant_vars = significant_vars.drop('const', errors='ignore')
        
        if len(significant_vars) > 0:
            st.markdown("**Variáveis com maior impacto (significativas):**")
            
            # Ordenar por magnitude do impacto
            if use_log_transform:
                # Para log-log, usar elasticidade
                impacts = [(var, abs(coef)) for var, coef in significant_vars.items() if var in selected_numerical]
                impacts.extend([(var, abs(np.exp(coef) - 1)) for var, coef in significant_vars.items() if var not in selected_numerical])
            else:
                impacts = [(var, abs(coef)) for var, coef in significant_vars.items()]
            
            impacts.sort(key=lambda x: x[1], reverse=True)
            
            for i, (var, impact) in enumerate(impacts[:5], 1):
                coef = significant_vars[var]
                if var in selected_numerical and use_log_transform:
                    st.write(f"{i}. **{var}**: Elasticidade de {coef*100:.1f}% - Alta prioridade para valorização")
                elif var.startswith(tuple(selected_categorical)):
                    if use_log_transform:
                        pct_effect = (np.exp(coef) - 1) * 100
                        st.write(f"{i}. **{var}**: {pct_effect:+.1f}% no preço - Característica importante")
                    else:
                        st.write(f"{i}. **{var}**: ${coef:+,.0f} no preço - Característica importante")
                else:
                    st.write(f"{i}. **{var}**: ${coef:+,.0f} por unidade - Impacto direto no valor")
        
        # Qualidade do modelo
        st.markdown("**📈 Qualidade do Modelo:**")
        
        col1, col2 = st.columns(2)
        with col1:
            r2_interpretation = ""
            if model.rsquared >= 0.8:
                r2_interpretation = "🌟 Excelente poder explicativo"
            elif model.rsquared >= 0.6:
                r2_interpretation = "✅ Bom poder explicativo"
            elif model.rsquared >= 0.4:
                r2_interpretation = "⚠️ Poder explicativo moderado"
            else:
                r2_interpretation = "❌ Baixo poder explicativo"
            
            st.write(f"**R² = {model.rsquared:.3f}** - {r2_interpretation}")
            st.write(f"O modelo explica {model.rsquared*100:.1f}% da variação nos preços")
        
        with col2:
            if use_log_transform:
                st.write(f"**RMSE = ${rmse:,.0f}**")
                st.write(f"**MAE = ${mae:,.0f}**")
                st.write("Erros em escala original (dólares)")

# Rodapé
st.sidebar.markdown("---")
st.sidebar.info("""
**📚 Sobre este Dashboard:**
- Análise completa do Ames Housing Dataset
- ANOVA para comparação de grupos  
- Regressão Linear para predição de preços
- Implementado para Tarefa 2 - UnB/EPR
""")

# Informações técnicas no final
if not df_ames.empty:
    with st.expander("ℹ️ Informações Técnicas"):
        st.markdown(f"""
        **Dataset:** Ames Housing Dataset  
        **Observações:** {df_ames.shape[0]:,}  
        **Variáveis:** {df_ames.shape[1]}  
        **Preço médio:** {'${:,.0f}'.format(df_ames['SalePrice'].mean()) if 'SalePrice' in df_ames.columns else 'N/A'}  
        **Nível de significância configurado:** {alpha_global}  
        
        **Métodos implementados:**
        - ANOVA F-test com verificação de pressupostos
        - Testes não-paramétricos (Kruskal-Wallis)
        - Regressão Linear Múltipla (OLS)
        - Transformações log-log para elasticidades
        - Diagnósticos de resíduos completos
        - Correção robusta para heterocedasticidade
        """)
        
        st.markdown("**👨‍🏫 Professor:** João Gabriel de Moraes Souza")
        st.markdown("**🏫 Instituição:** UnB - Engenharia de Produção")
