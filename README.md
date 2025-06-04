# 🏠 Dashboard de Análise de Preços Imobiliários - Ames Housing

## 📋 Sobre o Projeto

Dashboard interativo desenvolvido para a **Tarefa 2** da disciplina de Engenharia de Produção da UnB, focado em análise estatística do dataset Ames Housing para precificação imobiliária.

**Professor:** João Gabriel de Moraes Souza  
**Instituição:** Universidade de Brasília - Departamento de Engenharia de Produção

## 🎯 Objetivos

- **Análise ANOVA:** Identificar diferenças significativas no preço médio de venda entre categorias de variáveis
- **Regressão Linear:** Construir modelo preditivo para precificação de imóveis
- **Dashboard Interativo:** Interface web para análise dinâmica dos dados

## 🚀 Funcionalidades

### 📊 Visão Geral dos Dados
- Estatísticas descritivas completas
- Visualizações de distribuição de preços
- Matriz de correlação interativa
- Análise exploratória detalhada

### 🔬 Análise ANOVA
- Teste F para comparação de grupos
- Verificação de pressupostos (Shapiro-Wilk, Levene)
- Testes não-paramétricos (Kruskal-Wallis)
- Q-Q plots para análise de normalidade
- Post-hoc Tukey HSD
- Interpretação para tomada de decisão

### 📈 Regressão Linear Múltipla
- Modelo OLS com 4-6 variáveis explicativas
- Transformação log-log para elasticidades
- Diagnósticos completos dos pressupostos
- Correção robusta para heterocedasticidade (HC0)
- Métricas de performance (R², RMSE, MAE)
- Interpretação prática dos coeficientes
- Recomendações de negócio

## 🛠️ Tecnologias Utilizadas

- **Python 3.11+**
- **Streamlit** - Interface web interativa
- **Pandas & NumPy** - Manipulação de dados
- **Statsmodels** - Análise estatística
- **SciPy** - Testes estatísticos
- **Plotly & Matplotlib** - Visualizações
- **Scikit-learn** - Métricas de ML

## 📦 Instalação e Execução

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/ames-housing-dashboard.git
cd ames-housing-dashboard
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Adicione o dataset
Baixe o arquivo `AmesHousing.csv` e coloque na raiz do projeto.

### 4. Execute o dashboard
```bash
streamlit run app.py
```

### 5. Acesse no navegador
```
http://localhost:8501
```

## 📁 Estrutura do Projeto

```
ames-housing-dashboard/
├── app.py                 # Dashboard principal
├── requirements.txt       # Dependências Python
├── README.md             # Documentação
├── AmesHousing.csv       # Dataset (não incluído no repo)
└── devcontainer.json     # Configuração para Codespaces
```

## 📊 Dataset

O projeto utiliza o **Ames Housing Dataset**, que contém informações detalhadas sobre vendas de imóveis em Ames, Iowa, incluindo:

- **2930 observações** de vendas residenciais
- **82 variáveis** explicativas
- Características físicas, localização, qualidade, etc.
- Preços de venda para modelagem

### Variáveis Principais
- **SalePrice:** Preço de venda (variável dependente)
- **GrLivArea:** Área habitável
- **OverallQual:** Qualidade geral da construção
- **Neighborhood:** Bairro
- **YearBuilt:** Ano de construção
- **CentralAir:** Ar condicionado central
- E muitas outras...

## 📈 Metodologia Estatística

### ANOVA (Análise de Variância)
1. **Seleção de 2-3 variáveis categóricas**
2. **Teste F** para diferenças entre grupos
3. **Verificação de pressupostos:**
   - Normalidade dos resíduos (Shapiro-Wilk)
   - Homocedasticidade (Levene)
4. **Testes robustos** quando pressupostos violados
5. **Post-hoc Tukey** para comparações múltiplas

### Regressão Linear Múltipla
1. **Seleção de 4-6 preditores** (contínuos + categóricos)
2. **Transformação log-log** para elasticidades
3. **Diagnósticos dos pressupostos:**
   - Linearidade
   - Normalidade dos resíduos
   - Homocedasticidade (Breusch-Pagan)
   - Multicolinearidade (VIF)
4. **Modelo robusto** para correções
5. **Interpretação prática** dos coeficientes

## 🎯 Resultados Esperados

### Para Corretores e Investidores
- Identificação de características que mais impactam preços
- Elasticidades preço-demanda por atributos
- Recomendações para valorização de imóveis
- Análise de bairros e qualidades construtivas

### Para Tomada de Decisão
- Insights estatisticamente fundamentados
- Modelos preditivos confiáveis
- Interpretação percentual de impactos
- Estratégias de precificação baseadas em dados

## 🏆 Conformidade com a Tarefa 2

✅ **Análise ANOVA com 2-3 variáveis categóricas**  
✅ **Regressão com 4-6 variáveis (1+ contínua, 1+ categórica)**  
✅ **Verificação completa de pressupostos**  
✅ **Transformação logarítmica opcional**  
✅ **Testes robustos quando necessário**  
✅ **Interpretação prática dos resultados**  
✅ **Dashboard interativo (+2 pontos bônus)**  

## 🎨 Interface do Dashboard

- **Design profissional** inspirado nos materiais do professor
- **Navegação por abas** organizadas
- **Métricas visuais** destacadas
- **Gráficos interativos** com Plotly
- **Configurações dinâmicas** na sidebar
- **Interpretações contextuais** automáticas

## 📝 Como Usar

1. **Visão Geral:** Explore o dataset e correlações
2. **ANOVA:** Selecione variáveis categóricas para análise
3. **Regressão:** Configure modelo com suas variáveis preferidas
4. **Interpretação:** Analise resultados e recomendações

## 🚀 Deploy

O dashboard pode ser facilmente deployado em:
- **Streamlit Share**
- **Hugging Face Spaces**
- **Heroku**
- **GitHub Codespaces**

## 👨‍💻 Autor

Desenvolvido para a disciplina de Engenharia de Produção da UnB, sob orientação do Professor João Gabriel de Moraes Souza.

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🤝 Contribuições

Sugestões e melhorias são bem-vindas! Abra uma issue ou envie um pull request.

---

**🏫 Universidade de Brasília - Faculdade de Tecnologia - Departamento de Engenharia de Produção**
