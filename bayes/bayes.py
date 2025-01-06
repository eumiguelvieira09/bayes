import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Configuração do Streamlit
st.set_page_config(
    page_title="Bayes",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={'Get Help': 'https://www.google.com'}
)

# Caminho do arquivo para persistência
ARQUIVO_DADOS = "dados_bayes.csv"

# Dados iniciais (estado zero)
DADOS_INICIAIS = [
    ("fim da guerra comercial e crise", "alta"),
    ("mercados em alta depois dos resultados", "alta"),
    ("pressao do exercito derruba evo morales", "alta"),
    ("medo de guerra comercial derruba mercados", "baixa"),
    ("em alta do petroleo e crise", "baixa"),
    ("em crise guerra comercial derruba mercado", "baixa"),
]

# Função para carregar dados do arquivo
def carregar_dados():
    if os.path.exists(ARQUIVO_DADOS):
        return pd.read_csv(ARQUIVO_DADOS).to_records(index=False).tolist()
    else:
        return DADOS_INICIAIS

# Função para salvar dados no arquivo
def salvar_dados(dados):
    df = pd.DataFrame(dados, columns=["Frase", "Classe"])
    df.to_csv(ARQUIVO_DADOS, index=False)

# Estado inicial: carregar dados
if "dados" not in st.session_state:
    st.session_state.dados = carregar_dados()

# Função para resetar os dados para o estado inicial
def resetar_para_inicial():
    st.session_state.dados = DADOS_INICIAIS.copy()
    salvar_dados(st.session_state.dados)

# Treinar modelo
def treinar_modelo(dados):
    frases, classes = zip(*dados)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(frases)
    modelo = MultinomialNB()
    modelo.fit(X, classes)
    return modelo, vectorizer

modelo, vectorizer = treinar_modelo(st.session_state.dados)

# Layout com três colunas
col1, col2, col3 = st.columns(3)

# Coluna 1: Adicionar frases e resetar dados
with col3:
    st.subheader("Adicionar Frase aos Dados")
    nova_frase = st.text_input("Digite uma nova frase para adicionar:")
    nova_classe = st.selectbox("Selecione a classe para a nova frase:", ["alta", "baixa"])
    if st.button("Adicionar Frase"):
        if nova_frase:
            st.session_state.dados.append((nova_frase, nova_classe))
            salvar_dados(st.session_state.dados)
            st.success("Frase adicionada com sucesso!")
    
    if st.button("Resetar para o Estado Inicial"):
        resetar_para_inicial()
        st.success("Dados resetados para o estado inicial.")

# Coluna 2: Previsão
with col1:
    st.subheader("Prever Mercado")
    entrada_usuario = st.text_input("Digite uma frase para prever:")

    if entrada_usuario:
        entrada_transformada = vectorizer.transform([entrada_usuario])
        predicao = modelo.predict(entrada_transformada)[0]
        probabilidades = modelo.predict_proba(entrada_transformada)[0]
        classes = modelo.classes_

        st.write(f"A previsão é que o mercado estará: **{predicao.upper()}**")
        st.write("Probabilidades para cada classe:")
        for classe, probabilidade in zip(classes, probabilidades):
            st.write(f"- **{classe.capitalize()}**: {probabilidade * 100:.2f}%")

# Coluna 3: Base de Dados e visualização
with col2:
    st.subheader("Base de Dados Utilizada")
    df_dados = pd.DataFrame(st.session_state.dados, columns=["Frase", "Classe"])
    st.table(df_dados)

    st.subheader("Visualização da Distribuição de Classes")
    contagem_classes = df_dados["Classe"].value_counts()
    st.bar_chart(contagem_classes)
