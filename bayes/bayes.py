import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


st.set_page_config(
    page_title="Bayes",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://www.google.com'
    }
)

# Dados iniciais
dados_iniciais = [
    ("fim da guerra comercial e crise", "alta"),
    ("mercados em alta depois dos resultados", "alta"),
    ("pressao do exercito derruba evo morales", "alta"),
    ("medo de guerra comercial derruba mercados", "baixa"),
    ("em alta do petroleo e crise", "baixa"),
    ("em crise guerra comercial derruba mercado", "baixa"),
]

# Estado persistente
if "dados" not in st.session_state:
    st.session_state.dados = dados_iniciais.copy()

# Função para resetar os dados
def resetar_dados():
    st.session_state.dados = dados_iniciais.copy()

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
            st.success("Frase adicionada com sucesso!")
    
    if st.button("Resetar Dados"):
        resetar_dados()
        st.success("Dados resetados para o estado inicial.")

# Coluna 2: Previsão
with col1:
    st.subheader("Prever Mercado")
    entrada_usuario = st.text_input("Digite uma frase para prever:")

    if entrada_usuario:
        # Transformar a frase de entrada
        entrada_transformada = vectorizer.transform([entrada_usuario])

        # Predição e probabilidades
        predicao = modelo.predict(entrada_transformada)[0]
        probabilidades = modelo.predict_proba(entrada_transformada)[0]
        classes = modelo.classes_

        # Resultados como texto
        st.write(f"A previsão é que o mercado estará: **{predicao.upper()}**")
        st.write("Probabilidades para cada classe:")
        for classe, probabilidade in zip(classes, probabilidades):
            st.write(f"- **{classe.capitalize()}**: {probabilidade * 100:.2f}%")

        # Detalhes dos cálculos por palavra
        st.subheader("Cálculos Detalhados")
        palavras = entrada_usuario.split()
        palavras_no_vocabulario = [p for p in palavras if p in vectorizer.vocabulary_]
        detalhes = []

        if palavras_no_vocabulario:
            for classe in classes:
                prob_classe = modelo.class_count_[classes.tolist().index(classe)] / sum(modelo.class_count_)
                prob_palavras_dado_classe = 1.0
                palavra_detalhes = []

                for palavra in palavras_no_vocabulario:
                    indice_palavra = vectorizer.vocabulary_[palavra]
                    contador_palavra = modelo.feature_count_[classes.tolist().index(classe), indice_palavra]
                    prob_palavra = (contador_palavra + 1) / (
                            modelo.feature_count_[classes.tolist().index(classe)].sum() + len(vectorizer.vocabulary_)
                    )
                    prob_palavras_dado_classe *= prob_palavra
                    palavra_detalhes.append((palavra, prob_palavra))

                detalhes.append((classe, prob_classe, prob_palavras_dado_classe))

                # Exibir tabela de probabilidades por palavra para a classe atual
                palavra_df = pd.DataFrame(palavra_detalhes, columns=["Palavra", "Probabilidade"])
                st.write(f"**Classe '{classe.capitalize()}':**")
                st.table(palavra_df)

            # Probabilidade total para cada classe
            st.write("Probabilidade Total por Classe:")
            for classe, prob_classe, total_prob in detalhes:
                st.write(f"- **{classe.capitalize()}**: \( P({classe}|texto) = {total_prob:.4f} \)")
        else:
            st.write("Nenhuma palavra encontrada no vocabulário do modelo.")

# Coluna 3: Base de Dados e visualização
with col2:
    st.subheader("Base de Dados Utilizada")
    df_dados = pd.DataFrame(st.session_state.dados, columns=["Frase", "Classe"])
    st.table(df_dados)

    st.subheader("Visualização da Distribuição de Classes")
    contagem_classes = df_dados["Classe"].value_counts()
    st.bar_chart(contagem_classes)
