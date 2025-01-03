import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dados de treinamento
dados = [
    ("fim da guerra comercial e crise", "alta"),
    ("mercados em alta depois dos resultados", "alta"),
    ("pressao do exercito derruba evo morales", "alta"),
    ("medo de guerra comercial derruba mercados", "baixa"),
    ("em alta do petroleo e crise", "baixa"),
    ("em crise guerra comercial derruba mercado", "baixa"),
]

# Separar frases e classes
frases, classes = zip(*dados)

# Criar pipeline de classificação
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(frases)
modelo = MultinomialNB()
modelo.fit(X, classes)

# Streamlit com múltiplas abas
tab1, tab2 = st.tabs(["Previsão de Mercado", "Base de Dados"])

# Aba 1: Previsão de Mercado
with tab1:
    st.title("Previsão de Mercado: Alta ou Baixa")

    entrada_usuario = st.text_input("Digite uma frase sobre o mercado:")

    if entrada_usuario:
        # Transformar a frase de entrada
        entrada_transformada = vectorizer.transform([entrada_usuario])

        # Predição e probabilidades
        predicao = modelo.predict(entrada_transformada)[0]
        probabilidades = modelo.predict_proba(entrada_transformada)[0]
        classes = modelo.classes_

        # Resultados
        st.write(f"A previsão é que o mercado estará: **{predicao.upper()}**")
        st.write("Probabilidades:")
        for classe, probabilidade in zip(classes, probabilidades):
            st.write(f"- **{classe.capitalize()}**: {probabilidade * 100:.2f}%")

        # Explicação detalhada dos cálculos
        st.subheader("Explicação dos Cálculos")

        # Obter os contadores de palavras
        palavras = entrada_usuario.split()
        palavras_no_vocabulario = [p for p in palavras if p in vectorizer.vocabulary_]

        if palavras_no_vocabulario:
            for classe in classes:
                prob_classe = modelo.class_count_[classes.tolist().index(classe)] / sum(modelo.class_count_)
                st.write(f"**Para a classe '{classe.capitalize()}':**")
                st.write(f"- Probabilidade da classe: \( P({classe}) = {prob_classe:.2f} \)")

                prob_palavras_dado_classe = 1.0
                for palavra in palavras_no_vocabulario:
                    indice_palavra = vectorizer.vocabulary_[palavra]
                    contador_palavra = modelo.feature_count_[classes.tolist().index(classe), indice_palavra]
                    prob_palavra = (contador_palavra + 1) / (modelo.feature_count_[classes.tolist().index(classe)].sum() + len(vectorizer.vocabulary_))
                    prob_palavras_dado_classe *= prob_palavra
                    st.write(f"  - Probabilidade da palavra '{palavra}' dado '{classe}': \( P({palavra}|{classe}) = {prob_palavra:.4f} \)")

                total_prob = prob_palavras_dado_classe * prob_classe
                st.write(f"- Probabilidade total para esta classe: \( P({classe}|texto) = {total_prob:.4f} \)")
                if total_prob == 0:
                    st.write("*Aviso: Probabilidade total é zero devido à falta de palavras no vocabulário ou valores muito pequenos.*")
                st.write("---")
        else:
            st.write("Nenhuma palavra da frase foi encontrada no vocabulário do modelo.")

# Aba 2: Base de Dados
with tab2:
    st.title("Base de Dados Utilizada no Modelo")
    st.write("Abaixo estão os dados que foram usados para treinar o modelo:")
    st.table(dados)
