# CLASSIFICADOR DE URGÊNCIA

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import joblib

# carregando o dataset
df = pd.read_csv("dataset_pedidoajuda.csv")

# codificação
encoder_tipo = LabelEncoder()
encoder_crianca = LabelEncoder()
encoder_urgencia = LabelEncoder()

# treinamento dos encoders
X_tipo = encoder_tipo.fit_transform(df["TipoAjuda"])
X_crianca = encoder_crianca.fit_transform(df["CriancasNoLocal"])
X_pessoas = df["PessoasNoLocal"].values
X_texto = TfidfVectorizer().fit_transform(df["SituacaoDeRisco"])
y = encoder_urgencia.fit_transform(df["NivelUrgencia"])

# vetorizar texto completo para reuso
vetorizador = TfidfVectorizer()
X_texto = vetorizador.fit_transform(df["SituacaoDeRisco"])

# unificar X
X_numerico = pd.DataFrame({"TipoAjuda": X_tipo, "CriancasNoLocal": X_crianca, "PessoasNoLocal": X_pessoas}).values
X_final = hstack([X_numerico, X_texto])

# divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)

# modelo
modelo = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
modelo.fit(X_train, y_train)

# avaliação
y_pred = modelo.predict(X_test)
print("\nRelatório de Classificação:\n")
print(classification_report(y_test, y_pred, target_names=encoder_urgencia.classes_))

# função com input()
def prever_urgencia(tipo_ajuda, criancas_no_local, pessoas_no_local, situacao_texto):
    tipo_cod = encoder_tipo.transform([tipo_ajuda])[0]
    crianca_cod = encoder_crianca.transform([criancas_no_local])[0]
    texto_vetor = vetorizador.transform([situacao_texto])
    dados = pd.DataFrame([{"TipoAjuda": tipo_cod, "CriancasNoLocal": crianca_cod, "PessoasNoLocal": pessoas_no_local}])
    entrada_final = hstack([dados.values, texto_vetor])
    pred = modelo.predict(entrada_final)
    urgencia = encoder_urgencia.inverse_transform(pred)[0]
    print(f"\n Urgência prevista: {urgencia}")

# Execucao
if __name__ == "__main__":
    print("--- Previsor de Nível de Urgência ---")
    tipo = input("Tipo de ajuda (ex: Água, Abrigo, Resgate): ")
    crianca = input("Há crianças no local? (S/N): ")
    pessoas = int(input("Número de pessoas no local: "))
    situacao = input("Descreva a situação: ")

    prever_urgencia(tipo, crianca, pessoas, situacao)