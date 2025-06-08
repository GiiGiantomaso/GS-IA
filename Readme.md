# Classificador de Nível de Urgência para Pedidos de Ajuda — Supernova ML

Este projeto aplica **Machine Learning** para prever o nível de urgência de pedidos de ajuda em situações de desastre, com base em variáveis estruturadas e descrições textuais do ocorrido. O objetivo é garantir que os atendimentos sejam realizados de forma ágil e organizada conforme a real necessidade de cada situação.

---

## Integrantes

- Giovanna Lima Giantomaso | RM553369  
- Rebeca Silva Lopes | RM553764

---
## LINK DO VÍDEO
...

---
## Motivação

Em cenários de emergência, como enchentes, incêndios, deslizamentos ou crises, é essencial que os pedidos de ajuda sejam classificados corretamente. Este classificador atua como um apoio automatizado para priorizar os casos com maior gravidade, possibilitando uma resposta mais rápida e eficiente das equipes de apoio.

---

## Tecnologias Utilizadas

- **Python 3.x**
- **Pandas** – manipulação de dados
- **Scikit-learn** – machine learning
- **SciPy** – suporte estatístico
- **Joblib** – serialização do modelo
- **Matplotlib e Seaborn** – gráficos e visualizações
- **Jupyter Notebook** – experimentação e testes

---

## Dataset

O arquivo `dataset_pedidoajuda.csv` contém dados simulados com os seguintes campos:

- `tipo_ajuda`: tipo de socorro necessário
- `criancas`: se há crianças no local
- `pessoas`: número de pessoas envolvidas
- `descricao`: texto descritivo da situação
- `urgencia`: rótulo com o nível real de urgência (baixa, média, alta)

---

## Modelo de IA

- **Técnica:** Gradient Boosting (classificação supervisionada)
- **Pré-processamento:**  
  - LabelEncoder nas colunas categóricas  
  - Vetorização textual com `TfidfVectorizer` na coluna de descrição
- **Treinamento:** 80% dos dados são usados para treino, 20% para teste
- **Avaliação:** Métricas de acurácia, precisão e f1-score

Esse modelo aprende padrões em dados reais e é capaz de prever a urgência de novos pedidos, mesmo com descrições complexas.

---

## Estrutura do Projeto

```
GS-IA/
├── classificacao_urgencia_pedidoajuda.ipynb
├── dataset_pedidoajuda.csv
├── predictor.py  
├── requirements.txt      
└── Readme.md                 
```

---

## Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/GiiGiantomaso/GS-IA.git

```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Execute o arquivo .ipybn

Abra o arquivo `.ipynb` com o Jupyter:

```bash
jupyter notebook
```

### 4. Rodar o arquivo .py e inserir informações via terminal:

```bash
python predictor.py
```

---

## Exemplo de Funcionamento

### exemplo de urgência BAIXA
Tipo de ajuda (ex: Água, Abrigo, Resgate): Água

Há crianças no local? (S/N): N

Número de pessoas no local: 3

Descreva a situação: Necessitam de água

 Urgência prevista: BAIXA


### exemplo de urgência ALTA

Tipo de ajuda (ex: Água, Abrigo, Resgate): Resgate

Há crianças no local? (S/N): S

Número de pessoas no local: 4

Descreva a situação: Pessoas ilhadas em casa alagada

Urgência prevista: ALTA

### exemplo urgência MÉDIA

Tipo de ajuda (ex: Água, Abrigo, Resgate): Água

Há crianças no local? (S/N): N

Número de pessoas no local: 4

Descreva a situação: Falta de água potável

Urgência prevista: MÉDIA


## Benefícios Esperados

- Priorização automatizada dos pedidos mais urgentes
- Melhoria na gestão de recursos em emergências
- Auxílio a plataformas de ajuda humanitária e governos
- Aplicabilidade imediata em sistemas IoT com sensores ou aplicativos móveis

---

## Observação

Este projeto é parte da solução **SOS-GR**, uma plataforma integrada de ajuda comunitária com foco em acessibilidade e resposta rápida a emergências.
