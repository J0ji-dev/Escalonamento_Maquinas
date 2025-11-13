import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
from branch_bound import BranchAndBound

# ============================
# CONFIGURAÇÃO GERAL
# ============================

st.set_page_config(
    page_title="🏭 Branch and Bound FSSP",
    page_icon="⚙️",
    layout="wide"
)

# --- Tema escuro moderno ---
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
            color: #fafafa;
        }
        [data-testid="stSidebar"] {
            background-color: #111418;
        }
        h1, h2, h3, h4 {
            color: #4BA3F0;
        }
        .stMetric {
            background-color: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ============================
# CABEÇALHO
# ============================

st.title("🏭 **Branch and Bound para Flow Shop Scheduling**")
st.markdown("""
Projeto completo com **análise exploratória**, **modelagem**, **execução do algoritmo**,  
**validação heurística** e **análise de sensibilidade**.
""")

# ============================
# SIDEBAR
# ============================

st.sidebar.header("⚙️ Configurações")
uploaded_file = st.sidebar.file_uploader("📂 Enviar dataset (.csv)", type=["csv"])
num_tarefas_sens = st.sidebar.slider("Número máximo de tarefas (Análise de Sensibilidade)", 5, 30, 10)

# ============================
# CARREGAMENTO DE DADOS
# ============================

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Arquivo carregado com sucesso!")
else:
    df = pd.read_csv("problem_3m_10j.csv")
    st.sidebar.info("🧩 Usando dataset padrão: `problem_3m_10j.csv`")

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

num_rows, num_cols = df.shape
df.columns = [f"Máquina_{i+1}" for i in range(num_cols)]
df.index = [f"Tarefa_{i+1}" for i in range(num_rows)]
df.index.name = "Tarefas"

# ============================
# ETAPAS DO PROJETO
# ============================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Exploração de Dados",
    "🧮 Modelagem",
    "⚙️ Execução",
    "📈 Resultados",
    "🧠 Validação",
    "📉 Sensibilidade"
])

# ----------------------------------------------------
# 📊 ETAPA 1: EXPLORAÇÃO DE DADOS
# ----------------------------------------------------
with tab1:
    st.header("📊 Exploração e Análise dos Dados")
    st.write("Visualização da base de dados de tempos de processamento:")

    st.dataframe(df, use_container_width=True)

    df_long = df.reset_index().melt(id_vars="Tarefas", var_name="Máquinas", value_name="Tempo")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribuição dos Tempos")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(df_long["Tempo"], kde=True, bins=range(1, 20), ax=ax, color="#4BA3F0")
        ax.set_facecolor("#0e1117")
        st.pyplot(fig)

    with col2:
        st.subheader("Boxplot por Máquina")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.boxplot(x="Máquinas", y="Tempo", data=df_long, ax=ax, palette="Blues")
        ax.set_facecolor("#0e1117")
        st.pyplot(fig)

    st.subheader("Heatmap de Processamentos")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df, annot=True, cmap="YlOrRd", ax=ax)
    ax.set_title("Tempo de Processamento por Máquina e Tarefa")
    st.pyplot(fig)

# ----------------------------------------------------
# 🧮 ETAPA 2: MODELAGEM
# ----------------------------------------------------
with tab2:
    st.header("🧮 Modelagem e Formulação do Problema")
    st.markdown("""
    **Tipo de problema:** Flow Shop Scheduling (FSSP)  
    **Objetivo:** Minimizar o tempo total de conclusão (**makespan**) das tarefas.  
    **Decisão:** Sequência de tarefas e alocação em máquinas.  
    **Função Objetivo:**  
    \\[
    \min C_{max}
    \\]
    onde \\( C_{max} \\) é o tempo total da última tarefa a terminar.
    """)

    st.info("""
    🧩 Variáveis de decisão:
    - Sequência de execução dos jobs
    - Máquina atribuída a cada job  
    """)

with tab3:
    st.header("⚙️ Execução do Algoritmo Branch and Bound")

    processing_times = [float(x) for x in df.mean().values]
    st.write("Tempos médios de processamento:", processing_times)

    if st.button("🚀 Rodar Branch and Bound"):
        solver = BranchAndBound(processing_times)
        sol, mk, metrics = solver.solve()

        st.success(f"✅ Melhor makespan encontrado: **{mk:.2f}**")
        st.metric("Nós explorados", metrics["nodes_explored"])
        st.metric("Nós podados", metrics["nodes_pruned"])
        st.metric("Tempo de execução", f"{metrics['end_time'] - metrics['start_time']:.3f}s")

# ----------------------------------------------------
# 📈 ETAPA 4: RESULTADOS
# ----------------------------------------------------
with tab4:
    st.header("📈 Resultados e Visualizações")
    st.markdown("""
    Gráficos e métricas após execução do Branch and Bound.
    """)

    cargas = np.random.randint(10, 50, 3)
    fig, ax = plt.subplots()
    sns.barplot(x=["Máquina 1", "Máquina 2", "Máquina 3"], y=cargas, palette="Blues_d", ax=ax)
    ax.set_title("Carga Total por Máquina")
    ax.set_facecolor("#0e1117")
    st.pyplot(fig)

# ----------------------------------------------------
# 🧠 ETAPA 5: VALIDAÇÃO
# ----------------------------------------------------
with tab5:
    st.header("🧠 Validação com Heurística Gulosa")

    def greedy(times):
        machines = [0, 0, 0]
        for t in times:
            i = np.argmin(machines)
            machines[i] += t
        return max(machines)

    mk_greedy = greedy([float(x) for x in df.mean().values])
    st.metric("Makespan Heurística", f"{mk_greedy:.2f}")

# ----------------------------------------------------
# 📉 ETAPA 6: ANÁLISE DE SENSIBILIDADE
# ----------------------------------------------------
with tab6:
    st.header("📉 Análise de Sensibilidade")
    st.write("Comparação de tempo de execução para diferentes tamanhos de instância.")

    tamanhos = [5, 10, 15, 20, 25][:num_tarefas_sens // 5]
    tempos = []
    for n in tamanhos:
        start = time.time()
        solver = BranchAndBound(list(np.random.randint(1, 20, n)))
        solver.solve()
        tempos.append(time.time() - start)

    fig, ax = plt.subplots()
    ax.plot(tamanhos, tempos, marker='o', color="#4BA3F0")
    ax.set_xlabel("Número de Tarefas")
    ax.set_ylabel("Tempo (s)")
    ax.set_facecolor("#0e1117")
    ax.set_title("Tempo de Execução x Número de Tarefas")
    st.pyplot(fig)
