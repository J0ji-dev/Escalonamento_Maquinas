import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import heapq
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ============================
# CONFIGURAÇÃO INICIAL
# ============================

st.set_page_config(page_title="Branch and Bound FSSP", layout="wide")
st.title("🏭 Analisador FSSP com Branch and Bound")
st.markdown("Sistema interativo para análise e otimização de sequência de tarefas (Flow Shop Scheduling Problem).")

# ============================
# CARREGAR DATASET LOCAL
# ============================

try:
    df = pd.read_csv("problem_3m_10j.csv")

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    df.columns = [f"Maquina_{i+1}" for i in range(df.shape[1])]
    df.index = [f"Tarefa_{i+1}" for i in range(df.shape[0])]
    df.index.name = "Tarefas"

    st.success("✅ Arquivo `problem_3m_10j.csv` carregado automaticamente com sucesso!")

    st.subheader("📊 Visualização dos Dados")
    st.dataframe(df)

except FileNotFoundError:
    st.error("❌ O arquivo `problem_3m_10j.csv` não foi encontrado na pasta local. "
             "Coloque o arquivo na mesma pasta do `app.py` e reinicie o aplicativo.")
    st.stop()

# ============================
# GRÁFICOS EXPLORATÓRIOS
# ============================

df_long = df.reset_index().melt(id_vars="Tarefas", var_name="Maquinas", value_name="Tempo")

st.markdown("### 📈 Distribuição dos Tempos de Processamento")
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df_long["Tempo"], kde=True, bins=range(1, 20), ax=ax)
ax.set_title("Distribuição do Tempo de Processamento")
ax.set_xlabel("Tempo")
ax.set_ylabel("Frequência")
st.pyplot(fig)

st.markdown("### 🧭 Boxplot por Máquina")
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x="Maquinas", y="Tempo", data=df_long, ax=ax)
ax.set_title("Distribuição de Tempos por Máquina")
st.pyplot(fig)

st.markdown("### 🔥 Heatmap - Tempo por Máquina e Tarefa")
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(df, annot=True, cmap="YlOrRd", ax=ax)
ax.set_title("Mapa de Calor - Tempo de Processamento")
st.pyplot(fig)

# ============================
# ALGORITMO BRANCH AND BOUND
# ============================

st.subheader("⚙️ Execução do Algoritmo Branch and Bound")

@dataclass
class SearchMetrics:
    nodes_explored: int = 0
    nodes_pruned: int = 0
    max_depth: int = 0
    start_time: float = 0
    end_time: float = 0
    feasible_solutions: int = 0

class OptimizedBranchAndBound:
    def __init__(self, processing_times: List[float]):
        self.processing_times = processing_times
        self.num_jobs = len(processing_times)
        self.num_machines = 3
        self.best_solution = None
        self.best_makespan = float("inf")
        self.metrics = SearchMetrics()
        self.lb_cache = {}

    def calculate_lower_bound(self, machine_times: List[float], level: int) -> float:
        state_key = (tuple(machine_times), level)
        if state_key in self.lb_cache:
            return self.lb_cache[state_key]
        current_max = max(machine_times)
        unassigned_times = self.processing_times[level:]
        lb = max(current_max, sum(unassigned_times)/self.num_machines if unassigned_times else current_max)
        self.lb_cache[state_key] = lb
        return lb

    def solve(self) -> Tuple[List[int], float, Dict]:
        self.metrics.start_time = time.time()
        root_machine_times = [0.0, 0.0, 0.0]
        root_lb = self.calculate_lower_bound(root_machine_times, 0)
        pq = []
        heapq.heappush(pq, (root_lb, 0, root_machine_times, [], 0))
        while pq:
            current_lb, level, machine_times, assignment, depth = heapq.heappop(pq)
            self.metrics.nodes_explored += 1
            self.metrics.max_depth = max(self.metrics.max_depth, depth)
            if current_lb >= self.best_makespan:
                self.metrics.nodes_pruned += 1
                continue
            if level == self.num_jobs:
                self.metrics.feasible_solutions += 1
                makespan = max(machine_times)
                if makespan < self.best_makespan:
                    self.best_makespan = makespan
                    self.best_solution = assignment.copy()
                continue
            next_job = self.processing_times[level]
            for machine in range(self.num_machines):
                new_times = machine_times.copy()
                new_times[machine] += next_job
                new_lb = self.calculate_lower_bound(new_times, level + 1)
                if new_lb < self.best_makespan:
                    heapq.heappush(pq, (new_lb, level + 1, new_times, assignment + [machine], depth + 1))
        self.metrics.end_time = time.time()
        return self.best_solution, self.best_makespan, self._get_metrics_dict()

    def _get_metrics_dict(self):
        return {
            "nodes_explored": self.metrics.nodes_explored,
            "nodes_pruned": self.metrics.nodes_pruned,
            "max_depth": self.metrics.max_depth,
            "execution_time": self.metrics.end_time - self.metrics.start_time,
            "feasible_solutions": self.metrics.feasible_solutions,
            "best_makespan": self.best_makespan,
            "pruning_ratio": self.metrics.nodes_pruned / max(1, self.metrics.nodes_explored),
        }

# ============================
# INTERFACE DE EXECUÇÃO
# ============================

processing_times = [float(x) for x in df.mean().values]  # média dos tempos por máquina
st.markdown(f"**Tempos de processamento base (médias por máquina):** `{processing_times}`")

if st.button("🚀 Executar Branch and Bound"):
    scheduler = OptimizedBranchAndBound(processing_times)
    solution, makespan, metrics = scheduler.solve()

    st.success(f"✅ Melhor Makespan: **{metrics['best_makespan']:.2f}**")
    st.write("**Máquina atribuída para cada tarefa:**", solution)

    # Mostrar métricas
    st.markdown("### 📋 Métricas da Execução")
    st.json(metrics)

    # Gráfico de balanceamento
    st.markdown("### ⚖️ Balanceamento de Carga por Máquina")
    totals = [0, 0, 0]
    for job, machine in enumerate(solution):
        totals[machine] += processing_times[job % len(processing_times)]
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x=["M1", "M2", "M3"], y=totals, ax=ax)
    ax.set_title("Carga Total por Máquina")
    st.pyplot(fig)

    # ============================
    # SEÇÃO EXTRA — VALIDAÇÃO E COMPARAÇÃO
    # ============================
    st.markdown("---")
    st.header("⚖️ Evidências e Validação")
    st.markdown("""
    Nesta seção, comparamos o desempenho do **Branch and Bound** com uma **heurística simples** (gulosa),
    além de avaliar como o tempo de execução cresce com o número de tarefas.
    """)

    def greedy_heuristic(times):
        num_jobs = len(times)
        num_machines = 3
        machine_loads = [0] * num_machines
        for t in range(num_jobs):
            m = np.argmin(machine_loads)
            machine_loads[m] += times[t]
        return machine_loads, max(machine_loads)

    # Comparação direta
    greedy_loads, greedy_cost = greedy_heuristic(processing_times)

    st.subheader("📊 Comparativo de Resultados")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Branch and Bound - Makespan", f"{makespan:.2f}")
    with col2:
        st.metric("Heurística Gulosa - Makespan", f"{greedy_cost:.2f}")

    # Gráfico comparativo
    fig, ax = plt.subplots()
    ax.bar(["Branch and Bound", "Heurística"], [makespan, greedy_cost], color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Makespan")
    ax.set_title("Comparação entre Algoritmos")
    st.pyplot(fig)

        # --- SENSIBILIDADE ---
    st.subheader("📊 Análise de Sensibilidade")
    tamanhos = [5, 10, 15, 20, 25]
    tempos_bb, tempos_heur = [], []

    num_machines = 3  # define o número de máquinas fixo
    progress = st.progress(0)

    for i, n in enumerate(tamanhos):
        data = np.random.randint(1, 20, size=n)

        # --- Branch and Bound ---
        start = time.time()
        try:
            scheduler_test = OptimizedBranchAndBound(list(data))
            scheduler_test.solve()
            tempos_bb.append(time.time() - start)
        except Exception as e:
            st.warning(f"Erro ao rodar Branch and Bound com {n} tarefas: {e}")
            tempos_bb.append(np.nan)

        # --- Heurística ---
        start = time.time()
        try:
            greedy_heuristic(list(data))
            tempos_heur.append(time.time() - start)
        except Exception as e:
            st.warning(f"Erro ao rodar heurística com {n} tarefas: {e}")
            tempos_heur.append(np.nan)

        progress.progress((i + 1) / len(tamanhos))

    # --- GERA O GRÁFICO ---
    if len(tempos_bb) > 0 and len(tempos_heur) > 0:
        fig2, ax2 = plt.subplots()
        ax2.plot(tamanhos, tempos_bb, label="Branch and Bound", marker='o')
        ax2.plot(tamanhos, tempos_heur, label="Heurística", marker='s')
        ax2.set_xlabel("Número de Tarefas")
        ax2.set_ylabel("Tempo de Execução (s)")
        ax2.set_title("⏱️ Análise de Sensibilidade (Tempo x Nº de Tarefas)")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.warning("❌ Não foi possível gerar o gráfico — verifique se as funções retornam corretamente.")
