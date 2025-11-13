# Escalonamento de Máquinas

## Descrição

Este repositório propõe-se a resolver uma instância do problema de *Flowshop Scheduling* (FSS/FSSP): escalonar **10 tarefas** em **3 máquinas**, de forma a minimizar o **tempo total de conclusão (*makespan*)**. O algoritmo utilizado é uma implementação de *Branch and Bound* otimizado, capaz de explorar diferentes atribuições de tarefas às máquinas e descartar estados que não levam à solução ótima.

## Sumário

1. [Aquisição e preparo de dados](#aquisição-e-preparo-de-dados)
    - [Seleção do dataset](#seleção-do-dataset)
    - [Limpeza e padronização](#limpeza-e-padronização)
    - [Análise Exploratória de Dados (EDA)](#an%C3%A1lise-explorat%C3%B3ria-de-dados-eda)
2. [Implementação do Branch and Bound](#implementa%C3%A7%C3%A3o-do-branch-and-bound)
   - [Estrutura do algoritmo](#estrutura-do-algoritmo)
   - [Métricas de execução](#m%C3%A9tricas-de-execu%C3%A7%C3%A3o)
   - [Reprodutibilidade](#reprodutibilidade)

## Aquisição e preparo de dados

Aqui iremos nos aprofundar no dataset escolhido e os ajustes necessários aplicados

### Seleção do dataset

O dataset foi retirado de um repositório público do GitHub. Você pode acessá-lo <a href="https://github.com/akilelkamel/fssp-dataset/blob/main/probems/problem_3m_10j.csv" target="_blank">aqui</a>.

Entrando mais a fundo na estrutura, para a instância escolhida, com **10 tarefas** e **3 máquinas**, o dataset possui **originalmente** **11 linhas**, cada uma com **4 colunas**, onde cada célula representa o **tempo de processamento** da tarefa naquela máquina. O arquivo segue um padrão indexado: tanto as linhas quanto as colunas possuem prefixos — "J" para tarefas (*jobs*) e "M" para máquinas (*machines*).

---

### Limpeza e padronização

Tendo em vista que não há dados faltantes e nem duplicados, o primeiro ajuste foi a remoção de uma coluna redundante e não nomeada do conjunto de dados em análise.

Aqui está o trecho de código onde tal modificação é feita:
```py
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)
```

Em seguida, padronizamos os rótulos das colunas e linhas: as colunas receberam o prefixo `Maquina_` seguido de um índice (1 a 3), e as linhas receberam o prefixo `Tarefa_` seguido de seu índice correspondente.

Em código:
```py
num_rows, num_cols = df.shape
df.columns = [f"Máquina_{i+1}" for i in range(num_cols)]
df.index = [f"Tarefa_{i+1}" for i in range(num_rows)]
df.index.name = "Tarefas"
```

---

### Análise Exploratória de Dados (EDA)

> [!NOTE]
> Clique <a href= "https://github.com/J0ji-dev/Escalonamento_Maquinas/blob/main/Dados/Documenta%C3%A7%C3%A3o%20-%20Preparo%20de%20Dados.pdf" target="_blank">aqui</a> esteja interessado em maiores informações dessa etapa em particular.

## Implementação do Branch and Bound

Com os detalhes a respeito da base de dados escolhida esclarecidos, podemos seguir para maiores explicações sobre o algoritmo resolutivo utilizado. O código referente ao algoritmo está integralmente implementado em **2 classes** presentes no arquivo `branch_bound.py` na raiz do repositório.

### Estrutura do algoritmo

A implementação adjacente faz uso de uma fila de prioridade, esta implementada através da estrutura de dados *heap*, retirada do módulo `heapq` para os seguintes fins:

- Cálculo do limite inferior (*lower bound*)

Em código:
```py
# Método encarregado do cálculo
def lower_bound(self, machine_times: List[float]) -> float:
        return max(machine_times)
```

- Expansão dos nós e poda de ramos inviáveis

Em código:
```py
# Método responsável
   def solve(self) -> Tuple[List[int], float, Dict]:
        self.metrics.start_time = time.time()
        pq = [(0, 0, [0, 0, 0], [], 0)]  # (bound, level, times, seq, depth)

        while pq:
            pq.sort(key=lambda x: x[0])
            bound, level, times, seq, depth = pq.pop(0)
            self.metrics.nodes_explored += 1

            if bound >= self.best_makespan:
                self.metrics.nodes_pruned += 1
                continue

            if level == self.num_jobs:
                makespan = max(times)
                if makespan < self.best_makespan:
                    self.best_makespan = makespan
                    self.best_solution = seq
                continue

            next_job = self.processing_times[level]
            for m in range(3):
                new_times = times.copy()
                new_times[m] += next_job
                new_bound = self.lower_bound(new_times)
                pq.append((new_bound, level + 1, new_times, seq + [m], depth + 1))

        self.metrics.end_time = time.time()
        return self.best_solution, self.best_makespan, vars(self.metrics)
```

---

  ### Métricas de execução
  
Todas as informações relevantes para as métricas são armazenadas numa classe `SearchMetrics`. As duas propriedades *time* (`start_time` e `end_time`) são manipuladas para cálculo do tempo de execução efetivo do algoritmo

  Em código:
  ```py
@dataclass
class SearchMetrics:
    nodes_explored: int = 0     # Nós explorados
    nodes_pruned: int = 0       # Nós podados
    max_depth: int = 0          # Profundidade máxima
    start_time: float = 0       # Marcador começo
    end_time: float = 0         # Marcador encerramento
    feasible_solutions: int = 0 # Soluções viáveis
```

---

### Reprodutibilidade

Para garantia da reprodutibilidade da aplicação foram adotas as seguintes medidas:

- Arquivo `README.md` explicativo
- Arquivo `requirements.txt` para declaração de dependências
- Script `app.py` para centralização da execução
