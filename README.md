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
3. [Evidências e validação](#evid%C3%AAncias-e-valida%C3%A7%C3%A3o)
   - [Comparação de desempenho](#compara%C3%A7%C3%A3o-de-desempenho)
   - [Testes unitários](#testes-unit%C3%A1rios)
5. [Como executar a aplicação](#como-executar-a-aplica%C3%A7%C3%A3o)
    1. [Pré-requisitos](#pr%C3%A9-requisitos)
    2. [Clonando o repositório](#clonando-o-reposit%C3%B3rio)
    3. [Instalando dependências](#instalando-depend%C3%AAncias)
    4. [Executando](#executando)

## Aquisição e preparo de dados

Aqui iremos nos aprofundar no dataset escolhido e os ajustes necessários aplicados.

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
> Clique <a href= "https://github.com/J0ji-dev/Escalonamento_Maquinas/blob/main/Dados/Documenta%C3%A7%C3%A3o%20-%20Preparo%20de%20Dados.pdf" target="_blank">aqui</a> caso esteja interessado em maiores informações dessa etapa em particular.

## Implementação do Branch and Bound

Com os detalhes a respeito da base de dados escolhida esclarecidos, podemos seguir para maiores explicações sobre o algoritmo resolutivo utilizado. O código referente ao algoritmo está integralmente implementado em **2 classes** presentes no arquivo `branch_and_bound.py` na raiz do repositório.

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
  
Todas as informações relevantes para as métricas são armazenadas numa classe `SearchMetrics`. As duas propriedades *time* (`start_time` e `end_time`) são manipuladas para cálculo do tempo de execução efetivo do algoritmo.

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

## Evidências e validação

Aqui poderá testemunhar como validamos o sistema fazendo uso dos testes unitários e as evidências que eles nos deram.

### Comparação de desempenho

A nossa solução ótima com o algoritmo *Branch and Bound* é colocada à prova contra uma solução de heurística gulosa (*greedy*) e disponibiliza o resultado (*makespan*).

Em código:
```py
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
```

---

### Testes unitários

Todos os 6 testes unitários estão concentrados em um único arquivo na raiz do repositório (`test_branch_and_bound.py`) e são referentes ao *Branch and Bound*. O objetivo e motivação dos testes é verificar critérios-chave do algoritmo, assegurando assim robustez e confiabilidade. Conheça um pouco melhor as funções presentes no arquivo e o que testam em maiores detalhes:

- `test_initialization`: Assegura a **inicialização apropriada** do objeto de teste, importantíssima para as funções adiante pois partem desse ponto.
  
Em código:
```py
def test_initialization(simple_instance):
    """Verifica se a classe inicializa corretamente"""
    bnb = simple_instance
    assert bnb.num_jobs == 3
    assert bnb.num_machines == 3
    assert isinstance(bnb.metrics, type(bnb.metrics))  # tipo interno
    assert bnb.best_makespan == float('inf')
    assert bnb.best_solution is None
    assert isinstance(bnb.lb_cache, dict)
```

- `test_calculate_lower_bound_and_cache`: Verifica a precisão do cálculo do limite inferior e o bom funcionamento e consistência do cache aplicado para acelerar buscas ao limite calculado.
  
Em código
```py
def test_calculate_lower_bound_and_cache(simple_instance):
    """Verifica se o cálculo e o cache do limite inferior funcionam"""
    bnb = simple_instance
    machine_times = [0, 0, 0]

    lb1 = bnb.calculate_lower_bound(machine_times, 0)
    assert isinstance(lb1, float)

    key = (tuple(machine_times), 0)
    assert key in bnb.lb_cache

    # A segunda chamada deve vir do cache (mesmo valor)
    lb2 = bnb.calculate_lower_bound(machine_times, 0)
    assert lb1 == lb2
```

- `test_solve_returns_valid_solution`: Testa o retorno do método `solve` (solução ótima, menor tempo total (*makespan*) e as métricas) e sua consistência.

Em código:
```py
def test_solve_returns_valid_solution(simple_instance):
    """Verifica se o método solve retorna uma solução e métricas coerentes"""
    bnb = simple_instance
    solution, makespan, metrics = bnb.solve()

    # Tipos corretos
    assert isinstance(solution, list)
    assert isinstance(makespan, (int, float))
    assert isinstance(metrics, dict)

    # Tamanho da solução deve corresponder ao número de jobs
    assert len(solution) == len(bnb.processing_times)

    # Makespan deve ser finito e positivo
    assert makespan < float('inf')
    assert makespan >= 0

    # Métricas principais devem estar presentes
    expected_keys = [
        'nodes_explored', 'nodes_pruned', 'pruning_ratio',
        'max_depth', 'execution_time', 'feasible_solutions',
        'best_makespan', 'theoretical_lb'
    ]
    for k in expected_keys:
        assert k in metrics

    # Pruning ratio entre 0 e 1
    assert 0 <= metrics['pruning_ratio'] <= 1
```

- `test_best_makespan_consistency`: Confere consitência do *makespan* armazenado.

Em código
```py
def test_best_makespan_consistency(simple_instance):
    """Confere se o makespan armazenado corresponde ao das métricas"""
    bnb = simple_instance
    _, makespan, metrics = bnb.solve()
    assert makespan == bnb.best_makespan
    assert pytest.approx(makespan) == metrics['best_makespan']
```

- `test_print_detailed_analysis_output`: Garante que a análise detalhada (implementada pelo método `print_detailed_analysis`) seja exibida corretamente na tela.

Em código:
```py
def test_print_detailed_analysis_output(simple_instance, capsys):
    """Garante que print_detailed_analysis imprime corretamente"""
    bnb = simple_instance
    solution, makespan, metrics = bnb.solve()
    bnb.print_detailed_analysis(solution, metrics)

    output = capsys.readouterr().out
    assert "RELATÓRIO COMPLETO" in output
    assert "MÉTRICAS DE EXECUÇÃO" in output
    assert "ANÁLISE DA SOLUÇÃO" in output
    assert "Makespan alcançado" in output
```

- `test_branching_produces_valid_assignments`: Verifica o fluxo completo do algoritmo, principalmente as atribuições de máquinas, porém usando uma instãncia diferente (e menor) do problema, com *4 tarefas*.

Em código:
```py
def test_branching_produces_valid_assignments():
    """Testa uma instância com 4 jobs para garantir que as atribuições são válidas"""
    bnb = OptimizedBranchAndBound([2, 4, 6, 8])
    solution, makespan, metrics = bnb.solve()

    # Cada job deve ser atribuído a uma máquina válida (0, 1, 2)
    assert all(m in [0, 1, 2] for m in solution)

    # Makespan mínimo deve ser >= limite inferior teórico
    assert makespan >= metrics['theoretical_lb']
```

> [!NOTE]
> Execute o comando `pytest -v` no diretório do projeto caso esteja interessado em executar os testes aqui descritos e detalhados.

## Como executar a aplicação

Essa seção tem como propósito orientá-lo(a) durante o processo de **instalação** e **execução** da aplicação.

### Pré-requisitos

Certifique-se de ter ambas as ferramentas presentes na máquina em que deseja executar.

- [Python](https://www.python.org/downloads/)
- [Git](https://git-scm.com/install/)

---

### Clonando o repositório

Execute o seguinte comando em algum terminal:
```bash
git clone https://github.com/J0ji-dev/Escalonamento_Maquinas.git
```

Como resultado, será gerada uma pasta `Escalonamento_Maquinas` na pasta atual, navegue até ela.

---

### Instalando dependências

Na mesma pasta onde paramos na etapa anterior, vamos executar o seguinte comando:
```bash
pip install -r requirements.txt
```

Esse comando irá instalar as dependências listadas no já mencionado `requirements.txt`.

---

### Executando

Por fim, basta rodar o seguinte comando:
```bash
streamlit run app.py
```

Você será redirecionado ao navegador, onde poderá ver o sistema funcionado.
