import pytest
from branch_and_bound import OptimizedBranchAndBound


@pytest.fixture
def simple_instance():
    """Instância simples e pequena para facilitar a verificação"""
    processing_times = [5, 10, 15]
    return OptimizedBranchAndBound(processing_times)


def test_initialization(simple_instance):
    """Verifica se a classe inicializa corretamente"""
    bnb = simple_instance
    assert bnb.num_jobs == 3
    assert bnb.num_machines == 3
    assert isinstance(bnb.metrics, type(bnb.metrics))  # tipo interno
    assert bnb.best_makespan == float('inf')
    assert bnb.best_solution is None
    assert isinstance(bnb.lb_cache, dict)


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


def test_best_makespan_consistency(simple_instance):
    """Confere se o makespan armazenado corresponde ao das métricas"""
    bnb = simple_instance
    _, makespan, metrics = bnb.solve()
    assert makespan == bnb.best_makespan
    assert pytest.approx(makespan) == metrics['best_makespan']


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


def test_branching_produces_valid_assignments():
    """Testa uma instância com 4 jobs para garantir que as atribuições são válidas"""
    bnb = OptimizedBranchAndBound([2, 4, 6, 8])
    solution, makespan, metrics = bnb.solve()

    # Cada job deve ser atribuído a uma máquina válida (0, 1, 2)
    assert all(m in [0, 1, 2] for m in solution)

    # Makespan mínimo deve ser >= limite inferior teórico
    assert makespan >= metrics['theoretical_lb']
