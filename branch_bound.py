from dataclasses import dataclass
from typing import List, Tuple, Dict
import time
import heapq

# ----------------------------------------------------
# ⚙️ ETAPA 3: EXECUÇÃO DO ALGORITMO
# ----------------------------------------------------

@dataclass
class SearchMetrics:
    nodes_explored: int = 0     # Nós explorados
    nodes_pruned: int = 0       # Nós podados
    max_depth: int = 0          # Profundidade máxima
    start_time: float = 0       # Marcador começo
    end_time: float = 0         # Marcador encerramento
    feasible_solutions: int = 0 # Soluções viáveis


class BranchAndBound:
    def __init__(self, processing_times: List[float]):
        self.processing_times = processing_times
        self.num_jobs = len(processing_times)
        self.num_machines = 3
        self.best_solution = None
        self.best_makespan = float("inf")
        self.metrics = SearchMetrics()

    def lower_bound(self, machine_times: List[float]) -> float:
        return max(machine_times)

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