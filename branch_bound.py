from dataclasses import dataclass
from typing import List, Tuple, Dict
import time
import heapq

@dataclass
class SearchMetrics:
    nodes_explored: int = 0     # Nós explorados
    nodes_pruned: int = 0       # Nós podados
    max_depth: int = 0          # Profundidade máxima
    start_time: float = 0       # Marcador começo
    end_time: float = 0         # Marcador encerramento
    feasible_solutions: int = 0 # Soluções viáveis

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
            "nós explorados": self.metrics.nodes_explored,
            "nós podados": self.metrics.nodes_pruned,
            "profundidade máxima": self.metrics.max_depth,
            "tempo de execução": self.metrics.end_time - self.metrics.start_time,
            "soluções viáveis": self.metrics.feasible_solutions,
            "menor tempo máximo": self.best_makespan,
            "taxa de poda": self.metrics.nodes_pruned / max(1, self.metrics.nodes_explored),
        }
