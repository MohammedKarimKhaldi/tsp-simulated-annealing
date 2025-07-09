import numpy as np
import random
import math
from typing import List, Tuple, Optional, Dict, Any
import time
import json
from dataclasses import dataclass
from enum import Enum

class AnnealingType(Enum):
    SIMULATED = "simulated"
    QUANTUM = "quantum"

@dataclass
class TSPSolution:
    """Represents a TSP solution with route and cost."""
    route: List[int]
    cost: float
    computation_time: float
    iterations: int
    annealing_type: AnnealingType
    evolution_history: Optional[List[Tuple[int, float, List[int]]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route,
            "cost": self.cost,
            "computation_time": self.computation_time,
            "iterations": self.iterations,
            "annealing_type": self.annealing_type.value,
            "evolution_history": self.evolution_history
        }

class TSPSolver:
    """
    Traveling Salesman Problem solver using simulated annealing and quantum annealing.
    
    This solver implements both classical simulated annealing and quantum-inspired
    annealing approaches for solving the TSP.
    """
    
    def __init__(self, distance_matrix: np.ndarray):
        """
        Initialize the TSP solver.
        
        Args:
            distance_matrix: NxN matrix of distances between cities
        """
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        self.best_solution = None
        self.best_cost = float('inf')
        
    def calculate_route_cost(self, route: List[int]) -> float:
        """Calculate the total cost of a given route."""
        total_cost = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            total_cost += self.distance_matrix[from_city][to_city]
        return total_cost
    
    def generate_initial_solution(self) -> List[int]:
        """Generate a random initial solution."""
        route = list(range(self.n_cities))
        random.shuffle(route)
        return route
    
    def generate_neighbor(self, route: List[int]) -> List[int]:
        """Generate a neighbor solution using 2-opt swap."""
        new_route = route.copy()
        i, j = sorted(random.sample(range(self.n_cities), 2))
        new_route[i:j+1] = reversed(new_route[i:j+1])
        return new_route
    
    def acceptance_probability(self, current_cost: float, new_cost: float, temperature: float) -> float:
        """Calculate acceptance probability for simulated annealing."""
        if new_cost < current_cost:
            return 1.0
        return math.exp((current_cost - new_cost) / temperature)
    
    def quantum_acceptance_probability(self, current_cost: float, new_cost: float, 
                                     temperature: float, quantum_factor: float) -> float:
        """Calculate acceptance probability for quantum annealing."""
        if new_cost < current_cost:
            return 1.0
        
        # Quantum tunneling effect - allows escaping local minima
        delta_e = new_cost - current_cost
        quantum_tunneling = quantum_factor * math.exp(-delta_e / (temperature * 0.5))  # Enhanced tunneling
        classical_prob = math.exp(-delta_e / temperature)
        
        return max(classical_prob, quantum_tunneling)
    
    def simulated_annealing(self, 
                          initial_temp: float = 1000.0,
                          cooling_rate: float = 0.95,
                          min_temp: float = 1.0,
                          max_iterations: int = 10000,
                          track_evolution: bool = False) -> TSPSolution:
        """
        Solve TSP using simulated annealing.
        
        Args:
            initial_temp: Initial temperature
            cooling_rate: Rate at which temperature decreases
            min_temp: Minimum temperature
            max_iterations: Maximum number of iterations
            track_evolution: Whether to track solution evolution for GIF generation
            
        Returns:
            TSPSolution object with the best route found
        """
        start_time = time.time()
        
        current_route = self.generate_initial_solution()
        current_cost = self.calculate_route_cost(current_route)
        
        best_route = current_route.copy()
        best_cost = current_cost
        
        temperature = initial_temp
        iteration = 0
        
        # Track evolution for GIF generation
        evolution_history = []
        if track_evolution:
            evolution_history.append((iteration, best_cost, best_route.copy()))
        
        # Improved cooling schedule: exponential decay
        cooling_factor = math.pow(min_temp / initial_temp, 1.0 / max_iterations)
        
        while temperature > min_temp and iteration < max_iterations:
            # Generate neighbor solution
            new_route = self.generate_neighbor(current_route)
            new_cost = self.calculate_route_cost(new_route)
            
            # Accept or reject the new solution
            if self.acceptance_probability(current_cost, new_cost, temperature) > random.random():
                current_route = new_route
                current_cost = new_cost
                
                # Update best solution if necessary
                if current_cost < best_cost:
                    best_route = current_route.copy()
                    best_cost = current_cost
                    # Track only when best solution improves
                    if track_evolution:
                        evolution_history.append((iteration, best_cost, best_route.copy()))
            
            # Improved cooling schedule
            temperature = initial_temp * (cooling_factor ** iteration)
            iteration += 1
        
        computation_time = time.time() - start_time
        
        return TSPSolution(
            route=best_route,
            cost=best_cost,
            computation_time=computation_time,
            iterations=iteration,
            annealing_type=AnnealingType.SIMULATED,
            evolution_history=evolution_history if track_evolution else None
        )
    
    def quantum_annealing(self,
                         initial_temp: float = 1000.0,
                         cooling_rate: float = 0.95,
                         min_temp: float = 1.0,
                         max_iterations: int = 10000,
                         quantum_factor: float = 0.2,
                         quantum_decay: float = 0.999,
                         track_evolution: bool = False) -> TSPSolution:
        """
        Solve TSP using quantum-inspired annealing.
        
        Args:
            initial_temp: Initial temperature
            cooling_rate: Rate at which temperature decreases
            min_temp: Minimum temperature
            max_iterations: Maximum number of iterations
            quantum_factor: Initial quantum tunneling factor
            quantum_decay: Rate at which quantum factor decreases
            track_evolution: Whether to track solution evolution for GIF generation
            
        Returns:
            TSPSolution object with the best route found
        """
        start_time = time.time()
        
        current_route = self.generate_initial_solution()
        current_cost = self.calculate_route_cost(current_route)
        
        best_route = current_route.copy()
        best_cost = current_cost
        
        temperature = initial_temp
        current_quantum_factor = quantum_factor
        iteration = 0
        
        # Track evolution for GIF generation
        evolution_history = []
        if track_evolution:
            evolution_history.append((iteration, best_cost, best_route.copy()))
        
        # Improved cooling schedule: exponential decay
        cooling_factor = math.pow(min_temp / initial_temp, 1.0 / max_iterations)
        
        while temperature > min_temp and iteration < max_iterations:
            # Generate neighbor solution
            new_route = self.generate_neighbor(current_route)
            new_cost = self.calculate_route_cost(new_route)
            
            # Accept or reject using quantum acceptance probability
            if self.quantum_acceptance_probability(current_cost, new_cost, 
                                                 temperature, current_quantum_factor) > random.random():
                current_route = new_route
                current_cost = new_cost
                
                # Update best solution if necessary
                if current_cost < best_cost:
                    best_route = current_route.copy()
                    best_cost = current_cost
                    # Track only when best solution improves
                    if track_evolution:
                        evolution_history.append((iteration, best_cost, best_route.copy()))
            
            # Improved cooling schedule and quantum factor decay
            temperature = initial_temp * (cooling_factor ** iteration)
            current_quantum_factor *= quantum_decay
            iteration += 1
        
        computation_time = time.time() - start_time
        
        return TSPSolution(
            route=best_route,
            cost=best_cost,
            computation_time=computation_time,
            iterations=iteration,
            annealing_type=AnnealingType.QUANTUM,
            evolution_history=evolution_history if track_evolution else None
        )
    
    def solve(self, 
              annealing_type: AnnealingType = AnnealingType.SIMULATED,
              track_evolution: bool = False,
              **kwargs) -> TSPSolution:
        """
        Solve TSP using specified annealing method.
        
        Args:
            annealing_type: Type of annealing to use
            track_evolution: Whether to track solution evolution for GIF generation
            **kwargs: Additional parameters for the annealing method
            
        Returns:
            TSPSolution object with the best route found
        """
        if annealing_type == AnnealingType.SIMULATED:
            return self.simulated_annealing(track_evolution=track_evolution, **kwargs)
        elif annealing_type == AnnealingType.QUANTUM:
            return self.quantum_annealing(track_evolution=track_evolution, **kwargs)
        else:
            raise ValueError(f"Unknown annealing type: {annealing_type}")
    
    def compare_methods(self, track_evolution: bool = False, **kwargs) -> Dict[str, TSPSolution]:
        """
        Compare both annealing methods on the same problem.
        
        Args:
            track_evolution: Whether to track solution evolution for GIF generation
            **kwargs: Parameters to pass to both methods
            
        Returns:
            Dictionary with results from both methods
        """
        results = {}
        
        # Solve with simulated annealing
        sa_solution = self.simulated_annealing(track_evolution=track_evolution, **kwargs)
        results['simulated_annealing'] = sa_solution
        
        # Solve with quantum annealing
        qa_solution = self.quantum_annealing(track_evolution=track_evolution, **kwargs)
        results['quantum_annealing'] = qa_solution
        
        return results
    
    def save_solution(self, solution: TSPSolution, filename: str):
        """Save solution to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(solution.to_dict(), f, indent=2)
    
    def load_solution(self, filename: str) -> TSPSolution:
        """Load solution from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return TSPSolution(
            route=data['route'],
            cost=data['cost'],
            computation_time=data['computation_time'],
            iterations=data['iterations'],
            annealing_type=AnnealingType(data['annealing_type']),
            evolution_history=data.get('evolution_history')
        ) 