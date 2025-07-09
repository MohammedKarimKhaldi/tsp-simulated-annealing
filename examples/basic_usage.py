#!/usr/bin/env python3
"""
Basic usage example for the TSP solver.

This example demonstrates how to:
1. Generate a TSP instance
2. Solve it using both simulated and quantum annealing
3. Compare the results
4. Visualize the solutions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src import TSPSolver, TSPDataGenerator, TSPVisualizer, AnnealingType

def main():
    print("TSP Solver - Basic Usage Example")
    print("=" * 40)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Generate a TSP instance
    print("\n1. Generating TSP instance...")
    generator = TSPDataGenerator(seed=42)
    distance_matrix = generator.generate_euclidean_tsp(n_cities=15)
    
    print(f"Generated TSP instance with {len(distance_matrix)} cities")
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Min distance: {np.min(distance_matrix[distance_matrix > 0]):.2f}")
    print(f"Max distance: {np.max(distance_matrix):.2f}")
    
    # 2. Create solver and visualizer
    solver = TSPSolver(distance_matrix)
    visualizer = TSPVisualizer(distance_matrix)
    
    # 3. Solve using simulated annealing
    print("\n2. Solving with Simulated Annealing...")
    sa_solution = solver.simulated_annealing(
        initial_temp=1000.0,
        cooling_rate=0.95,
        min_temp=1.0,
        max_iterations=5000
    )
    
    print(f"SA Solution:")
    print(f"  Cost: {sa_solution.cost:.2f}")
    print(f"  Route: {sa_solution.route}")
    print(f"  Time: {sa_solution.computation_time:.3f}s")
    print(f"  Iterations: {sa_solution.iterations}")
    
    # 4. Solve using quantum annealing
    print("\n3. Solving with Quantum Annealing...")
    qa_solution = solver.quantum_annealing(
        initial_temp=1000.0,
        cooling_rate=0.95,
        min_temp=1.0,
        max_iterations=5000,
        quantum_factor=0.1,
        quantum_decay=0.99
    )
    
    print(f"QA Solution:")
    print(f"  Cost: {qa_solution.cost:.2f}")
    print(f"  Route: {qa_solution.route}")
    print(f"  Time: {qa_solution.computation_time:.3f}s")
    print(f"  Iterations: {qa_solution.iterations}")
    
    # 5. Compare results
    print("\n4. Comparing Results...")
    solutions = {
        'Simulated Annealing': sa_solution,
        'Quantum Annealing': qa_solution
    }
    
    best_method = min(solutions.keys(), key=lambda x: solutions[x].cost)
    print(f"Best method: {best_method}")
    print(f"Cost improvement: {((max(s.cost for s in solutions.values()) - min(s.cost for s in solutions.values())) / max(s.cost for s in solutions.values()) * 100):.1f}%")
    
    # 6. Visualize results
    print("\n5. Creating visualizations...")
    
    # Plot individual routes
    print("  - Plotting individual routes...")
    visualizer.plot_route(sa_solution, title="Simulated Annealing Solution")
    visualizer.plot_route(qa_solution, title="Quantum Annealing Solution")
    
    # Plot comparison
    print("  - Plotting comparison...")
    visualizer.plot_convergence_comparison(solutions)
    
    # Plot multiple routes side by side
    print("  - Plotting multiple routes...")
    visualizer.plot_multiple_routes(solutions)
    
    # 7. Generate summary report
    print("\n6. Generating summary report...")
    report = visualizer.create_summary_report(solutions)
    print(report)
    
    # Save results
    print("\n7. Saving results...")
    os.makedirs('results', exist_ok=True)
    
    # Save solutions
    solver.save_solution(sa_solution, 'results/sa_solution.json')
    solver.save_solution(qa_solution, 'results/qa_solution.json')
    
    # Save distance matrix
    np.save('results/distance_matrix.npy', distance_matrix)
    
    print("Results saved to 'results/' directory")
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 