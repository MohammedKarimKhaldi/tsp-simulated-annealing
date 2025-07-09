#!/usr/bin/env python3
"""
Generate GitHub-compatible GIFs for TSP solutions.

This script creates animated GIFs that can be embedded directly in GitHub README files,
showing the evolution of both simulated annealing and quantum annealing solutions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from tsp_solver import TSPSolver, AnnealingType
from gif_generator import TSPGifGenerator
from data_generator import TSPDataGenerator
import numpy as np

def main():
    """Generate GitHub-compatible GIFs for TSP solutions."""
    print("Generating GitHub-compatible GIFs for TSP solutions...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'github_gifs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate test data
    data_gen = TSPDataGenerator(seed=42)
    distance_matrix = data_gen.generate_euclidean_tsp(15)
    
    # Initialize solver and GIF generator
    solver = TSPSolver(distance_matrix)
    gif_gen = TSPGifGenerator(distance_matrix)
    
    # Solve with simulated annealing
    print("\n1. Solving with Simulated Annealing...")
    sa_solution = solver.simulated_annealing(
        initial_temp=1000, 
        cooling_rate=0.95, 
        max_iterations=1000,
        track_evolution=True
    )
    
    # Create GitHub-compatible GIF for simulated annealing
    sa_gif_path = os.path.join(output_dir, 'simulated_annealing_evolution.gif')
    gif_gen.create_github_compatible_gif(sa_solution, sa_gif_path)
    
    # Solve with quantum annealing
    print("\n2. Solving with Quantum Annealing...")
    qa_solution = solver.quantum_annealing(
        initial_temp=1000,
        cooling_rate=0.95,
        max_iterations=1000,
        quantum_factor=0.3,
        track_evolution=True
    )
    
    # Create GitHub-compatible GIF for quantum annealing
    qa_gif_path = os.path.join(output_dir, 'quantum_annealing_evolution.gif')
    gif_gen.create_github_compatible_gif(qa_solution, qa_gif_path)
    
    # Print results
    print(f"\n=== Results ===")
    print(f"Simulated Annealing:")
    print(f"  Final Cost: {sa_solution.cost:.2f}")
    print(f"  GIF: {sa_gif_path}")
    print(f"\nQuantum Annealing:")
    print(f"  Final Cost: {qa_solution.cost:.2f}")
    print(f"  GIF: {qa_gif_path}")
    
    print(f"\nGitHub-compatible GIFs saved to: {output_dir}")
    print("These GIFs can be embedded directly in GitHub README files!")

if __name__ == "__main__":
    main() 