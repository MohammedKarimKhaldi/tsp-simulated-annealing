#!/usr/bin/env python3
"""
GIF Demo: Visualize the evolution of TSP solutions for both simulated and quantum annealing.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src import TSPSolver, TSPDataGenerator, AnnealingType
from src.gif_generator import TSPGifGenerator

def main():
    print("TSP GIF Demo - Solution Evolution Visualization")
    print("=" * 50)
    
    # Generate a TSP instance
    generator = TSPDataGenerator(seed=42)
    distance_matrix = generator.generate_euclidean_tsp(n_cities=15)
    
    # Create solver and GIF generator
    solver = TSPSolver(distance_matrix)
    gifgen = TSPGifGenerator(distance_matrix)
    
    # Solve with simulated annealing (track evolution)
    print("\nSolving with Simulated Annealing (tracking evolution)...")
    sa_solution = solver.simulated_annealing(
        initial_temp=1000.0,
        cooling_rate=0.97,
        min_temp=1.0,
        max_iterations=3000,
        track_evolution=True
    )
    print(f"Simulated Annealing best cost: {sa_solution.cost:.2f}")
    
    # Solve with quantum annealing (track evolution)
    print("\nSolving with Quantum Annealing (tracking evolution)...")
    qa_solution = solver.quantum_annealing(
        initial_temp=1000.0,
        cooling_rate=0.97,
        min_temp=1.0,
        max_iterations=3000,
        quantum_factor=0.2,
        quantum_decay=0.999,
        track_evolution=True
    )
    print(f"Quantum Annealing best cost: {qa_solution.cost:.2f}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Generate GIFs (non-looping)
    print("\nGenerating non-looping GIF for Simulated Annealing...")
    gifgen.create_evolution_gif(sa_solution, 'results/sa_evolution_noloop.gif', fps=5, duration_per_frame=0.2, loop=False)
    
    print("\nGenerating non-looping GIF for Quantum Annealing...")
    gifgen.create_evolution_gif(qa_solution, 'results/qa_evolution_noloop.gif', fps=5, duration_per_frame=0.2, loop=False)
    
    # Generate MP4 videos (no loop by default)
    print("\nGenerating MP4 video for Simulated Annealing...")
    gifgen.create_evolution_video(sa_solution, 'results/sa_evolution.mp4', fps=5)
    
    print("\nGenerating MP4 video for Quantum Annealing...")
    gifgen.create_evolution_video(qa_solution, 'results/qa_evolution.mp4', fps=5)
    
    # Generate comparison GIF (non-looping)
    print("\nGenerating comparison GIF...")
    gifgen.create_comparison_gif(
        {'Simulated Annealing': sa_solution, 'Quantum Annealing': qa_solution},
        'results/annealing_comparison_noloop.gif',
        fps=5, duration_per_frame=0.2
    )
    
    print("\nAll GIFs generated in the 'results/' directory!")

if __name__ == "__main__":
    main() 