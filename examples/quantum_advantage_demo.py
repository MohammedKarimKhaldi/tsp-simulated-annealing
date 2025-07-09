#!/usr/bin/env python3
"""
Demonstration of Quantum Annealing Advantage

This example shows how quantum annealing can outperform simulated annealing
on challenging TSP instances with many local minima.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src import TSPSolver, TSPDataGenerator, TSPVisualizer, AnnealingType

def demonstrate_quantum_advantage():
    """Demonstrate quantum annealing advantage on challenging instances."""
    print("Quantum Annealing Advantage Demonstration")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate a challenging clustered TSP instance
    print("\n1. Generating challenging TSP instance...")
    generator = TSPDataGenerator(seed=42)
    distance_matrix = generator.generate_clustered_tsp(
        n_cities=30,
        n_clusters=5,
        cluster_radius=8.0,
        inter_cluster_distance=60.0
    )
    
    print(f"Generated clustered TSP with {len(distance_matrix)} cities in 5 clusters")
    print(f"Distance matrix shape: {distance_matrix.shape}")
    
    # Create solver and visualizer
    solver = TSPSolver(distance_matrix)
    visualizer = TSPVisualizer(distance_matrix)
    
    # Test parameters for both methods
    test_params = {
        'initial_temp': 1000.0,
        'cooling_rate': 0.97,  # Slower cooling for better exploration
        'min_temp': 0.1,
        'max_iterations': 15000
    }
    
    # Run multiple trials to show consistency
    n_trials = 5
    sa_results = []
    qa_results = []
    
    print(f"\n2. Running {n_trials} trials for each method...")
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        
        # Simulated Annealing
        sa_solution = solver.simulated_annealing(**test_params)
        sa_results.append(sa_solution.cost)
        print(f"  SA Cost: {sa_solution.cost:.2f}")
        
        # Quantum Annealing
        qa_solution = solver.quantum_annealing(
            **test_params,
            quantum_factor=0.15,  # Higher quantum factor for better tunneling
            quantum_decay=0.995   # Slower quantum decay
        )
        qa_results.append(qa_solution.cost)
        print(f"  QA Cost: {qa_solution.cost:.2f}")
    
    # Analyze results
    print(f"\n3. Results Analysis:")
    print("-" * 30)
    
    sa_mean = np.mean(sa_results)
    sa_std = np.std(sa_results)
    qa_mean = np.mean(qa_results)
    qa_std = np.std(qa_results)
    
    print(f"Simulated Annealing:")
    print(f"  Mean Cost: {sa_mean:.2f} ± {sa_std:.2f}")
    print(f"  Best Cost: {min(sa_results):.2f}")
    print(f"  Worst Cost: {max(sa_results):.2f}")
    
    print(f"\nQuantum Annealing:")
    print(f"  Mean Cost: {qa_mean:.2f} ± {qa_std:.2f}")
    print(f"  Best Cost: {min(qa_results):.2f}")
    print(f"  Worst Cost: {max(qa_results):.2f}")
    
    # Calculate improvement
    improvement = ((sa_mean - qa_mean) / sa_mean) * 100
    print(f"\nQuantum Annealing Improvement: {improvement:.2f}%")
    
    # Statistical significance
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(sa_results, qa_results)
    print(f"Statistical significance (p-value): {p_value:.4f}")
    
    if p_value < 0.05:
        print("✅ Statistically significant improvement!")
    else:
        print("❌ No statistically significant difference")
    
    # Create detailed comparison plot
    print(f"\n4. Creating detailed comparison...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Trial-by-trial comparison
    trials = range(1, n_trials + 1)
    ax1.plot(trials, sa_results, 'o-', label='Simulated Annealing', linewidth=2, markersize=8)
    ax1.plot(trials, qa_results, 's-', label='Quantum Annealing', linewidth=2, markersize=8)
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Cost')
    ax1.set_title('Cost Comparison Across Trials')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot comparison
    ax2.boxplot([sa_results, qa_results], labels=['Simulated\nAnnealing', 'Quantum\nAnnealing'])
    ax2.set_ylabel('Cost')
    ax2.set_title('Cost Distribution Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Histogram comparison
    ax3.hist(sa_results, alpha=0.7, label='Simulated Annealing', bins=8, edgecolor='black')
    ax3.hist(qa_results, alpha=0.7, label='Quantum Annealing', bins=8, edgecolor='black')
    ax3.set_xlabel('Cost')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Cost Distribution Histogram')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Improvement analysis
    improvements = [(sa - qa) / sa * 100 for sa, qa in zip(sa_results, qa_results)]
    ax4.bar(trials, improvements, color='green', alpha=0.7, edgecolor='black')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Trial Number')
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('Quantum Annealing Improvement per Trial')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/quantum_advantage_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Show best solutions
    print(f"\n5. Best Solutions Comparison:")
    print("-" * 30)
    
    best_sa_idx = np.argmin(sa_results)
    best_qa_idx = np.argmin(qa_results)
    
    # Re-run to get the best solutions
    print("Re-running to get best solutions...")
    
    # Get best SA solution
    for i in range(n_trials):
        sa_solution = solver.simulated_annealing(**test_params)
        if i == best_sa_idx:
            best_sa_solution = sa_solution
            break
    
    # Get best QA solution
    for i in range(n_trials):
        qa_solution = solver.quantum_annealing(
            **test_params,
            quantum_factor=0.15,
            quantum_decay=0.995
        )
        if i == best_qa_idx:
            best_qa_solution = qa_solution
            break
    
    solutions = {
        'Best Simulated Annealing': best_sa_solution,
        'Best Quantum Annealing': best_qa_solution
    }
    
    # Plot best routes
    print("Plotting best routes...")
    visualizer.plot_multiple_routes(solutions, save_path='results/best_routes_comparison.png')
    
    # Generate summary report
    report = visualizer.create_summary_report(solutions, save_path='results/quantum_advantage_report.txt')
    print("\nSummary Report:")
    print(report)
    
    print(f"\n✅ Quantum Annealing Advantage Demonstration Completed!")
    print(f"Check the 'results/' directory for detailed outputs.")

def main():
    """Main function."""
    try:
        demonstrate_quantum_advantage()
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 