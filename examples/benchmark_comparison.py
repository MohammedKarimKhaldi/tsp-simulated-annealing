#!/usr/bin/env python3
"""
Benchmark comparison example for the TSP solver.

This example demonstrates how to:
1. Generate a benchmark suite of TSP instances
2. Test both annealing methods on different instance types
3. Compare performance across different problem sizes
4. Analyze parameter sensitivity
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import TSPSolver, TSPDataGenerator, TSPVisualizer, AnnealingType

def run_benchmark_suite():
    """Run comprehensive benchmark on different TSP instances."""
    print("TSP Solver - Benchmark Comparison")
    print("=" * 40)
    
    # Generate benchmark suite
    print("\n1. Generating benchmark suite...")
    generator = TSPDataGenerator(seed=42)
    instances = generator.generate_benchmark_suite("data")
    
    # Benchmark parameters
    benchmark_params = {
        'initial_temp': 1000.0,
        'cooling_rate': 0.95,
        'min_temp': 1.0,
        'max_iterations': 10000,
        'quantum_factor': 0.1,
        'quantum_decay': 0.99
    }
    
    # Results storage
    results = []
    
    print("\n2. Running benchmarks...")
    for name, distance_matrix in instances:
        print(f"\nTesting instance: {name}")
        print(f"  Cities: {len(distance_matrix)}")
        
        solver = TSPSolver(distance_matrix)
        
        # Test simulated annealing
        print("  Running Simulated Annealing...")
        sa_solution = solver.simulated_annealing(**benchmark_params)
        
        # Test quantum annealing
        print("  Running Quantum Annealing...")
        qa_solution = solver.quantum_annealing(**benchmark_params)
        
        # Store results
        results.append({
            'instance': name,
            'n_cities': len(distance_matrix),
            'instance_type': name.split('_')[1],
            'sa_cost': sa_solution.cost,
            'sa_time': sa_solution.computation_time,
            'sa_iterations': sa_solution.iterations,
            'qa_cost': qa_solution.cost,
            'qa_time': qa_solution.computation_time,
            'qa_iterations': qa_solution.iterations,
            'cost_improvement': ((sa_solution.cost - qa_solution.cost) / sa_solution.cost * 100) if sa_solution.cost > qa_solution.cost else 0,
            'time_ratio': qa_solution.computation_time / sa_solution.computation_time if sa_solution.computation_time > 0 else 1.0
        })
        
        print(f"    SA Cost: {sa_solution.cost:.2f}, Time: {sa_solution.computation_time:.3f}s")
        print(f"    QA Cost: {qa_solution.cost:.2f}, Time: {qa_solution.computation_time:.3f}s")
    
    return pd.DataFrame(results)

def analyze_results(df):
    """Analyze and visualize benchmark results."""
    print("\n3. Analyzing results...")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Save results to CSV
    df.to_csv('results/benchmark_results.csv', index=False)
    print("Results saved to 'results/benchmark_results.csv'")
    
    # Summary statistics
    print("\nSummary Statistics:")
    print("-" * 30)
    
    print(f"Total instances tested: {len(df)}")
    print(f"SA wins: {(df['sa_cost'] < df['qa_cost']).sum()}")
    print(f"QA wins: {(df['qa_cost'] < df['sa_cost']).sum()}")
    print(f"Ties: {(df['sa_cost'] == df['qa_cost']).sum()}")
    
    print(f"\nAverage cost improvement with QA: {df['cost_improvement'].mean():.2f}%")
    print(f"Average time ratio (QA/SA): {df['time_ratio'].mean():.2f}")
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Cost comparison by instance type
    instance_types = df['instance_type'].unique()
    sa_costs = [df[df['instance_type'] == it]['sa_cost'].mean() for it in instance_types]
    qa_costs = [df[df['instance_type'] == it]['qa_cost'].mean() for it in instance_types]
    
    x = np.arange(len(instance_types))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, sa_costs, width, label='Simulated Annealing', alpha=0.8)
    axes[0, 0].bar(x + width/2, qa_costs, width, label='Quantum Annealing', alpha=0.8)
    axes[0, 0].set_xlabel('Instance Type')
    axes[0, 0].set_ylabel('Average Cost')
    axes[0, 0].set_title('Cost Comparison by Instance Type')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(instance_types)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cost vs Problem Size
    axes[0, 1].scatter(df['n_cities'], df['sa_cost'], label='Simulated Annealing', alpha=0.7)
    axes[0, 1].scatter(df['n_cities'], df['qa_cost'], label='Quantum Annealing', alpha=0.7)
    axes[0, 1].set_xlabel('Number of Cities')
    axes[0, 1].set_ylabel('Cost')
    axes[0, 1].set_title('Cost vs Problem Size')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Time comparison
    axes[1, 0].scatter(df['n_cities'], df['sa_time'], label='Simulated Annealing', alpha=0.7)
    axes[1, 0].scatter(df['n_cities'], df['qa_time'], label='Quantum Annealing', alpha=0.7)
    axes[1, 0].set_xlabel('Number of Cities')
    axes[1, 0].set_ylabel('Computation Time (seconds)')
    axes[1, 0].set_title('Time vs Problem Size')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cost improvement distribution
    axes[1, 1].hist(df['cost_improvement'], bins=10, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Cost Improvement (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Cost Improvements')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/benchmark_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Detailed analysis by instance type
    print("\nDetailed Analysis by Instance Type:")
    print("-" * 40)
    
    for instance_type in instance_types:
        type_df = df[df['instance_type'] == instance_type]
        print(f"\n{instance_type.upper()}:")
        print(f"  Instances: {len(type_df)}")
        print(f"  Average SA cost: {type_df['sa_cost'].mean():.2f}")
        print(f"  Average QA cost: {type_df['qa_cost'].mean():.2f}")
        print(f"  Average improvement: {type_df['cost_improvement'].mean():.2f}%")
        print(f"  Average time ratio: {type_df['time_ratio'].mean():.2f}")

def parameter_sensitivity_analysis():
    """Analyze sensitivity to different parameters."""
    print("\n5. Parameter Sensitivity Analysis...")
    
    # Generate a test instance
    generator = TSPDataGenerator(seed=42)
    distance_matrix = generator.generate_euclidean_tsp(20)
    solver = TSPSolver(distance_matrix)
    visualizer = TSPVisualizer(distance_matrix)
    
    # Test cooling rate sensitivity
    print("Testing cooling rate sensitivity...")
    cooling_rates = [0.90, 0.92, 0.94, 0.96, 0.98]
    visualizer.plot_parameter_sensitivity(
        solver, 'cooling_rate', cooling_rates, 
        AnnealingType.SIMULATED, n_trials=3,
        save_path='results/cooling_rate_sensitivity.png'
    )
    
    # Test quantum factor sensitivity
    print("Testing quantum factor sensitivity...")
    quantum_factors = [0.05, 0.1, 0.15, 0.2, 0.25]
    visualizer.plot_parameter_sensitivity(
        solver, 'quantum_factor', quantum_factors,
        AnnealingType.QUANTUM, n_trials=3,
        save_path='results/quantum_factor_sensitivity.png'
    )

def main():
    """Main function to run the benchmark comparison."""
    try:
        # Run benchmark suite
        results_df = run_benchmark_suite()
        
        # Analyze results
        analyze_results(results_df)
        
        # Parameter sensitivity analysis
        parameter_sensitivity_analysis()
        
        print("\nBenchmark comparison completed successfully!")
        print("Check the 'results/' directory for detailed outputs.")
        
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 