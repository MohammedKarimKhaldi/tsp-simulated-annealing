#!/usr/bin/env python3
"""
Simple test script to verify the TSP solver functionality.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import TSPSolver, TSPDataGenerator, TSPVisualizer, AnnealingType

def test_basic_functionality():
    """Test basic solver functionality."""
    print("Testing basic TSP solver functionality...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate a small TSP instance
    generator = TSPDataGenerator(seed=42)
    distance_matrix = generator.generate_euclidean_tsp(n_cities=8)
    
    print(f"Generated TSP instance with {len(distance_matrix)} cities")
    
    # Create solver
    solver = TSPSolver(distance_matrix)
    
    # Test simulated annealing
    print("Testing simulated annealing...")
    sa_solution = solver.simulated_annealing(
        initial_temp=100.0,
        cooling_rate=0.9,
        min_temp=0.1,
        max_iterations=1000
    )
    
    print(f"SA Solution - Cost: {sa_solution.cost:.2f}, Time: {sa_solution.computation_time:.3f}s")
    
    # Test quantum annealing
    print("Testing quantum annealing...")
    qa_solution = solver.quantum_annealing(
        initial_temp=100.0,
        cooling_rate=0.9,
        min_temp=0.1,
        max_iterations=1000,
        quantum_factor=0.1,
        quantum_decay=0.99
    )
    
    print(f"QA Solution - Cost: {qa_solution.cost:.2f}, Time: {qa_solution.computation_time:.3f}s")
    
    # Test comparison method
    print("Testing comparison method...")
    solutions = solver.compare_methods(
        initial_temp=100.0,
        cooling_rate=0.9,
        min_temp=0.1,
        max_iterations=1000
    )
    
    print(f"Comparison completed - {len(solutions)} methods tested")
    
    # Test visualization (without showing plots)
    print("Testing visualization...")
    visualizer = TSPVisualizer(distance_matrix)
    
    # Test distance matrix plot
    visualizer.plot_distance_matrix(save_path='test_distance_matrix.png')
    print("Distance matrix plot saved")
    
    # Test route plotting
    visualizer.plot_route(sa_solution, save_path='test_sa_route.png')
    visualizer.plot_route(qa_solution, save_path='test_qa_route.png')
    print("Route plots saved")
    
    # Test summary report
    report = visualizer.create_summary_report(solutions, save_path='test_report.txt')
    print("Summary report generated")
    
    print("\nAll tests completed successfully!")
    return True

def test_data_generation():
    """Test data generation functionality."""
    print("\nTesting data generation...")
    
    generator = TSPDataGenerator(seed=42)
    
    # Test different instance types
    euclidean = generator.generate_euclidean_tsp(10)
    symmetric = generator.generate_symmetric_tsp(10)
    clustered = generator.generate_clustered_tsp(12, 3)
    grid = generator.generate_grid_tsp(4)
    
    print(f"Euclidean TSP: {euclidean.shape}")
    print(f"Symmetric TSP: {symmetric.shape}")
    print(f"Clustered TSP: {clustered.shape}")
    print(f"Grid TSP: {grid.shape}")
    
    # Test saving and loading
    generator.save_distance_matrix(euclidean, 'test_euclidean.npy')
    loaded = generator.load_distance_matrix('test_euclidean.npy')
    
    if np.array_equal(euclidean, loaded):
        print("Save/load functionality works correctly")
    else:
        print("ERROR: Save/load functionality failed")
        return False
    
    # Clean up test files
    os.remove('test_euclidean.npy')
    
    return True

def main():
    """Run all tests."""
    print("TSP Solver Test Suite")
    print("=" * 30)
    
    try:
        # Test basic functionality
        if not test_basic_functionality():
            print("Basic functionality test failed!")
            return False
        
        # Test data generation
        if not test_data_generation():
            print("Data generation test failed!")
            return False
        
        print("\n" + "=" * 30)
        print("ALL TESTS PASSED! ✅")
        print("The TSP solver is working correctly.")
        
        # Clean up test files
        test_files = ['test_distance_matrix.png', 'test_sa_route.png', 
                     'test_qa_route.png', 'test_report.txt']
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
        
        return True
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 