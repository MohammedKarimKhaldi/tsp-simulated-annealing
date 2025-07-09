import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import seaborn as sns
from .tsp_solver import TSPSolution, AnnealingType

class TSPVisualizer:
    """
    Visualization tools for TSP solutions and results comparison.
    """
    
    def __init__(self, distance_matrix: np.ndarray):
        """
        Initialize the visualizer.
        
        Args:
            distance_matrix: Distance matrix for the TSP instance
        """
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_route(self, solution: TSPSolution, 
                   city_coordinates: Optional[np.ndarray] = None,
                   title: Optional[str] = None,
                   save_path: Optional[str] = None):
        """
        Plot the TSP route.
        
        Args:
            solution: TSP solution to visualize
            city_coordinates: Optional 2D coordinates for cities (if None, uses random layout)
            title: Plot title
            save_path: Path to save the plot
        """
        if city_coordinates is None:
            # Generate random coordinates for visualization
            np.random.seed(42)  # For reproducible layout
            city_coordinates = np.random.rand(self.n_cities, 2) * 100
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot cities
        ax.scatter(city_coordinates[:, 0], city_coordinates[:, 1], 
                  c='red', s=100, zorder=5, label='Cities')
        
        # Plot route
        route = solution.route
        for i in range(len(route)):
            current_city = route[i]
            next_city = route[(i + 1) % len(route)]
            
            x1, y1 = city_coordinates[current_city]
            x2, y2 = city_coordinates[next_city]
            
            ax.plot([x1, x2], [y1, y2], 'b-', alpha=0.7, linewidth=2)
        
        # Add city labels
        for i, (x, y) in enumerate(city_coordinates):
            ax.annotate(f'{i}', (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10)
        
        # Add start/end marker
        start_city = route[0]
        x, y = city_coordinates[start_city]
        ax.scatter(x, y, c='green', s=200, marker='s', zorder=6, label='Start/End')
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        if title is None:
            title = f'TSP Route - {solution.annealing_type.value.title()} Annealing\n'
            title += f'Cost: {solution.cost:.2f}, Time: {solution.computation_time:.3f}s'
        
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def plot_convergence_comparison(self, solutions: Dict[str, TSPSolution],
                                  save_path: Optional[str] = None):
        """
        Plot convergence comparison between different methods.
        
        Args:
            solutions: Dictionary of solutions from different methods
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cost comparison
        methods = list(solutions.keys())
        costs = [solutions[method].cost for method in methods]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:len(methods)]
        
        bars = ax1.bar(methods, costs, color=colors)
        ax1.set_ylabel('Total Cost')
        ax1.set_title('Cost Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add cost values on bars
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{cost:.2f}', ha='center', va='bottom')
        
        # Time comparison
        times = [solutions[method].computation_time for method in methods]
        
        bars = ax2.bar(methods, times, color=colors)
        ax2.set_ylabel('Computation Time (seconds)')
        ax2.set_title('Time Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add time values on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_parameter_sensitivity(self, solver, 
                                 parameter_name: str,
                                 parameter_values: List[float],
                                 annealing_type: AnnealingType = AnnealingType.SIMULATED,
                                 n_trials: int = 5,
                                 save_path: Optional[str] = None):
        """
        Plot sensitivity analysis for a parameter.
        
        Args:
            solver: TSP solver instance
            parameter_name: Name of the parameter to test
            parameter_values: List of parameter values to test
            annealing_type: Type of annealing to use
            n_trials: Number of trials per parameter value
            save_path: Path to save the plot
        """
        results = []
        
        for param_value in parameter_values:
            param_results = []
            for _ in range(n_trials):
                kwargs = {parameter_name: param_value}
                solution = solver.solve(annealing_type=annealing_type, **kwargs)
                param_results.append(solution.cost)
            results.append(param_results)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        ax1.boxplot(results, labels=[f'{v:.2f}' for v in parameter_values])
        ax1.set_xlabel(parameter_name)
        ax1.set_ylabel('Cost')
        ax1.set_title(f'{parameter_name} Sensitivity - Box Plot')
        ax1.grid(True, alpha=0.3)
        
        # Mean and std plot
        means = [np.mean(r) for r in results]
        stds = [np.std(r) for r in results]
        
        ax2.errorbar(parameter_values, means, yerr=stds, 
                    marker='o', capsize=5, capthick=2)
        ax2.set_xlabel(parameter_name)
        ax2.set_ylabel('Mean Cost')
        ax2.set_title(f'{parameter_name} Sensitivity - Mean ± Std')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_distance_matrix(self, save_path: Optional[str] = None):
        """
        Plot the distance matrix as a heatmap.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(self.distance_matrix, 
                   annot=True, 
                   fmt='.1f', 
                   cmap='viridis',
                   cbar_kws={'label': 'Distance'})
        
        plt.title('TSP Distance Matrix')
        plt.xlabel('City Index')
        plt.ylabel('City Index')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def plot_multiple_routes(self, solutions: Dict[str, TSPSolution],
                           city_coordinates: Optional[np.ndarray] = None,
                           save_path: Optional[str] = None):
        """
        Plot multiple routes on the same graph for comparison.
        
        Args:
            solutions: Dictionary of solutions to compare
            city_coordinates: Optional 2D coordinates for cities
            save_path: Path to save the plot
        """
        if city_coordinates is None:
            np.random.seed(42)
            city_coordinates = np.random.rand(self.n_cities, 2) * 100
        
        n_methods = len(solutions)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
        
        if n_methods == 1:
            axes = [axes]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:n_methods]
        
        for i, (method_name, solution) in enumerate(solutions.items()):
            ax = axes[i]
            
            # Plot cities
            ax.scatter(city_coordinates[:, 0], city_coordinates[:, 1], 
                      c='red', s=50, zorder=5)
            
            # Plot route
            route = solution.route
            for j in range(len(route)):
                current_city = route[j]
                next_city = route[(j + 1) % len(route)]
                
                x1, y1 = city_coordinates[current_city]
                x2, y2 = city_coordinates[next_city]
                
                ax.plot([x1, x2], [y1, y2], color=colors[i], alpha=0.7, linewidth=2)
            
            # Add start/end marker
            start_city = route[0]
            x, y = city_coordinates[start_city]
            ax.scatter(x, y, c='green', s=100, marker='s', zorder=6)
            
            ax.set_title(f'{method_name}\nCost: {solution.cost:.2f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_summary_report(self, solutions: Dict[str, TSPSolution],
                            save_path: Optional[str] = None) -> str:
        """
        Create a text summary report of the results.
        
        Args:
            solutions: Dictionary of solutions
            save_path: Path to save the report
            
        Returns:
            Report text
        """
        report = "TSP SOLVER RESULTS SUMMARY\n"
        report += "=" * 50 + "\n\n"
        
        # Find best solution
        best_method = min(solutions.keys(), 
                         key=lambda x: solutions[x].cost)
        best_solution = solutions[best_method]
        
        report += f"BEST SOLUTION: {best_method.upper()}\n"
        report += f"Cost: {best_solution.cost:.2f}\n"
        report += f"Route: {best_solution.route}\n"
        report += f"Computation Time: {best_solution.computation_time:.3f} seconds\n"
        report += f"Iterations: {best_solution.iterations}\n\n"
        
        report += "DETAILED COMPARISON:\n"
        report += "-" * 30 + "\n"
        
        for method, solution in solutions.items():
            report += f"\n{method.upper()}:\n"
            report += f"  Cost: {solution.cost:.2f}\n"
            report += f"  Time: {solution.computation_time:.3f}s\n"
            report += f"  Iterations: {solution.iterations}\n"
            report += f"  Route: {solution.route}\n"
        
        report += f"\nPERFORMANCE ANALYSIS:\n"
        report += "-" * 30 + "\n"
        
        costs = [s.cost for s in solutions.values()]
        times = [s.computation_time for s in solutions.values()]
        
        report += f"Cost Range: {min(costs):.2f} - {max(costs):.2f}\n"
        report += f"Cost Improvement: {((max(costs) - min(costs)) / max(costs) * 100):.1f}%\n"
        report += f"Time Range: {min(times):.3f}s - {max(times):.3f}s\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report 