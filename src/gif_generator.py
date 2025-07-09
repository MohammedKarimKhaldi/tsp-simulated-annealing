import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from matplotlib.figure import Figure
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os
from .tsp_solver import TSPSolution, AnnealingType

class TSPGifGenerator:
    """
    Generate animated GIFs showing the evolution of TSP solutions during annealing.
    """
    
    def __init__(self, distance_matrix: np.ndarray):
        """
        Initialize the GIF generator.
        
        Args:
            distance_matrix: Distance matrix for the TSP instance
        """
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        
        # Set style
        plt.style.use('default')
    
    def generate_city_coordinates(self, route: List[int]) -> np.ndarray:
        """
        Generate 2D coordinates for cities based on route order.
        This creates a more visually appealing layout.
        """
        # Create coordinates in a circle for better visualization
        angles = np.linspace(0, 2 * np.pi, self.n_cities, endpoint=False)
        radius = 50
        x_coords = radius * np.cos(angles)
        y_coords = radius * np.sin(angles)
        
        # Add some randomness for more natural look
        np.random.seed(42)  # For reproducibility
        x_coords += np.random.normal(0, 5, self.n_cities)
        y_coords += np.random.normal(0, 5, self.n_cities)
        
        return np.column_stack([x_coords, y_coords])
    
    def create_route_frame(self, route: List[int], cost: float, iteration: int, 
                          title: str, city_coordinates: np.ndarray) -> Figure:
        """
        Create a single frame for the GIF showing the current route.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot cities
        ax.scatter(city_coordinates[:, 0], city_coordinates[:, 1], 
                  c='red', s=200, zorder=5, label='Cities')
        
        # Plot route
        for i in range(len(route)):
            current_city = route[i]
            next_city = route[(i + 1) % len(route)]
            
            x1, y1 = city_coordinates[current_city]
            x2, y2 = city_coordinates[next_city]
            
            # Create arrow for direction
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                   arrowstyle='->', mutation_scale=20,
                                   color='blue', alpha=0.7, linewidth=2)
            ax.add_patch(arrow)
        
        # Add city labels
        for i, (x, y) in enumerate(city_coordinates):
            ax.annotate(f'{i}', (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=12, fontweight='bold')
        
        # Add start/end marker
        start_city = route[0]
        x, y = city_coordinates[start_city]
        ax.scatter(x, y, c='green', s=300, marker='s', zorder=6, label='Start/End')
        
        # Add title and info
        ax.set_title(f'{title}\nCost: {cost:.2f} | Iteration: {iteration}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        return fig
    
    def create_evolution_gif(self, solution: TSPSolution, 
                           output_path: str,
                           fps: int = 2,
                           duration_per_frame: float = 0.5,
                           loop: bool = False) -> str:
        """
        Create a GIF showing the evolution of a TSP solution.
        
        Args:
            solution: TSP solution with evolution history
            output_path: Path to save the GIF
            fps: Frames per second
            duration_per_frame: Duration to show each frame (seconds)
            loop: Whether the GIF should loop (False for no loop)
            
        Returns:
            Path to the created GIF
        """
        if not solution.evolution_history:
            raise ValueError("Solution must have evolution history enabled")
        
        print(f"Creating evolution GIF for {solution.annealing_type.value} annealing...")
        
        # Generate city coordinates
        city_coords = self.generate_city_coordinates(solution.route)
        
        # Create frames
        frames = []
        for iteration, cost, route in solution.evolution_history:
            fig = self.create_route_frame(
                route, cost, iteration,
                f"{solution.annealing_type.value.title()} Annealing",
                city_coords
            )
            
            # Convert figure to image
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            import imageio
            img = imageio.imread(buf)
            frames.append(img)
            buf.close()
            
            plt.close(fig)
        
        # Create GIF using imageio
        try:
            import imageio
        except ImportError:
            print("imageio not found. Installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "imageio"])
            import imageio
        
        # Save as GIF with loop control
        if loop:
            imageio.mimsave(output_path, frames, fps=fps, duration=duration_per_frame)
        else:
            # For no loop, we need to use a different approach
            # Add a pause at the end to show the final result
            final_frame = frames[-1]
            for _ in range(int(fps * 2)):  # Show final frame for 2 seconds
                frames.append(final_frame)
            imageio.mimsave(output_path, frames, fps=fps, duration=duration_per_frame)
        
        print(f"GIF saved to: {output_path}")
        return output_path
    
    def create_evolution_video(self, solution: TSPSolution, 
                             output_path: str,
                             fps: int = 5) -> str:
        """
        Create an MP4 video showing the evolution of a TSP solution.
        
        Args:
            solution: TSP solution with evolution history
            output_path: Path to save the video
            fps: Frames per second
            
        Returns:
            Path to the created video
        """
        if not solution.evolution_history:
            raise ValueError("Solution must have evolution history enabled")
        
        print(f"Creating evolution video for {solution.annealing_type.value} annealing...")
        
        # Generate city coordinates
        city_coords = self.generate_city_coordinates(solution.route)
        
        # Create frames
        frames = []
        for iteration, cost, route in solution.evolution_history:
            fig = self.create_route_frame(
                route, cost, iteration,
                f"{solution.annealing_type.value.title()} Annealing",
                city_coords
            )
            
            # Convert figure to image
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            import imageio
            img = imageio.imread(buf)
            frames.append(img)
            buf.close()
            
            plt.close(fig)
        
        # Create video using imageio
        try:
            import imageio
        except ImportError:
            print("imageio not found. Installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "imageio"])
            import imageio
        
        # Save as MP4 video (no loop by default)
        imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
        
        print(f"Video saved to: {output_path}")
        return output_path
    
    def create_comparison_gif(self, solutions: Dict[str, TSPSolution],
                            output_path: str,
                            fps: int = 2,
                            duration_per_frame: float = 0.5) -> str:
        """
        Create a side-by-side comparison GIF of multiple solutions.
        
        Args:
            solutions: Dictionary of solutions with evolution history
            output_path: Path to save the GIF
            fps: Frames per second
            duration_per_frame: Duration to show each frame (seconds)
            
        Returns:
            Path to the created GIF
        """
        # Check that all solutions have evolution history
        for name, solution in solutions.items():
            if not solution.evolution_history:
                raise ValueError(f"Solution '{name}' must have evolution history enabled")
        
        print("Creating comparison GIF...")
        
        # Generate city coordinates (use the same for all solutions)
        first_solution = next(iter(solutions.values()))
        city_coords = self.generate_city_coordinates(first_solution.route)
        
        # Find the maximum number of frames
        max_frames = max(len(sol.evolution_history) for sol in solutions.values() if sol.evolution_history is not None)
        
        # Create frames
        frames = []
        for frame_idx in range(max_frames):
            fig, axes = plt.subplots(1, len(solutions), figsize=(6*len(solutions), 6))
            
            if len(solutions) == 1:
                axes = [axes]
            
            for i, (name, solution) in enumerate(solutions.items()):
                ax = axes[i]
                
                # Get frame data (or use last frame if not enough)
                if solution.evolution_history is not None:
                    if frame_idx < len(solution.evolution_history):
                        iteration, cost, route = solution.evolution_history[frame_idx]
                    else:
                        iteration, cost, route = solution.evolution_history[-1]
                else:
                    # Fallback if no evolution history
                    iteration, cost, route = 0, solution.cost, solution.route
                
                # Plot cities
                ax.scatter(city_coords[:, 0], city_coords[:, 1], 
                          c='red', s=100, zorder=5)
                
                # Plot route
                for j in range(len(route)):
                    current_city = route[j]
                    next_city = route[(j + 1) % len(route)]
                    
                    x1, y1 = city_coords[current_city]
                    x2, y2 = city_coords[next_city]
                    
                    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                           arrowstyle='->', mutation_scale=15,
                                           color='blue', alpha=0.7, linewidth=1.5)
                    ax.add_patch(arrow)
                
                # Add start/end marker
                start_city = route[0]
                x, y = city_coords[start_city]
                ax.scatter(x, y, c='green', s=200, marker='s', zorder=6)
                
                # Add title
                ax.set_title(f'{name}\nCost: {cost:.2f} | Iter: {iteration}', 
                           fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
            
            # Convert figure to image
            fig.canvas.draw()
            img_data = fig.canvas.buffer_rgba()
            img = np.asarray(img_data)
            img = img[:, :, :3]  # Remove alpha channel
            frames.append(img)
            
            plt.close(fig)
        
        # Save as GIF
        try:
            import imageio
        except ImportError:
            print("imageio not found. Installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "imageio"])
            import imageio
        
        imageio.mimsave(output_path, frames, fps=fps, duration=duration_per_frame)
        
        print(f"Comparison GIF saved to: {output_path}")
        return output_path
    
    def create_cost_evolution_plot(self, solutions: Dict[str, TSPSolution],
                                 output_path: str) -> str:
        """
        Create a plot showing cost evolution over iterations.
        
        Args:
            solutions: Dictionary of solutions with evolution history
            output_path: Path to save the plot
            
        Returns:
            Path to the created plot
        """
        plt.figure(figsize=(12, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (name, solution) in enumerate(solutions.items()):
            if solution.evolution_history:
                iterations = [frame[0] for frame in solution.evolution_history]
                costs = [frame[1] for frame in solution.evolution_history]
                
                plt.plot(iterations, costs, 'o-', 
                        label=name, color=colors[i % len(colors)],
                        linewidth=2, markersize=4, alpha=0.8)
        
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost Evolution During Annealing')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale to see improvements better
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cost evolution plot saved to: {output_path}")
        return output_path 