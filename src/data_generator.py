import numpy as np
import random
from typing import List, Tuple, Optional
import json
import os

class TSPDataGenerator:
    """
    Generate TSP test instances with various characteristics.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_euclidean_tsp(self, n_cities: int, 
                              x_range: Tuple[float, float] = (0, 100),
                              y_range: Tuple[float, float] = (0, 100)) -> np.ndarray:
        """
        Generate TSP instance with cities in 2D Euclidean space.
        
        Args:
            n_cities: Number of cities
            x_range: Range for x coordinates
            y_range: Range for y coordinates
            
        Returns:
            Distance matrix
        """
        # Generate random city coordinates
        x_coords = np.random.uniform(x_range[0], x_range[1], n_cities)
        y_coords = np.random.uniform(y_range[0], y_range[1], n_cities)
        
        # Calculate distance matrix
        distance_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    dx = x_coords[i] - x_coords[j]
                    dy = y_coords[i] - y_coords[j]
                    distance_matrix[i][j] = np.sqrt(dx*dx + dy*dy)
        
        return distance_matrix
    
    def generate_symmetric_tsp(self, n_cities: int, 
                              min_distance: float = 1.0,
                              max_distance: float = 100.0) -> np.ndarray:
        """
        Generate symmetric TSP instance with random distances.
        
        Args:
            n_cities: Number of cities
            min_distance: Minimum distance between cities
            max_distance: Maximum distance between cities
            
        Returns:
            Distance matrix
        """
        distance_matrix = np.zeros((n_cities, n_cities))
        
        for i in range(n_cities):
            for j in range(i+1, n_cities):
                distance = np.random.uniform(min_distance, max_distance)
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance
        
        return distance_matrix
    
    def generate_clustered_tsp(self, n_cities: int, 
                              n_clusters: int = 3,
                              cluster_radius: float = 10.0,
                              inter_cluster_distance: float = 50.0) -> np.ndarray:
        """
        Generate TSP instance with cities clustered in groups.
        
        Args:
            n_cities: Number of cities
            n_clusters: Number of clusters
            cluster_radius: Radius of each cluster
            inter_cluster_distance: Distance between cluster centers
            
        Returns:
            Distance matrix
        """
        # Generate cluster centers
        cluster_centers = []
        for i in range(n_clusters):
            x = (i % 3) * inter_cluster_distance
            y = (i // 3) * inter_cluster_distance
            cluster_centers.append((x, y))
        
        # Generate cities within clusters
        cities_per_cluster = n_cities // n_clusters
        remaining_cities = n_cities % n_clusters
        
        x_coords = []
        y_coords = []
        
        for i, center in enumerate(cluster_centers):
            n_cities_in_cluster = cities_per_cluster + (1 if i < remaining_cities else 0)
            
            for _ in range(n_cities_in_cluster):
                # Add some randomness to cluster center
                x = center[0] + np.random.normal(0, cluster_radius)
                y = center[1] + np.random.normal(0, cluster_radius)
                x_coords.append(x)
                y_coords.append(y)
        
        # Calculate distance matrix
        distance_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    dx = x_coords[i] - x_coords[j]
                    dy = y_coords[i] - y_coords[j]
                    distance_matrix[i][j] = np.sqrt(dx*dx + dy*dy)
        
        return distance_matrix
    
    def generate_grid_tsp(self, grid_size: int) -> np.ndarray:
        """
        Generate TSP instance with cities arranged in a grid.
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size cities)
            
        Returns:
            Distance matrix
        """
        n_cities = grid_size * grid_size
        distance_matrix = np.zeros((n_cities, n_cities))
        
        # Generate grid coordinates
        x_coords = []
        y_coords = []
        for i in range(grid_size):
            for j in range(grid_size):
                x_coords.append(i * 10.0)  # 10 units apart
                y_coords.append(j * 10.0)
        
        # Calculate distance matrix
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    dx = x_coords[i] - x_coords[j]
                    dy = y_coords[i] - y_coords[j]
                    distance_matrix[i][j] = np.sqrt(dx*dx + dy*dy)
        
        return distance_matrix
    
    def save_distance_matrix(self, distance_matrix: np.ndarray, filename: str):
        """Save distance matrix to a file."""
        np.save(filename, distance_matrix)
    
    def load_distance_matrix(self, filename: str) -> np.ndarray:
        """Load distance matrix from a file."""
        return np.load(filename)
    
    def save_metadata(self, metadata: dict, filename: str):
        """Save metadata about the generated instance."""
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def generate_benchmark_suite(self, output_dir: str = "data"):
        """
        Generate a comprehensive benchmark suite of TSP instances.
        
        Args:
            output_dir: Directory to save the instances
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Small instances (for quick testing)
        small_instances = [
            ("small_euclidean_10", self.generate_euclidean_tsp(10)),
            ("small_symmetric_10", self.generate_symmetric_tsp(10)),
            ("small_clustered_12", self.generate_clustered_tsp(12, 3)),
            ("small_grid_4", self.generate_grid_tsp(4))
        ]
        
        # Medium instances
        medium_instances = [
            ("medium_euclidean_25", self.generate_euclidean_tsp(25)),
            ("medium_symmetric_25", self.generate_symmetric_tsp(25)),
            ("medium_clustered_30", self.generate_clustered_tsp(30, 5)),
            ("medium_grid_5", self.generate_grid_tsp(5))
        ]
        
        # Large instances
        large_instances = [
            ("large_euclidean_50", self.generate_euclidean_tsp(50)),
            ("large_symmetric_50", self.generate_symmetric_tsp(50)),
            ("large_clustered_60", self.generate_clustered_tsp(60, 6)),
            ("large_grid_7", self.generate_grid_tsp(7))
        ]
        
        all_instances = small_instances + medium_instances + large_instances
        
        for name, distance_matrix in all_instances:
            # Save distance matrix
            matrix_file = os.path.join(output_dir, f"{name}.npy")
            self.save_distance_matrix(distance_matrix, matrix_file)
            
            # Save metadata
            metadata = {
                "name": name,
                "n_cities": len(distance_matrix),
                "type": name.split('_')[1],
                "min_distance": np.min(distance_matrix[distance_matrix > 0]),
                "max_distance": np.max(distance_matrix),
                "mean_distance": np.mean(distance_matrix[distance_matrix > 0])
            }
            
            metadata_file = os.path.join(output_dir, f"{name}_metadata.json")
            self.save_metadata(metadata, metadata_file)
        
        print(f"Generated {len(all_instances)} TSP instances in {output_dir}/")
        return all_instances 