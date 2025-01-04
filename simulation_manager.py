import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tsp_solver import TSPSolver, generate_cities
import plotly.graph_objects as go

class MLTSPSolver(TSPSolver):
    """ML-enhanced variant of the TSP Solver that uses clustering for optimization"""
    def __init__(self, cities, cooling_scheme="Exponential"):
        super().__init__(cities, cooling_scheme)
        self.cities_df = pd.DataFrame(cities, columns=['name', 'x', 'y'])
        self.cluster_assignments = None
        self.cluster_centers = None
        
    def initialize_with_clusters(self, n_clusters=3):
        """Initialize tour using K-means clustering"""
        scaler = StandardScaler()
        coords = self.cities_df[['x', 'y']].values
        coords_scaled = scaler.fit_transform(coords)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_assignments = kmeans.fit_predict(coords_scaled)
        self.cluster_centers = kmeans.cluster_centers_
        
        tour = []
        for cluster in range(n_clusters):
            cluster_cities = np.where(self.cluster_assignments == cluster)[0]
            tour.extend(cluster_cities)
        
        tour.append(tour[0])
        
        self.current_tour = np.array(tour)
        self.best_tour = self.current_tour.copy()
        self.best_distance = self._energy_tour(self.current_tour)
        self.distances_history = [self.best_distance]
        self.improvements = 0

    def get_cluster_info(self):
        """Get clustering information for visualization"""
        return {
            'assignments': self.cluster_assignments,
            'centers': self.cluster_centers
        }

def create_clustered_tour_figure(cities, cluster_info, tour=None, title="Clustered Cities", show_path=True, path_progress=None):
    """Create a plotly figure that shows clustering and tour"""
    fig = go.Figure()
    
    # Add city points colored by cluster
    for cluster in range(len(np.unique(cluster_info['assignments']))):
        mask = cluster_info['assignments'] == cluster
        cluster_cities = [cities[i] for i in range(len(cities)) if mask[i]]
        
        fig.add_trace(go.Scatter(
            x=[city[1] for city in cluster_cities],
            y=[city[2] for city in cluster_cities],
            mode='markers+text',
            name=f'Cluster {cluster + 1}',
            text=[city[0] for city in cluster_cities],
            textposition="top center",
            marker=dict(size=10)
        ))
    
    # Add cluster centers
    if cluster_info['centers'] is not None:
        centers_transformed = StandardScaler().fit_transform(
            np.array([[city[1], city[2]] for city in cities])
        )
        centers = StandardScaler().inverse_transform(cluster_info['centers'])
        
        fig.add_trace(go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode='markers',
            name='Cluster Centers',
            marker=dict(
                symbol='x',
                size=15,
                color='black',
                line=dict(width=2)
            )
        ))
    
    # Add path segments
    if tour is not None and show_path and path_progress is not None:
        for i in range(min(len(tour)-1, path_progress + 1)):
            start_city = cities[tour[i]]
            end_city = cities[tour[i+1]]
            
            fig.add_trace(go.Scatter(
                x=[start_city[1], end_city[1]],
                y=[start_city[2], end_city[2]],
                mode='lines',
                name=f'Path {i+1}',
                line=dict(
                    color=f'rgba(0, 0, 255, {0.5 if i < path_progress else 1.0})',
                    width=2
                ),
                showlegend=False
            ))
    
    fig.update_layout(
        title=title,
        showlegend=True,
        width=800,
        height=800,
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate"
    )
    
    return fig

class SimulationManager:
    """Manages different types of TSP simulations"""
    def __init__(self):
        self.cities = None
        self.solver = None
        self.simulation_type = None
    
    def initialize_simulation(self, simulation_type, num_cities, cooling_scheme="Exponential"):
        """Initialize a new simulation with given parameters"""
        self.simulation_type = simulation_type
        self.cities = generate_cities(num_cities)
        
        if simulation_type == "ML-Enhanced SA":
            self.solver = MLTSPSolver(self.cities, cooling_scheme)
        else:
            self.solver = TSPSolver(self.cities, cooling_scheme)
    
    def initialize_solver(self, n_clusters=None):
        """Initialize the solver with appropriate method"""
        if self.simulation_type == "ML-Enhanced SA" and n_clusters:
            self.solver.initialize_with_clusters(n_clusters)
        else:
            self.solver.initialize()
    
    def get_initial_visualization(self, show_path=False):
        """Get the initial visualization based on simulation type"""
        if self.simulation_type == "ML-Enhanced SA":
            cluster_info = self.solver.get_cluster_info()
            return create_clustered_tour_figure(
                self.cities,
                cluster_info,
                None,
                "Initial Clustered Cities",
                show_path=show_path
            )
        else:
            return create_tour_figure(
                self.cities,
                None,
                "Initial Cities",
                show_path=show_path
            )
    
    def update_visualization(self, step, total_steps, path_progress):
        """Update visualization based on current state"""
        state = self.solver.get_current_state()
        title = f"Current Tour (Step {step+1}/{total_steps})"
        
        if self.simulation_type == "ML-Enhanced SA":
            cluster_info = self.solver.get_cluster_info()
            return create_clustered_tour_figure(
                self.cities,
                cluster_info,
                state['best_tour'],
                title,
                path_progress=path_progress
            )
        else:
            return create_tour_figure(
                self.cities,
                state['best_tour'],
                title,
                path_progress=path_progress
            )
    
    def step(self, step_num, total_steps, initial_temp, final_temp):
        """Perform one step of the simulation"""
        return self.solver.step(step_num, total_steps, initial_temp, final_temp)
    
    def get_current_state(self):
        """Get current state of the simulation"""
        return self.solver.get_current_state()