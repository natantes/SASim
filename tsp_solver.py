import numpy as np
import random
from itertools import combinations

class TSPSolver:
    def __init__(self, cities, cooling_scheme="Exponential"):
        self.cities = cities
        self.distance_matrix = self._calculate_distance_matrix()
        self.cooling_scheme = cooling_scheme
        self.current_tour = None
        self.best_tour = None
        self.best_distance = float('inf')
        self.distances_history = []
        self.improvements = 0
        
    def _calculate_distance_matrix(self):
        """Calculate distance matrix from city coordinates."""
        n = len(self.cities)
        distance_matrix = np.zeros((n, n))
        
        for i, j in combinations(range(n), 2):
            dist = np.sqrt((self.cities[i][1] - self.cities[j][1])**2 + 
                         (self.cities[i][2] - self.cities[j][2])**2)
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
        
        return distance_matrix
    
    def _calculate_temperature(self, initial_temp, step, total_steps, final_temp):
        """Calculate temperature based on different cooling schemes."""
        if self.cooling_scheme == "Linear":
            return initial_temp * (1 - step/total_steps)
        elif self.cooling_scheme == "Exponential":
            return initial_temp * ((final_temp / initial_temp) ** (step / total_steps))
        elif self.cooling_scheme == "Quadratic":
            return initial_temp * (1 - (step/total_steps)**2)
        else:  # Logarithmic
            return initial_temp / (1 + np.log(1 + step))
    
    def _energy_tour(self, tour):
        """Calculate total distance of a tour."""
        return np.sum(self.distance_matrix[tour[:-1], tour[1:]])
    
    def _metropolis_filter(self, tour, proposal, temperature):
        """Decide whether to accept a proposed tour based on the Metropolis criterion."""
        current_energy = self._energy_tour(tour)
        proposed_energy = self._energy_tour(proposal)
        
        if proposed_energy < current_energy:
            return True, proposed_energy
        else:
            acceptance_probability = np.exp((current_energy - proposed_energy) / temperature)
            return random.random() < acceptance_probability, proposed_energy
    
    def initialize(self):
        """Initialize a random tour."""
        n = len(self.cities)
        self.current_tour = np.arange(n)
        np.random.shuffle(self.current_tour[1:])
        self.current_tour = np.append(self.current_tour, 0)
        
        self.best_tour = self.current_tour.copy()
        self.best_distance = self._energy_tour(self.current_tour)
        self.distances_history = [self.best_distance]
        self.improvements = 0
    
    def step(self, step_num, total_steps, initial_temp, final_temp):
        """Perform one step of the simulated annealing algorithm."""
        n = len(self.cities)
        
        # Select two random cities to swap
        i, j = sorted(random.sample(range(1, n), 2))
        
        # Create new proposal by reversing a segment of the tour
        proposal_tour = np.concatenate([
            self.current_tour[:i],
            self.current_tour[i:j+1][::-1],
            self.current_tour[j+1:]
        ])
        
        # Update temperature according to cooling scheme
        T = self._calculate_temperature(initial_temp, step_num, total_steps, final_temp)
        
        # Check if we should accept the new tour
        accepted, current_distance = self._metropolis_filter(
            self.current_tour, proposal_tour, T
        )
        
        improved = False
        if accepted:
            self.current_tour = proposal_tour
            if current_distance < self.best_distance:
                self.best_tour = proposal_tour.copy()
                self.best_distance = current_distance
                self.distances_history.append(current_distance)
                self.improvements += 1
                improved = True
        
        return improved, T
    
    def get_current_state(self):
        """Get the current state of the solver."""
        return {
            'best_tour': self.best_tour,
            'best_distance': self.best_distance,
            'distances_history': self.distances_history,
            'improvements': self.improvements
        }

def generate_cities(n, max_coord=1000):
    """Generate n random cities with their coordinates."""
    cities = []
    city_coords = set()
    
    while len(cities) < n:
        x = random.randint(0, max_coord)
        y = random.randint(0, max_coord)
        
        if (x, y) not in city_coords:
            cities.append((f"City_{len(cities)}", x, y))
            city_coords.add((x, y))
    
    return cities