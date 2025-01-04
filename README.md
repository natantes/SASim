# Traveling Salesman Problem Solver

This application demonstrates different approaches to solving the Traveling Salesman Problem (TSP) using Simulated Annealing (SA) and Machine Learning techniques.

## Problem Overview

The Traveling Salesman Problem is a classic optimization challenge where the goal is to find the shortest possible route that:
- Visits each city exactly once
- Returns to the starting city
- Minimizes the total distance traveled

## Solution Approaches

### 1. Classic Simulated Annealing (SA)
The classic implementation uses pure SA with the following features:
- Random initial tour generation
- Temperature-based acceptance of worse solutions
- Gradually decreasing temperature to focus on improvements
- Multiple cooling schedules (Exponential, Linear, Quadratic, Logarithmic)

## Try it!

1. Clone the repository:
```bash
git clone [repository-url]
cd tsp-solver
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install streamlit numpy pandas scikit-learn plotly
```

## Running the Simulation

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Access the web interface (should open automatically) at:
```
http://localhost:8501
```

## Using the Application

1. Select Simulation Type:
   - "Classic SA": Traditional simulated annealing
   - "ML-Enhanced SA": Clustering-based approach

2. Adjust Parameters:
   - Basic Settings:
     - Number of Cities (5-100)
     - Steps Multiplier (affects runtime)
     - Update Interval (visualization frequency)
     - Path Segments per Update (animation smoothness)
   
   - Cooling Schedule:
     - Exponential: Smooth, fast cooling
     - Linear: Constant cooling rate
     - Quadratic: Slower at start, faster at end
     - Logarithmic: Very slow cooling
   
   - ML-Specific Settings (ML-Enhanced mode only):
     - Number of Clusters (2-10)

3. Start Optimization:
   - Click "Generate New Cities" to create a new problem
   - Click "Start Optimization" to begin solving
   - Watch real-time visualization of the tour improvement
   - Monitor metrics like best distance, temperature, and improvements

## Understanding the Visualization

### Main Plot
- Points represent cities
- Lines show the current tour
- In ML-Enhanced mode:
  - Colors indicate city clusters
  - X markers show cluster centers

### Progress Chart
- Shows best distance over time
- Downward trend indicates improvement
- Plateaus suggest local optima

### Metrics
- Best Distance: Length of shortest tour found
- Temperature: Current SA temperature
- Improvements: Number of better solutions found

## Files Structure

- `app.py`: Main Streamlit application
- `simulation_manager.py`: Core simulation logic
- `tsp_solver.py`: Base TSP solver implementation
- `visualization.py`: Plotting and visualization functions

## Tips for Best Results

1. For small problems (< 20 cities):
   - Use Classic SA
   - Higher temperature range
   - Faster cooling (Exponential)

2. For large problems (> 50 cities):
   - Use ML-Enhanced SA
   - More clusters (6-8)
   - Slower cooling (Logarithmic)
   - Higher steps multiplier

3. For balanced performance:
   - 20-50 cities
   - 3-5 clusters in ML mode
   - Quadratic cooling
   - Default steps multiplier
