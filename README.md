# SA (Simulated Annealing) Simulation

This application demonstrates different approaches to solving the Traveling Salesman Problem (TSP) using Simulated Annealing (SA) and Machine Learning techniques.


<img width="1512" alt="Screenshot 2025-01-04 at 3 49 14 PM" src="https://github.com/user-attachments/assets/b9f7afb3-114b-4004-82ba-d68b0d77d49a" />

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
