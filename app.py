import streamlit as st
import numpy as np
import time
from tsp_solver import TSPSolver, generate_cities
from visualization import create_tour_figure, create_progress_figure

st.set_page_config(page_title="Simulated Annealing TSP Solver", layout="wide")

st.title("Simulated Annealing for the Traveling Salesman Problem")

st.markdown("""
This application demonstrates the Simulated Annealing algorithm solving the Traveling Salesman Problem (TSP). 
The TSP involves finding the shortest possible route that visits each city exactly once and returns to the starting city.

**How it works:**
1. Cities are randomly placed on a 2D grid
2. The algorithm starts with a random tour and gradually improves it
3. At high temperatures, worse solutions may be accepted to escape local minima
4. As temperature cools, the algorithm becomes more selective, focusing on improvements
""")

def main():
    # Parameters in sidebar
    st.sidebar.header("Algorithm Parameters")
    
    st.sidebar.subheader("Basic Settings")
    num_cities = st.sidebar.slider("Number of Cities", 5, 100, 20)
    n_multiplier = st.sidebar.slider("Steps Multiplier", 10, 200, 50)
    update_interval = st.sidebar.slider(
        "Update Interval",
        1, 20, 5,
        help="Number of improvements between display updates"
    )

    
    st.sidebar.subheader("Cooling Schedule")
    cooling_scheme = st.sidebar.selectbox(
        "Cooling Scheme",
        ["Exponential", "Linear", "Quadratic", "Logarithmic"],
        help="""
        Different ways to reduce temperature over time:
        - Exponential: Smooth, fast cooling (T = T0 * rate^step)
        - Linear: Constant cooling rate (T = T0 * (1 - step/total))
        - Quadratic: Slower at start, faster at end
        - Logarithmic: Very slow cooling, good for complex problems
        """
    )
    
    st.sidebar.subheader("Temperature Settings")
    Tmax = st.sidebar.number_input(
        "Initial Temperature", 
        value=np.exp(8.7), 
        format="%.2f",
        help="Higher values allow more 'bad' moves early on"
    )
    Tmin = st.sidebar.number_input(
        "Final Temperature", 
        value=np.exp(-6.5), 
        format="%.2f",
        help="Lower values make the algorithm more 'picky' at the end"
    )
    
    # Generate random cities
    if 'cities' not in st.session_state or st.sidebar.button("Generate New Cities"):
        st.session_state.cities = generate_cities(num_cities)
        st.session_state.solver = TSPSolver(st.session_state.cities, cooling_scheme)
    
    # Create columns for visualizations
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Create placeholder for the tour visualization
        tour_container = st.empty()
        # Show initial cities without path
        tour_container.plotly_chart(
            create_tour_figure(
                st.session_state.cities,
                None,
                "Initial Cities",
                show_path=False
            ),
            use_container_width=True
        )
    
    with col2:
        # Create placeholder for the progress chart
        progress_container = st.empty()
        metrics_container = st.empty()
        
        # Initialize empty progress chart
        progress_container.plotly_chart(
            create_progress_figure([]),
            use_container_width=True
        )
        
        # Initialize empty metrics
        col1, col2, col3 = metrics_container.columns(3)
        col1.metric("Best Distance", "N/A")
        col2.metric("Temperature", f"{Tmax:.2e}")
        col3.metric("Improvements", 0)
    
    # Start optimization button
    if st.button("Start Optimization"):
        progress_bar = st.progress(0)
        solver = st.session_state.solver
        solver.initialize()
        
        steps = n_multiplier * num_cities**2
        update_count = 0
        
        start_time = time.time()
        
        # Main optimization loop
        for step in range(steps):
            improved, T = solver.step(step, steps, Tmax, Tmin)
            
            if improved:
                update_count += 1
                state = solver.get_current_state()
                
                # Update visualization every update_interval improvements
                if update_count % update_interval == 0:
                    # Animate path construction
                    for segment in range(0, len(state['best_tour']), 1):
                        tour_container.plotly_chart(
                            create_tour_figure(
                                st.session_state.cities,
                                state['best_tour'],
                                f"Current Tour (Step {step+1}/{steps})",
                                path_progress=segment
                            ),
                            use_container_width=True
                        )
                    
                    progress_container.plotly_chart(
                        create_progress_figure(state['distances_history']),
                        use_container_width=True
                    )
                    
                    # Update metrics
                    col1, col2, col3 = metrics_container.columns(3)
                    col1.metric("Best Distance", f"{state['best_distance']:.2f}")
                    col2.metric("Temperature", f"{T:.2e}")
                    col3.metric("Improvements", state['improvements'])
            
            progress_bar.progress((step + 1) / steps)
        
        end_time = time.time()
        
        # Final update
        state = solver.get_current_state()
        tour_container.plotly_chart(
            create_tour_figure(
                st.session_state.cities,
                state['best_tour'],
                f"Final Tour (Distance: {state['best_distance']:.2f})",
                path_progress=len(state['best_tour']) - 1
            ),
            use_container_width=True
        )
        
        st.success(f"Optimization completed in {end_time - start_time:.2f} seconds!")

if __name__ == "__main__":
    main()