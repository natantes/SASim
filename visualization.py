import plotly.graph_objects as go
import numpy as np

def create_tour_figure(cities, tour=None, title="Cities", show_path=True, path_progress=None):
    """Create a plotly figure for the current state with step-by-step path visualization."""
    fig = go.Figure()
    
    # Add city points
    fig.add_trace(go.Scatter(
        x=[city[1] for city in cities],
        y=[city[2] for city in cities],
        mode='markers+text',
        name='Cities',
        text=[city[0] for city in cities],
        textposition="top center",
        marker=dict(size=10, color='red')
    ))
    
    # Add path segments one by one if tour and path_progress are provided
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

def create_progress_figure(distances, include_trend=True):
    """Create a plotly figure for the progress chart."""
    fig = go.Figure()
    
    # Add actual distances
    fig.add_trace(go.Scatter(
        y=distances,
        mode='lines',
        name='Best Distance',
        line=dict(color='blue')
    ))
    
    if include_trend and len(distances) > 1:
        # Add trend line
        x = list(range(len(distances)))
        y = distances
        coeffs = np.polyfit(x, y, 1)
        trend = np.poly1d(coeffs)
        
        fig.add_trace(go.Scatter(
            x=x,
            y=trend(x),
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash')
        ))
    
    fig.update_layout(
        title='Optimization Progress',
        xaxis_title='Improvement Steps',
        yaxis_title='Total Distance',
        showlegend=True
    )
    
    return fig