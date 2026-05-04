# charts.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

def create_interactive_forecast(actual_df, forecast_df, title="Forecast Visualization"):
    """Create an interactive chart with actual and forecast data"""
    
    fig = go.Figure()
    
    # Actual data
    fig.add_trace(go.Scatter(
        x=actual_df['date'],
        y=actual_df['value'],
        mode='lines',
        name='Actual',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
    ))
    
    # Forecast data
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dot'),
        marker=dict(size=6, color='#ff7f0e'),
        hovertemplate='Date: %{x}<br>Forecast: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500
    )
    
    return fig

def create_model_comparison_chart(results_df):
    """Create bar chart comparing model performance"""
    
    # Sort by MAE for better visualization
    results_df = results_df.sort_values('MAE')
    
    fig = go.Figure(data=[
        go.Bar(
            name='MAE',
            x=results_df.index,
            y=results_df['MAE'],
            marker_color='#3498db',
            text=results_df['MAE'].round(2),
            textposition='outside',
            textfont=dict(size=10)
        )
    ])
    
    fig.update_layout(
        title="Model Performance Comparison (Lower is Better)",
        xaxis_title="Model",
        yaxis_title="Mean Absolute Error (MAE)",
        template='plotly_white',
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig

def create_components_chart(df):
    """Show trend, seasonal, and residual components"""
    
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Set date as index for decomposition
        df_decomp = df.set_index('date').copy()
        
        # Need at least 14 points for decomposition
        if len(df_decomp) < 14:
            return None
        
        # Decompose time series
        decomposition = seasonal_decompose(
            df_decomp['value'], 
            model='additive', 
            period=min(7, len(df_decomp)//2)
        )
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Original
        fig.add_trace(
            go.Scatter(x=df['date'], y=decomposition.observed, mode='lines', name='Original'),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(x=df['date'], y=decomposition.trend, mode='lines', name='Trend', line=dict(color='red')),
            row=2, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(x=df['date'], y=decomposition.seasonal, mode='lines', name='Seasonal', line=dict(color='green')),
            row=3, col=1
        )
        
        # Residual
        fig.add_trace(
            go.Scatter(x=df['date'], y=decomposition.resid, mode='lines', name='Residual', line=dict(color='orange')),
            row=4, col=1
        )
        
        fig.update_layout(
            height=800, 
            title_text="Time Series Decomposition",
            showlegend=False
        )
        fig.update_xaxes(title_text="Date", row=4, col=1)
        
        return fig
        
    except Exception as e:
        print(f"Decomposition failed: {e}")
        return None

def create_forecast_accuracy_chart(actual_values, predicted_values, dates):
    """Create a scatter plot comparing actual vs predicted"""
    
    fig = go.Figure()
    
    # Perfect prediction line
    min_val = min(min(actual_values), min(predicted_values))
    max_val = max(max(actual_values), max(predicted_values))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='gray', dash='dash')
    ))
    
    # Actual vs Predicted points
    fig.add_trace(go.Scatter(
        x=actual_values,
        y=predicted_values,
        mode='markers',
        name='Predictions',
        marker=dict(size=10, color='#3498db', opacity=0.6),
        text=dates,
        hovertemplate='Date: %{text}<br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Actual vs Predicted Values",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        template='plotly_white',
        height=500
    )
    
    return fig