# scenario_analyzer.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

class ScenarioAnalyzer:
    def display_scenario_ui(self, base_forecast_df):
        """Static scenario comparison - NO INTERACTIONS, NO RESTART"""
        
        st.subheader("📊 Scenario Comparison Table")
        
        st.markdown("""
        Below is how different events would affect your forecast.
        **No interaction needed** - just see the possibilities.
        """)
        
        # Define scenarios
        scenarios = {
            "🚀 Very Optimistic": 1.50,
            "📈 Optimistic": 1.20,
            "➡️ Base Case": 1.00,
            "📉 Pessimistic": 0.80,
            "💔 Very Pessimistic": 0.60
        }
        
        # Create comparison table
        table_data = []
        for name, multiplier in scenarios.items():
            final_value = base_forecast_df['forecast'].iloc[-1] * multiplier
            change_percent = (multiplier - 1) * 100
            table_data.append({
                "Scenario": name,
                "Impact": f"{change_percent:+.0f}%",
                "Forecast Value": f"{final_value:.2f}"
            })
        
        comparison_df = pd.DataFrame(table_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Show chart with all scenarios
        fig = go.Figure()
        
        # Add base forecast
        fig.add_trace(go.Scatter(
            x=base_forecast_df['date'],
            y=base_forecast_df['forecast'],
            mode='lines',
            name='Base Forecast',
            line=dict(color='blue', width=3)
        ))
        
        # Add scenario lines
        colors = ['red', 'orange', 'gray', 'lightblue', 'darkred']
        for (name, multiplier), color in zip(scenarios.items(), colors):
            if multiplier != 1.00:  # Skip base case (already added)
                scenario_values = base_forecast_df['forecast'] * multiplier
                fig.add_trace(go.Scatter(
                    x=base_forecast_df['date'],
                    y=scenario_values,
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=1.5, dash='dot'),
                    opacity=0.7
                ))
        
        fig.update_layout(
            title="Multiple Scenario Forecasts",
            xaxis_title="Date",
            yaxis_title="Value",
            template='plotly_white',
            height=450,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Simple explanation
        st.info("""
        💡 **How to use this:**
        - **Optimistic scenario (+20%)** = Good news (earnings beat, product launch)
        - **Pessimistic scenario (-20%)** = Bad news (lawsuit, market crash)
        - **Very Optimistic (+50%)** = Extremely positive event
        - **Very Pessimistic (-40%)** = Major crisis
        """)