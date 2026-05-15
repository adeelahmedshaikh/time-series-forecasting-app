# minute_forecast.py
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

class MinuteForecaster:
    def __init__(self):
        self.frequencies = {
            "1 minute": 1,
            "5 minutes": 5,
            "10 minutes": 10,
            "15 minutes": 15,
            "30 minutes": 30,
            "1 hour": 60
        }
    
    def resample_to_minutes(self, df, frequency_minutes):
        """Convert daily data to minute frequency (simulated)"""
        
        # Create minute-level timestamps
        last_date = df['date'].max()
        start_date = last_date - timedelta(days=7)
        
        # Generate minute timestamps
        freq_minutes = self.frequencies.get(frequency_minutes, 5)
        periods = int(7 * 24 * 60 / freq_minutes)
        
        minute_timestamps = pd.date_range(
            start=start_date,
            periods=periods,
            freq=f'{freq_minutes}T'
        )
        
        # Simulate intraday patterns
        simulated_values = []
        for ts in minute_timestamps:
            # Add intraday pattern (higher during trading hours)
            hour = ts.hour
            minute = ts.minute
            
            # Business hours effect (9 AM - 4 PM)
            if 9 <= hour <= 16:
                intraday_factor = 1 + (np.sin((hour - 9) * np.pi / 14) * 0.1)
            else:
                intraday_factor = 0.95
            
            # Random noise
            noise = np.random.normal(0, 0.01)
            
            # Base value from historical pattern
            base_value = df['value'].iloc[-1] * intraday_factor * (1 + noise)
            simulated_values.append(base_value)
        
        result_df = pd.DataFrame({
            'date': minute_timestamps,
            'value': simulated_values
        })
        
        return result_df
    
    def forecast_minutes(self, df, frequency, minutes_ahead):
        """Generate minute-level forecast"""
        
        # Resample to chosen frequency
        minute_df = self.resample_to_minutes(df, frequency)
        
        # Simple forecasting: trend + intraday pattern
        last_values = minute_df['value'].tail(60).values
        trend = np.mean(np.diff(last_values))
        
        forecast_minutes = []
        forecast_values = []
        
        last_date = minute_df['date'].max()
        last_value = minute_df['value'].iloc[-1]
        
        freq_minutes = self.frequencies.get(frequency, 5)
        
        for i in range(minutes_ahead):
            next_date = last_date + timedelta(minutes=freq_minutes * (i + 1))
            next_value = last_value + trend * (i + 1)
            
            # Add intraday pattern
            hour = next_date.hour
            if 9 <= hour <= 16:
                intraday_factor = 1 + (np.sin((hour - 9) * np.pi / 14) * 0.05)
            else:
                intraday_factor = 0.98
            
            next_value = next_value * intraday_factor
            
            forecast_minutes.append(next_date)
            forecast_values.append(next_value)
        
        forecast_df = pd.DataFrame({
            'date': forecast_minutes,
            'forecast': forecast_values
        })
        
        return forecast_df, minute_df
    
    def display_minute_ui(self, df):
        """UI for minute-level forecasting"""
        
        st.subheader("⏱️ Real-Time Minute Forecast")
        
        col1, col2 = st.columns(2)
        
        with col1:
            frequency = st.selectbox(
                "Time interval",
                list(self.frequencies.keys()),
                key="minute_freq"
            )
        
        with col2:
            minutes_ahead = st.selectbox(
                "Forecast ahead",
                [30, 60, 120, 240, 480],
                format_func=lambda x: f"{x} minutes ({x//60} hours)" if x >= 60 else f"{x} minutes"
            )
        
        if st.button("Generate Minute Forecast", key="minute_btn"):
            with st.spinner("Generating minute-level forecast..."):
                forecast_df, historical_df = self.forecast_minutes(df, frequency, minutes_ahead)
            
            # Show metrics
            st.success(f"✅ Generated {len(forecast_df)} minute-level forecasts")
            
            # Display forecast table
            st.write("#### 📊 Minute Forecast")
            display_df = forecast_df.copy()
            display_df['date'] = display_df['date'].dt.strftime('%H:%M:%S')
            st.dataframe(display_df.head(20))
            
            # Chart
            import plotly.graph_objects as go
            fig = go.Figure()
            
            # Historical (last 60 minutes)
            hist_last = historical_df.tail(60)
            fig.add_trace(go.Scatter(
                x=hist_last['date'],
                y=hist_last['value'],
                mode='lines',
                name='Historical (last 60 periods)',
                line=dict(color='blue', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['forecast'],
                mode='lines',
                name=f'Forecast (next {minutes_ahead} min)',
                line=dict(color='red', width=2, dash='dot')
            ))
            
            fig.update_layout(
                title=f"Minute Forecast - {frequency} intervals",
                xaxis_title="Time",
                yaxis_title="Value",
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Download option
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Minute Forecast CSV",
                data=csv,
                file_name=f"minute_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
            return forecast_df
        
        return None