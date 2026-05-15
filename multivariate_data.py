# multivariate_data.py
import pandas as pd
import numpy as np
import streamlit as st

class MultivariateDataFetcher:
    """Simple, working for ANY dataset"""
    
    def analyze_data(self, df):
        """Analyze ANY time series data"""
        
        if df is None or len(df) < 10:
            return ["⚠️ Need at least 10 data points for analysis"]
        
        insights = []
        values = df['value'].values
        
        # Overall change
        first_val = values[0]
        last_val = values[-1]
        total_change = ((last_val - first_val) / first_val) * 100
        
        if total_change > 20:
            insights.append(f"📈 **Strong growth**: +{total_change:.1f}% overall")
        elif total_change > 5:
            insights.append(f"📈 **Moderate growth**: +{total_change:.1f}% overall")
        elif total_change < -20:
            insights.append(f"📉 **Strong decline**: {total_change:.1f}% overall")
        elif total_change < -5:
            insights.append(f"📉 **Moderate decline**: {total_change:.1f}% overall")
        else:
            insights.append(f"➡️ **Stable**: {total_change:.1f}% overall change")
        
        # Volatility
        if len(values) > 10:
            daily_changes = np.diff(values) / values[:-1]
            volatility = np.std(daily_changes) * 100
            if volatility > 5:
                insights.append(f"⚡ **Very volatile**: {volatility:.1f}% average daily swings")
            elif volatility > 2:
                insights.append(f"⚠️ **Moderately volatile**: {volatility:.1f}% daily swings")
            else:
                insights.append(f"✅ **Stable**: {volatility:.1f}% daily swings")
        
        # Current position
        min_val = values.min()
        max_val = values.max()
        position = ((last_val - min_val) / (max_val - min_val)) * 100
        insights.append(f"📊 **Current position**: {position:.0f}% of historical range (min: {min_val:.2f}, max: {max_val:.2f})")
        
        # Recent trend
        if len(values) >= 14:
            recent_avg = np.mean(values[-7:])
            prev_avg = np.mean(values[-14:-7])
            recent_change = ((recent_avg - prev_avg) / prev_avg) * 100
            
            if recent_change > 5:
                insights.append(f"🚀 **Accelerating**: +{recent_change:.1f}% in last 7 days")
            elif recent_change > 1:
                insights.append(f"📈 **Rising**: +{recent_change:.1f}% in last 7 days")
            elif recent_change < -5:
                insights.append(f"📉 **Falling sharply**: {recent_change:.1f}% in last 7 days")
            elif recent_change < -1:
                insights.append(f"📉 **Falling**: {recent_change:.1f}% in last 7 days")
            else:
                insights.append(f"➡️ **Flat**: {recent_change:.1f}% in last 7 days")
        
        return insights
    
    def display_insights(self, insights):
        for insight in insights:
            st.info(insight)