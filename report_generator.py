# report_generator.py
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
from datetime import datetime

def generate_forecast_report(df, forecast_df, results, best_model, forecast_change, target_type, target_col):
    """Generate a professional PDF report"""
    
    # Create buffer for PDF
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30
    )
    
    # Title
    story.append(Paragraph("Time Series Forecasting Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Report metadata
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    summary_text = f"""
    <b>Target Variable:</b> {target_col}<br/>
    <b>Forecast Type:</b> {target_type}<br/>
    <b>Best Model:</b> {best_model}<br/>
    <b>Forecast Change:</b> {forecast_change:.2f}%<br/>
    <b>Data Points Analyzed:</b> {len(df)}<br/>
    <b>Forecast Horizon:</b> {len(forecast_df)} days
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Trend interpretation
    if forecast_change > 2:
        trend = "📈 UPWARD - Values expected to increase"
    elif forecast_change < -2:
        trend = "📉 DOWNWARD - Values expected to decrease"
    else:
        trend = "➡️ STABLE - Values expected to remain steady"
    
    story.append(Paragraph(f"<b>Trend:</b> {trend}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Model Performance Table
    story.append(Paragraph("Model Performance Comparison", styles['Heading2']))
    
    table_data = [['Model', 'MAE', 'MSE']]
    for model, metrics in results.items():
        table_data.append([model, f"{metrics['MAE']:.2f}", f"{metrics['MSE']:.2f}"])
    
    # Limit to top 10 models if too many
    if len(table_data) > 11:
        table_data = table_data[:11]
    
    table = Table(table_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3*inch))
    
    # Forecast Results Table (Top 10)
    story.append(Paragraph("Forecast Results (Next Days)", styles['Heading2']))
    
    forecast_data = [['Date', 'Forecast Value']]
    for i, row in forecast_df.head(10).iterrows():
        forecast_data.append([row['date'].strftime('%Y-%m-%d'), f"{row['forecast']:.2f}"])
    
    forecast_table = Table(forecast_data, colWidths=[2*inch, 2*inch])
    forecast_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(forecast_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Recommendations
    story.append(Paragraph("Recommendations", styles['Heading2']))
    
    if target_type == "Sales / Demand":
        if forecast_change > 2:
            rec = "Demand is expected to rise. Consider increasing inventory and preparing for higher volume."
        elif forecast_change < -2:
            rec = "Demand is expected to decline. Avoid overstocking and consider promotional activities."
        else:
            rec = "Demand appears stable. Maintain current inventory strategy."
    elif target_type == "Stock / Asset Price":
        rec = "These forecasts are for informational purposes only. Consult a financial advisor before making investment decisions."
    else:
        rec = f"Based on the {forecast_change:.2f}% forecast change, monitor your {target_type} metrics closely over the coming period."
    
    story.append(Paragraph(rec, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#666666'),
        alignment=1
    )
    story.append(Paragraph("This report is automatically generated. Forecasts are estimates and may not reflect actual outcomes.", disclaimer_style))
    
    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    
    return pdf_buffer