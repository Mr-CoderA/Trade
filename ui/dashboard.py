"""Web dashboard using Dash for real-time monitoring and visualization."""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from main import TradingSystem
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__)


class TradingDashboard:
    """Web dashboard for trading system."""
    
    def __init__(self):
        """Initialize dashboard."""
        self.system = TradingSystem()
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            html.Div([
                html.H1("AED/CNY Intelligent Trading System", style={'textAlign': 'center', 'marginBottom': 30}),
                html.Hr()
            ], style={'backgroundColor': '#f0f0f0', 'padding': '20px'}),
            
            # Metrics Row
            html.Div([
                html.Div([
                    html.Div(id='current-price', style={'fontSize': 24, 'fontWeight': 'bold'}),
                    html.Div('Current Price', style={'color': 'gray'})
                ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'flex': 1, 'margin': '10px'}),
                
                html.Div([
                    html.Div(id='signal-display', style={'fontSize': 24, 'fontWeight': 'bold'}),
                    html.Div('Trading Signal', style={'color': 'gray'})
                ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'flex': 1, 'margin': '10px'}),
                
                html.Div([
                    html.Div(id='confidence-display', style={'fontSize': 24, 'fontWeight': 'bold'}),
                    html.Div('Model Confidence', style={'color': 'gray'})
                ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'flex': 1, 'margin': '10px'}),
                
                html.Div([
                    html.Div(id='account-balance', style={'fontSize': 24, 'fontWeight': 'bold'}),
                    html.Div('Account Balance', style={'color': 'gray'})
                ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'flex': 1, 'margin': '10px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap'}),
            
            # Charts Row
            html.Div([
                html.Div([
                    dcc.Graph(id='price-chart')
                ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
                
                html.Div([
                    dcc.Graph(id='signals-chart')
                ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            ]),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='equity-curve')
                ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
                
                html.Div([
                    dcc.Graph(id='performance-metrics')
                ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            ]),
            
            # Update interval
            dcc.Interval(
                id='interval-component',
                interval=config.get_nested('data.update_interval', 60) * 1000,
                n_intervals=0
            )
        ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('current-price', 'children'),
             Output('signal-display', 'children'),
             Output('confidence-display', 'children'),
             Output('account-balance', 'children'),
             Output('price-chart', 'figure'),
             Output('signals-chart', 'figure'),
             Output('equity-curve', 'figure'),
             Output('performance-metrics', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            """Update all dashboard elements."""
            try:
                # Get latest signal
                latest = self.system.get_latest_signal()
                signal_text = latest['signal'] if latest else 'LOADING'
                confidence = latest['confidence'] if latest else 0
                price = latest['close'] if latest else 0
                
                # Signal color
                if signal_text == 'BUY':
                    signal_color = 'green'
                elif signal_text == 'SELL':
                    signal_color = 'red'
                else:
                    signal_color = 'gray'
                
                # Price chart
                price_fig = self._create_price_chart()
                
                # Signals chart
                signals_fig = self._create_signals_chart()
                
                # Equity curve
                equity_fig = self._create_equity_chart()
                
                # Performance metrics
                metrics_fig = self._create_metrics_chart()
                
                return (
                    f"${price:.6f}",
                    html.Span(signal_text, style={'color': signal_color, 'fontWeight': 'bold'}),
                    f"{confidence:.2%}",
                    f"${self.system.risk_manager.current_balance:.2f}",
                    price_fig,
                    signals_fig,
                    equity_fig,
                    metrics_fig
                )
            except Exception as e:
                logger.error(f"Error updating dashboard: {e}")
                return "ERROR", "ERROR", "ERROR", "ERROR", {}, {}, {}, {}
    
    def _create_price_chart(self):
        """Create price chart."""
        if self.system.historical_data is None or len(self.system.historical_data) == 0:
            return go.Figure().add_annotation(text="No data available")
        
        data = self.system.historical_data.tail(100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title='Price Action (Last 100 Days)',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified'
        )
        
        return fig
    
    def _create_signals_chart(self):
        """Create signals distribution chart."""
        if self.system.latest_signals is None or len(self.system.latest_signals) == 0:
            return go.Figure().add_annotation(text="No signals available")
        
        signals = self.system.latest_signals['final_signal'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(x=signals.index, y=signals.values, marker_color=['green', 'red', 'gray'])
        ])
        
        fig.update_layout(
            title='Signal Distribution',
            xaxis_title='Signal Type',
            yaxis_title='Count'
        )
        
        return fig
    
    def _create_equity_chart(self):
        """Create equity curve chart."""
        backtest_metrics = self.system.performance_metrics.get('backtest', {})
        
        if not backtest_metrics:
            return go.Figure().add_annotation(text="No backtest data")
        
        equity = backtest_metrics.get('final_balance', 0)
        initial = 10000
        
        fig = go.Figure(data=[
            go.Scatter(
                y=[initial, equity],
                mode='lines+markers',
                name='Equity',
                line=dict(color='darkblue', width=3)
            )
        ])
        
        fig.update_layout(
            title='Account Equity Growth',
            xaxis_title='Time',
            yaxis_title='Balance ($)',
            hovermode='x unified'
        )
        
        return fig
    
    def _create_metrics_chart(self):
        """Create performance metrics chart."""
        backtest_metrics = self.system.performance_metrics.get('backtest', {})
        
        if not backtest_metrics:
            return go.Figure().add_annotation(text="No metrics available")
        
        metrics_text = f"""
        <b>Performance Metrics</b><br>
        Win Rate: {backtest_metrics.get('win_rate', 0):.2%}<br>
        Sharpe Ratio: {backtest_metrics.get('sharpe_ratio', 0):.2f}<br>
        Max Drawdown: {backtest_metrics.get('max_drawdown', 0):.2%}<br>
        Profit Factor: {backtest_metrics.get('profit_factor', 0):.2f}<br>
        Total P&L: ${backtest_metrics.get('total_pnl', 0):.2f}
        """
        
        fig = go.Figure()
        fig.add_annotation(text=metrics_text, showarrow=False, font=dict(size=12))
        fig.update_layout(title='Backtest Results', showlegend=False, xaxis_visible=False, yaxis_visible=False)
        
        return fig
    
    def run(self, debug: bool = True, port: int = 8050):
        """Run the dashboard.
        
        Args:
            debug: Debug mode
            port: Port to run on
        """
        logger.info(f"Starting dashboard on http://localhost:{port}")
        self.app.run_server(debug=debug, port=port)


if __name__ == '__main__':
    dashboard = TradingDashboard()
    dashboard.run()
