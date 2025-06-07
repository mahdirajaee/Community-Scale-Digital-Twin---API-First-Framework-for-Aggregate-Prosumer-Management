"""
Comprehensive Streamlit Dashboard for Community-Scale Digital Twin Framework
Real-time monitoring, forecasting visualization, optimization controls, and scenario analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
import asyncio
from typing import Dict, List, Any, Optional

# Configure Streamlit page
st.set_page_config(
    page_title="Digital Twin Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class DigitalTwinDashboard:
    """Main dashboard class for the Digital Twin Framework"""
    
    def __init__(self):
        self.api_base_url = st.secrets.get("API_BASE_URL", "http://localhost:8000")
        self.session_state = st.session_state
        
        # Initialize session state
        if 'selected_community' not in self.session_state:
            self.session_state.selected_community = "residential_suburb"
        if 'auto_refresh' not in self.session_state:
            self.session_state.auto_refresh = False
        if 'refresh_interval' not in self.session_state:
            self.session_state.refresh_interval = 30
    
    def make_api_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Optional[Dict]:
        """Make API request with error handling"""
        try:
            url = f"{self.api_base_url}{endpoint}"
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=30)
            else:
                st.error(f"Unsupported HTTP method: {method}")
                return None
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"API connection error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return None
    
    def render_sidebar(self):
        """Render sidebar with navigation and controls"""
        st.sidebar.title("🏘️ Digital Twin Control")
        
        # Community selection
        communities = self.get_communities()
        if communities:
            community_options = {comm['name']: comm['id'] for comm in communities}
            selected_name = st.sidebar.selectbox(
                "Select Community",
                options=list(community_options.keys()),
                index=0
            )
            self.session_state.selected_community = community_options[selected_name]
        
        st.sidebar.markdown("---")
        
        # Navigation
        st.sidebar.subheader("📊 Navigation")
        page = st.sidebar.radio(
            "Select Page",
            ["Real-time Monitoring", "Forecasting", "Optimization", "Scenario Analysis", "System Health"]
        )
        
        st.sidebar.markdown("---")
        
        # Auto-refresh controls
        st.sidebar.subheader("🔄 Auto Refresh")
        self.session_state.auto_refresh = st.sidebar.checkbox("Enable Auto Refresh")
        if self.session_state.auto_refresh:
            self.session_state.refresh_interval = st.sidebar.slider(
                "Refresh Interval (seconds)", 
                min_value=10, 
                max_value=300, 
                value=30
            )
        
        return page
    
    def get_communities(self) -> List[Dict]:
        """Get list of available communities"""
        response = self.make_api_request("/communities")
        if response and response.get('success'):
            return response.get('data', [])
        return []
    
    def get_current_state(self, community_id: str) -> Optional[Dict]:
        """Get current community state"""
        response = self.make_api_request(f"/communities/{community_id}/current-state")
        if response and response.get('success'):
            return response.get('data')
        return None
    
    def get_historical_data(self, community_id: str, hours: int = 24) -> Optional[pd.DataFrame]:
        """Get historical data for the community"""
        response = self.make_api_request(
            f"/communities/{community_id}/historical-data?hours={hours}"
        )
        if response and response.get('success'):
            data = response.get('data', [])
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        return None
    
    def render_real_time_monitoring(self):
        """Render real-time monitoring page"""
        st.markdown('<h1 class="main-header">⚡ Real-time Community Monitoring</h1>', 
                   unsafe_allow_html=True)
        
        community_id = self.session_state.selected_community
        
        # Get current state
        current_state = self.get_current_state(community_id)
        
        if current_state:
            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Net Load",
                    value=f"{current_state.get('net_power_kw', 0):.1f} kW",
                    delta=f"{current_state.get('net_power_change', 0):.1f} kW"
                )
            
            with col2:
                st.metric(
                    label="PV Generation",
                    value=f"{current_state.get('solar_generation_kw', 0):.1f} kW",
                    delta=f"{current_state.get('generation_change', 0):.1f} kW"
                )
            
            with col3:
                st.metric(
                    label="Grid Exchange",
                    value=f"{current_state.get('grid_exchange_kw', 0):.1f} kW",
                    delta=f"{current_state.get('grid_change', 0):.1f} kW"
                )
            
            with col4:
                st.metric(
                    label="Battery SOC",
                    value=f"{current_state.get('battery_soc_percent', 0):.1f}%",
                    delta=f"{current_state.get('soc_change', 0):.1f}%"
                )
            
            # Real-time charts
            st.subheader("📈 Real-time Energy Flows")
            
            # Get historical data for plotting
            df = self.get_historical_data(community_id, hours=24)
            
            if df is not None and not df.empty:
                # Create subplots
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Energy Generation & Consumption', 'Grid Exchange', 
                                  'Battery State of Charge', 'Energy Prices'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Generation and consumption
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df['pv_generation_kw'], 
                              name='PV Generation', line=dict(color='orange')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df['net_load_kw'], 
                              name='Net Load', line=dict(color='blue')),
                    row=1, col=1
                )
                
                # Grid exchange
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df['grid_import_kw'], 
                              name='Grid Import', line=dict(color='red')),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=-df['grid_export_kw'], 
                              name='Grid Export', line=dict(color='green')),
                    row=1, col=2
                )
                
                # Battery SOC
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df['storage_soc_kwh']/10, 
                              name='Battery SOC (%)', line=dict(color='purple')),
                    row=2, col=1
                )
                
                # Energy prices (if available)
                if 'energy_price' in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df['timestamp'], y=df['energy_price'], 
                                  name='Energy Price', line=dict(color='black')),
                        row=2, col=2
                    )
                
                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
            # System status indicators
            st.subheader("🔧 System Status")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                voltage = current_state.get('voltage_pu', 1.0)
                voltage_status = "healthy" if 0.95 <= voltage <= 1.05 else "warning"
                st.markdown(
                    f"**Grid Voltage:** <span class='status-{voltage_status}'>{voltage:.3f} pu</span>",
                    unsafe_allow_html=True
                )
            
            with col2:
                frequency = current_state.get('frequency_hz', 50.0)
                freq_status = "healthy" if 49.8 <= frequency <= 50.2 else "warning"
                st.markdown(
                    f"**Frequency:** <span class='status-{freq_status}'>{frequency:.2f} Hz</span>",
                    unsafe_allow_html=True
                )
            
            with col3:
                active_prosumers = current_state.get('active_prosumers', 0)
                st.markdown(f"**Active Prosumers:** {active_prosumers}")
        
        else:
            st.error("Failed to retrieve current community state")
    
    def render_forecasting_page(self):
        """Render forecasting page"""
        st.markdown('<h1 class="main-header">🔮 Energy Forecasting</h1>', 
                   unsafe_allow_html=True)
        
        community_id = self.session_state.selected_community
        
        # Forecasting controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_horizon = st.selectbox(
                "Forecast Horizon",
                options=[6, 12, 24, 48, 168],
                format_func=lambda x: f"{x} hours" if x < 168 else "1 week",
                index=2
            )
        
        with col2:
            forecast_model = st.selectbox(
                "Model Type",
                options=["ensemble", "lstm", "prophet"],
                format_func=lambda x: x.upper()
            )
        
        with col3:
            confidence_intervals = st.checkbox("Show Confidence Intervals", value=True)
        
        # Generate forecast button
        if st.button("🔄 Generate Forecast", type="primary"):
            with st.spinner("Generating forecast..."):
                forecast_data = {
                    "community_id": community_id,
                    "forecast_horizon_hours": forecast_horizon,
                    "forecast_model": forecast_model,
                    "confidence_intervals": confidence_intervals
                }
                
                response = self.make_api_request("/forecast", method="POST", data=forecast_data)
                
                if response and response.get('success'):
                    forecast_result = response.get('data')
                    
                    # Display forecast results
                    st.success(f"Forecast generated successfully using {forecast_model.upper()} model")
                    
                    # Create forecast DataFrame
                    forecast_df = pd.DataFrame({
                        'timestamp': pd.to_datetime(forecast_result['timestamps']),
                        'consumption_forecast': forecast_result['consumption_forecast'],
                        'generation_forecast': forecast_result['generation_forecast'],
                        'net_demand_forecast': forecast_result['net_demand_forecast']
                    })
                    
                    if confidence_intervals and 'consumption_lower_bound' in forecast_result:
                        forecast_df['consumption_lower'] = forecast_result['consumption_lower_bound']
                        forecast_df['consumption_upper'] = forecast_result['consumption_upper_bound']
                        forecast_df['generation_lower'] = forecast_result['generation_lower_bound']
                        forecast_df['generation_upper'] = forecast_result['generation_upper_bound']
                    
                    # Plot forecast
                    fig = go.Figure()
                    
                    # Consumption forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_df['timestamp'],
                        y=forecast_df['consumption_forecast'],
                        name='Consumption Forecast',
                        line=dict(color='blue')
                    ))
                    
                    # Generation forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_df['timestamp'],
                        y=forecast_df['generation_forecast'],
                        name='Generation Forecast',
                        line=dict(color='orange')
                    ))
                    
                    # Net demand forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_df['timestamp'],
                        y=forecast_df['net_demand_forecast'],
                        name='Net Demand Forecast',
                        line=dict(color='purple')
                    ))
                    
                    # Add confidence intervals if available
                    if confidence_intervals and 'consumption_lower' in forecast_df.columns:
                        fig.add_trace(go.Scatter(
                            x=forecast_df['timestamp'],
                            y=forecast_df['consumption_upper'],
                            fill=None,
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=forecast_df['timestamp'],
                            y=forecast_df['consumption_lower'],
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            name='Consumption CI',
                            fillcolor='rgba(0,100,200,0.2)'
                        ))
                    
                    fig.update_layout(
                        title=f"Energy Forecast - {forecast_horizon} Hours Ahead",
                        xaxis_title="Time",
                        yaxis_title="Power (kW)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast accuracy metrics
                    if 'forecast_accuracy' in forecast_result:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Forecast Accuracy", 
                                    f"{forecast_result['forecast_accuracy']*100:.1f}%")
                        with col2:
                            if 'mape' in forecast_result:
                                st.metric("MAPE", f"{forecast_result['mape']:.2f}%")
                        with col3:
                            if 'rmse' in forecast_result:
                                st.metric("RMSE", f"{forecast_result['rmse']:.2f} kW")
                
                else:
                    st.error("Failed to generate forecast")
    
    def render_optimization_page(self):
        """Render optimization page"""
        st.markdown('<h1 class="main-header">⚙️ Energy Optimization</h1>', 
                   unsafe_allow_html=True)
        
        community_id = self.session_state.selected_community
        
        # Optimization configuration
        st.subheader("🎯 Optimization Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            objective = st.selectbox(
                "Optimization Objective",
                options=["welfare_maximization", "cost_minimization", "peak_minimization", "self_consumption"],
                format_func=lambda x: x.replace("_", " ").title()
            )
            
            time_horizon = st.selectbox(
                "Time Horizon",
                options=[6, 12, 24, 48],
                format_func=lambda x: f"{x} hours",
                index=2
            )
        
        with col2:
            import_price = st.number_input("Import Price ($/kWh)", value=0.25, step=0.01)
            export_price = st.number_input("Export Price ($/kWh)", value=0.15, step=0.01)
        
        # Advanced constraints
        with st.expander("⚙️ Advanced Constraints"):
            col1, col2 = st.columns(2)
            
            with col1:
                max_grid_import = st.number_input("Max Grid Import (kW)", value=500.0, step=10.0)
                min_battery_soc = st.slider("Min Battery SOC (%)", 0, 50, 20)
            
            with col2:
                max_grid_export = st.number_input("Max Grid Export (kW)", value=400.0, step=10.0)
                max_battery_soc = st.slider("Max Battery SOC (%)", 50, 100, 90)
        
        # Run optimization
        if st.button("🚀 Run Optimization", type="primary"):
            with st.spinner("Running optimization..."):
                optimization_data = {
                    "community_id": community_id,
                    "objective": objective,
                    "time_horizon_hours": time_horizon,
                    "import_price_per_kwh": import_price,
                    "export_price_per_kwh": export_price,
                    "max_grid_import_kw": max_grid_import,
                    "max_grid_export_kw": max_grid_export,
                    "min_battery_soc": min_battery_soc,
                    "max_battery_soc": max_battery_soc
                }
                
                response = self.make_api_request("/optimize", method="POST", data=optimization_data)
                
                if response and response.get('success'):
                    optimization_result = response.get('data')
                    
                    if optimization_result.get('feasible'):
                        st.success(f"Optimization completed successfully in {optimization_result.get('solve_time_seconds', 0):.2f} seconds")
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame({
                            'timestamp': pd.to_datetime(optimization_result['timestamps']),
                            'optimal_generation': optimization_result['optimal_generation'],
                            'optimal_consumption': optimization_result['optimal_consumption'],
                            'optimal_battery': optimization_result['optimal_battery_schedule'],
                            'optimal_grid_exchange': optimization_result['optimal_grid_exchange']
                        })
                        
                        # Plot optimization results
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('Optimal Energy Schedule', 'Battery Schedule', 
                                          'Grid Exchange', 'Cost Breakdown')
                        )
                        
                        # Energy schedule
                        fig.add_trace(
                            go.Scatter(x=results_df['timestamp'], y=results_df['optimal_generation'],
                                      name='Generation', line=dict(color='orange')),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=results_df['timestamp'], y=results_df['optimal_consumption'],
                                      name='Consumption', line=dict(color='blue')),
                            row=1, col=1
                        )
                        
                        # Battery schedule
                        fig.add_trace(
                            go.Scatter(x=results_df['timestamp'], y=results_df['optimal_battery'],
                                      name='Battery Power', line=dict(color='purple')),
                            row=1, col=2
                        )
                        
                        # Grid exchange
                        fig.add_trace(
                            go.Scatter(x=results_df['timestamp'], y=results_df['optimal_grid_exchange'],
                                      name='Grid Exchange', line=dict(color='red')),
                            row=2, col=1
                        )
                        
                        # Cost breakdown (pie chart)
                        costs = [
                            optimization_result.get('grid_cost', 0),
                            optimization_result.get('battery_degradation_cost', 0),
                            -optimization_result.get('demand_response_reward', 0)
                        ]
                        labels = ['Grid Cost', 'Battery Degradation', 'DR Reward']
                        
                        fig.add_trace(
                            go.Pie(values=costs, labels=labels, name="Cost Breakdown"),
                            row=2, col=2
                        )
                        
                        fig.update_layout(height=600, showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Cost", f"${optimization_result.get('total_cost', 0):.2f}")
                        with col2:
                            st.metric("Objective Value", f"{optimization_result.get('objective_value', 0):.2f}")
                        with col3:
                            if 'peak_reduction_percent' in optimization_result:
                                st.metric("Peak Reduction", f"{optimization_result['peak_reduction_percent']:.1f}%")
                        with col4:
                            if 'self_consumption_ratio' in optimization_result:
                                st.metric("Self-Consumption", f"{optimization_result['self_consumption_ratio']*100:.1f}%")
                    
                    else:
                        st.error("Optimization problem is infeasible with current constraints")
                
                else:
                    st.error("Failed to run optimization")
    
    def render_scenario_analysis_page(self):
        """Render scenario analysis page"""
        st.markdown('<h1 class="main-header">🎭 Scenario Analysis</h1>', 
                   unsafe_allow_html=True)
        
        community_id = self.session_state.selected_community
        
        # Scenario configuration
        st.subheader("📋 Scenario Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scenario_name = st.text_input("Scenario Name", value="Policy Impact Analysis")
            simulation_hours = st.selectbox(
                "Simulation Period",
                options=[168, 720, 8760],
                format_func=lambda x: f"{x//24} days" if x < 720 else ("1 month" if x == 720 else "1 year"),
                index=0
            )
        
        with col2:
            monte_carlo_runs = st.slider("Monte Carlo Runs", 10, 500, 100)
            weather_uncertainty = st.checkbox("Include Weather Uncertainty", value=True)
        
        # Technology penetration scenarios
        st.subheader("🔧 Technology Scenarios")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            solar_penetration = st.slider("Solar Penetration (%)", 0, 100, 50)
        with col2:
            battery_penetration = st.slider("Battery Penetration (%)", 0, 100, 30)
        with col3:
            ev_penetration = st.slider("EV Penetration (%)", 0, 100, 20)
        
        # Tariff structure
        st.subheader("💰 Tariff Structure")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            peak_rate = st.number_input("Peak Rate ($/kWh)", value=0.35, step=0.01)
        with col2:
            shoulder_rate = st.number_input("Shoulder Rate ($/kWh)", value=0.25, step=0.01)
        with col3:
            off_peak_rate = st.number_input("Off-Peak Rate ($/kWh)", value=0.15, step=0.01)
        
        # Run scenario analysis
        if st.button("🎬 Run Scenario Analysis", type="primary"):
            with st.spinner("Running scenario simulation..."):
                scenario_data = {
                    "community_id": community_id,
                    "scenario_name": scenario_name,
                    "simulation_hours": simulation_hours,
                    "monte_carlo_runs": monte_carlo_runs,
                    "solar_penetration_percent": solar_penetration,
                    "battery_penetration_percent": battery_penetration,
                    "ev_penetration_percent": ev_penetration,
                    "tariff_structure": {
                        "peak": peak_rate,
                        "shoulder": shoulder_rate,
                        "off_peak": off_peak_rate
                    },
                    "weather_uncertainty": weather_uncertainty
                }
                
                response = self.make_api_request("/scenario/simulate", method="POST", data=scenario_data)
                
                if response and response.get('success'):
                    scenario_result = response.get('data')
                    
                    st.success("Scenario analysis completed successfully")
                    
                    # Results summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Mean Community Cost",
                            f"${scenario_result.get('mean_community_cost', 0):.2f}",
                            delta=f"±${scenario_result.get('std_community_cost', 0):.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Grid Impact",
                            f"{scenario_result.get('mean_grid_impact', 0):.2f} kW"
                        )
                    
                    with col3:
                        st.metric(
                            "Self-Sufficiency",
                            f"{scenario_result.get('mean_self_sufficiency', 0)*100:.1f}%"
                        )
                    
                    with col4:
                        st.metric(
                            "Carbon Reduction",
                            f"{scenario_result.get('carbon_reduction_tons', 0):.1f} tons"
                        )
                    
                    # Distribution plots
                    if 'cost_percentiles' in scenario_result:
                        percentiles = scenario_result['cost_percentiles']
                        
                        fig = go.Figure()
                        
                        # Box plot for cost distribution
                        fig.add_trace(go.Box(
                            y=[percentiles.get('P5', 0), percentiles.get('P25', 0), 
                               percentiles.get('P50', 0), percentiles.get('P75', 0), 
                               percentiles.get('P95', 0)],
                            name="Cost Distribution",
                            boxpoints='all'
                        ))
                        
                        fig.update_layout(
                            title="Cost Distribution Analysis",
                            yaxis_title="Cost ($)",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error("Failed to run scenario analysis")
    
    def render_system_health_page(self):
        """Render system health monitoring page"""
        st.markdown('<h1 class="main-header">🔧 System Health Monitoring</h1>', 
                   unsafe_allow_html=True)
        
        # Get system health data
        health_response = self.make_api_request("/health")
        
        if health_response and health_response.get('success'):
            health_data = health_response.get('data', {})
            
            # Service status overview
            st.subheader("🏥 Service Status")
            
            services = ['api_gateway', 'forecasting_service', 'optimization_service', 'database', 'redis']
            
            cols = st.columns(len(services))
            
            for i, service in enumerate(services):
                with cols[i]:
                    service_health = health_data.get(service, {})
                    is_healthy = service_health.get('is_healthy', False)
                    
                    status_class = "healthy" if is_healthy else "error"
                    status_text = "🟢 Healthy" if is_healthy else "🔴 Error"
                    
                    st.markdown(f"**{service.replace('_', ' ').title()}**")
                    st.markdown(f"<span class='status-{status_class}'>{status_text}</span>", 
                               unsafe_allow_html=True)
                    
                    if 'response_time_ms' in service_health:
                        st.text(f"Response: {service_health['response_time_ms']:.2f}ms")
            
            # Performance metrics
            st.subheader("📊 Performance Metrics")
            
            # Create performance charts
            if 'performance_history' in health_data:
                perf_data = health_data['performance_history']
                df_perf = pd.DataFrame(perf_data)
                df_perf['timestamp'] = pd.to_datetime(df_perf['timestamp'])
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Response Time', 'CPU Usage', 'Memory Usage', 'Throughput')
                )
                
                # Response time
                fig.add_trace(
                    go.Scatter(x=df_perf['timestamp'], y=df_perf['response_time_ms'],
                              name='Response Time', line=dict(color='blue')),
                    row=1, col=1
                )
                
                # CPU usage
                fig.add_trace(
                    go.Scatter(x=df_perf['timestamp'], y=df_perf['cpu_usage_percent'],
                              name='CPU Usage', line=dict(color='red')),
                    row=1, col=2
                )
                
                # Memory usage
                fig.add_trace(
                    go.Scatter(x=df_perf['timestamp'], y=df_perf['memory_usage_percent'],
                              name='Memory Usage', line=dict(color='green')),
                    row=2, col=1
                )
                
                # Throughput
                fig.add_trace(
                    go.Scatter(x=df_perf['timestamp'], y=df_perf['throughput_requests_per_second'],
                              name='Throughput', line=dict(color='purple')),
                    row=2, col=2
                )
                
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("Failed to retrieve system health data")
    
    def run(self):
        """Main dashboard run method"""
        # Header
        st.markdown(
            '<h1 class="main-header">⚡ Community Digital Twin Dashboard</h1>',
            unsafe_allow_html=True
        )
        
        # Sidebar navigation
        page = self.render_sidebar()
        
        # Auto-refresh logic
        if self.session_state.auto_refresh:
            time.sleep(self.session_state.refresh_interval)
            st.experimental_rerun()
        
        # Render selected page
        if page == "Real-time Monitoring":
            self.render_real_time_monitoring()
        elif page == "Forecasting":
            self.render_forecasting_page()
        elif page == "Optimization":
            self.render_optimization_page()
        elif page == "Scenario Analysis":
            self.render_scenario_analysis_page()
        elif page == "System Health":
            self.render_system_health_page()


# Main execution
if __name__ == "__main__":
    dashboard = DigitalTwinDashboard()
    dashboard.run()
