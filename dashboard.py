import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import numpy as np

st.set_page_config(
    page_title="Community Digital Twin Dashboard", 
    page_icon="⚡", 
    layout="wide"
)

API_BASE_URL = "http://localhost:8000"

def call_api(endpoint, method="GET", data=None):
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

st.title("⚡ Community Digital Twin Dashboard")
st.markdown("Real-time monitoring and control of prosumer communities")

with st.sidebar:
    st.header("🏘️ Community Selection")
    
    communities_response = call_api("communities")
    if communities_response and communities_response.get("success"):
        communities = communities_response["data"]["communities"]
        
        if communities:
            community_names = [f"{c['name']} ({c['id']})" for c in communities]
            selected_community = st.selectbox("Select Community", community_names)
            
            if selected_community:
                community_id = selected_community.split("(")[1].split(")")[0]
                selected_community_data = next(c for c in communities if c['id'] == community_id)
        else:
            st.warning("No communities found. Create one first!")
            community_id = None
            selected_community_data = None
    else:
        st.error("Failed to load communities")
        community_id = None
        selected_community_data = None
    
    st.markdown("---")
    
    st.header("⚙️ Controls")
    
    if st.button("🔄 Start Macro-Twin"):
        if community_id:
            result = call_api(f"communities/{community_id}/macro-twin/start", method="POST")
            if result and result.get("success"):
                st.success("Macro-twin started!")
            else:
                st.error("Failed to start macro-twin")
    
    if st.button("📈 Train Models"):
        if community_id:
            with st.spinner("Training forecasting models..."):
                result = call_api(f"communities/{community_id}/forecasting/train", method="POST")
                if result and result.get("success"):
                    st.success("Models trained successfully!")
                else:
                    st.error("Failed to train models")

if community_id and selected_community_data:
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "👥 Prosumers", 
            selected_community_data['num_prosumers']
        )
    
    with col2:
        st.metric(
            "☀️ PV Capacity", 
            f"{selected_community_data['total_pv_capacity_kw']:.1f} kW"
        )
    
    with col3:
        st.metric(
            "🔋 Storage Capacity", 
            f"{selected_community_data['total_storage_capacity_kwh']:.1f} kWh"
        )
    
    with col4:
        kpis_response = call_api(f"communities/{community_id}/analytics/kpis?days_back=7")
        if kpis_response and kpis_response.get("success"):
            kpis = kpis_response["data"]
            st.metric(
                "📊 Self-Consumption", 
                f"{kpis.get('self_consumption_rate', 0)*100:.1f}%"
            )
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Real-time", "🔮 Forecasting", "⚡ Optimization", "🎯 Simulation", "📈 Analytics"])
    
    with tab1:
        st.header("Real-time Community State")
        
        state_response = call_api(f"communities/{community_id}/macro-twin/state")
        
        if state_response and state_response.get("success"):
            state = state_response["data"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Current Metrics")
                
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Net Load", f"{state.get('net_load_kw', 0):.1f} kW")
                    st.metric("Grid Import", f"{state.get('grid_import_kw', 0):.1f} kW")
                    st.metric("Temperature", f"{state.get('temperature_c', 0):.1f}°C")
                
                with metrics_col2:
                    st.metric("PV Generation", f"{state.get('pv_generation_kw', 0):.1f} kW")
                    st.metric("Storage SOC", f"{state.get('storage_soc_kwh', 0):.1f} kWh")
                    st.metric("Irradiance", f"{state.get('irradiance_w_m2', 0):.0f} W/m²")
            
            with col2:
                st.subheader("System Status")
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = state.get('storage_soc_kwh', 0),
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Storage SOC (kWh)"},
                    gauge = {
                        'axis': {'range': [None, selected_community_data['total_storage_capacity_kwh']]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, selected_community_data['total_storage_capacity_kwh']*0.3], 'color': "lightgray"},
                            {'range': [selected_community_data['total_storage_capacity_kwh']*0.3, selected_community_data['total_storage_capacity_kwh']*0.7], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': selected_community_data['total_storage_capacity_kwh']*0.9
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Macro-twin not running. Start it from the sidebar.")
    
    with tab2:
        st.header("Energy Forecasting")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Forecast Parameters")
            horizon_hours = st.slider("Forecast Horizon (hours)", 1, 168, 24)
            confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95)
            include_uncertainty = st.checkbox("Include Uncertainty Bands", True)
            
            if st.button("Generate Forecast"):
                forecast_request = {
                    "start_time": datetime.utcnow().isoformat(),
                    "end_time": (datetime.utcnow() + timedelta(hours=horizon_hours)).isoformat(),
                    "horizon_hours": horizon_hours,
                    "confidence_level": confidence_level,
                    "include_uncertainty": include_uncertainty
                }
                
                with st.spinner("Generating forecasts..."):
                    forecast_response = call_api(
                        f"communities/{community_id}/forecasting/generate",
                        method="POST",
                        data=forecast_request
                    )
                    
                    if forecast_response and forecast_response.get("success"):
                        st.session_state.forecasts = forecast_response["data"]["forecasts"]
                        st.success("Forecasts generated!")
        
        with col2:
            if "forecasts" in st.session_state:
                st.subheader("Forecast Results")
                
                for forecast in st.session_state.forecasts:
                    variable = forecast["variable"]
                    predictions = forecast["predictions"]
                    
                    if predictions:
                        timestamps = [p["timestamp"] for p in predictions]
                        values = [p["value"] for p in predictions]
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=timestamps,
                            y=values,
                            mode='lines',
                            name=f'{variable} Forecast',
                            line=dict(color='blue')
                        ))
                        
                        if include_uncertainty and forecast.get("lower_bound") and forecast.get("upper_bound"):
                            lower_values = [p["value"] for p in forecast["lower_bound"]]
                            upper_values = [p["value"] for p in forecast["upper_bound"]]
                            
                            fig.add_trace(go.Scatter(
                                x=timestamps + timestamps[::-1],
                                y=upper_values + lower_values[::-1],
                                fill='toself',
                                fillcolor='rgba(0,100,80,0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                hoverinfo="skip",
                                showlegend=False,
                                name='Confidence Interval'
                            ))
                        
                        fig.update_layout(
                            title=f"{variable.replace('_', ' ').title()} Forecast",
                            xaxis_title="Time",
                            yaxis_title=predictions[0]["unit"],
                            height=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Energy Optimization")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Optimization Settings")
            
            objective = st.selectbox(
                "Optimization Objective",
                ["maximize_welfare", "minimize_cost", "minimize_peak", "maximize_self_consumption"]
            )
            
            opt_horizon = st.slider("Optimization Horizon (hours)", 1, 48, 24)
            current_soc = st.number_input("Current SOC (kWh)", 0.0, float(selected_community_data['total_storage_capacity_kwh']), 50.0)
            
            with st.expander("Tariff Settings"):
                peak_rate = st.number_input("Peak Rate ($/kWh)", 0.1, 1.0, 0.3)
                offpeak_rate = st.number_input("Off-peak Rate ($/kWh)", 0.05, 0.5, 0.15)
                feed_in_rate = st.number_input("Feed-in Rate ($/kWh)", 0.01, 0.2, 0.08)
            
            if st.button("Run Optimization"):
                optimization_request = {
                    "objective": objective,
                    "horizon_hours": opt_horizon,
                    "current_soc_kwh": current_soc,
                    "forecasted_load": [50.0] * (opt_horizon * 4),  
                    "forecasted_pv": [30.0] * (opt_horizon * 4),   
                    "current_tariff": {
                        "time_of_use_rates": {"peak": peak_rate, "off_peak": offpeak_rate},
                        "feed_in_tariff": feed_in_rate,
                        "demand_charge": 10.0,
                        "fixed_charge_daily": 5.0
                    }
                }
                
                with st.spinner("Running optimization..."):
                    opt_response = call_api(
                        f"communities/{community_id}/optimization/optimize",
                        method="POST",
                        data=optimization_request
                    )
                    
                    if opt_response and opt_response.get("success"):
                        st.session_state.optimization_result = opt_response["data"]
                        st.success("Optimization completed!")
        
        with col2:
            if "optimization_result" in st.session_state:
                result = st.session_state.optimization_result
                
                st.subheader("Optimization Results")
                
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Cost Savings", f"${result.get('cost_savings', 0):.2f}")
                    st.metric("Self-Consumption Rate", f"{result.get('self_consumption_rate', 0)*100:.1f}%")
                
                with metrics_col2:
                    st.metric("Peak Reduction", f"{result.get('peak_reduction_kw', 0):.1f} kW")
                    st.metric("Objective Value", f"{result.get('objective_value', 0):.2f}")
                
                if result.get("battery_dispatch"):
                    times = list(range(len(result["battery_dispatch"])))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=times,
                        y=result["battery_dispatch"],
                        mode='lines',
                        name='Battery Dispatch',
                        line=dict(color='green')
                    ))
                    
                    fig.update_layout(
                        title="Optimal Battery Dispatch Schedule",
                        xaxis_title="Time Period (15-min intervals)",
                        yaxis_title="Power (kW)",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Scenario Simulation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Scenario Definition")
            
            scenario_name = st.text_input("Scenario Name", "Test Scenario")
            scenario_description = st.text_area("Description", "Test scenario description")
            simulation_days = st.slider("Simulation Duration (days)", 1, 90, 30)
            
            st.subheader("Parameter Changes")
            
            pv_change = st.slider("PV Capacity Change (%)", -50, 100, 0)
            storage_change = st.slider("Storage Capacity Change (%)", -50, 100, 0)
            prosumer_change = st.slider("Number of Prosumers Change (%)", -20, 50, 0)
            
            if st.button("Run Scenario"):
                scenario_def = {
                    "scenario_id": f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "name": scenario_name,
                    "description": scenario_description,
                    "parameter_changes": {
                        "total_pv_capacity_kw": selected_community_data['total_pv_capacity_kw'] * (1 + pv_change/100),
                        "total_storage_capacity_kwh": selected_community_data['total_storage_capacity_kwh'] * (1 + storage_change/100),
                        "num_prosumers": int(selected_community_data['num_prosumers'] * (1 + prosumer_change/100))
                    },
                    "simulation_duration_days": simulation_days
                }
                
                with st.spinner("Running scenario simulation..."):
                    sim_response = call_api(
                        f"communities/{community_id}/simulation/run-scenario",
                        method="POST",
                        data=scenario_def
                    )
                    
                    if sim_response and sim_response.get("success"):
                        st.session_state.simulation_result = sim_response["data"]
                        st.success("Scenario simulation completed!")
        
        with col2:
            if "simulation_result" in st.session_state:
                result = st.session_state.simulation_result
                
                st.subheader("Simulation Results")
                
                kpis = result.get("kpis", {})
                
                kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
                
                with kpi_col1:
                    st.metric("Total Energy Consumed", f"{kpis.get('total_energy_consumed_kwh', 0):.0f} kWh")
                    st.metric("Peak Demand", f"{kpis.get('peak_demand_kw', 0):.1f} kW")
                
                with kpi_col2:
                    st.metric("Total PV Generated", f"{kpis.get('total_pv_generated_kwh', 0):.0f} kWh")
                    st.metric("Self-Consumption", f"{kpis.get('self_consumption_rate', 0)*100:.1f}%")
                
                with kpi_col3:
                    st.metric("Grid Export", f"{kpis.get('total_grid_export_kwh', 0):.0f} kWh")
                    st.metric("Cost Savings", f"${kpis.get('total_cost_savings', 0):.2f}")
                
                if "time_series_results" in result:
                    st.subheader("Time Series Results")
                    
                    variable_to_plot = st.selectbox(
                        "Select Variable", 
                        list(result["time_series_results"].keys())
                    )
                    
                    if variable_to_plot:
                        data_points = result["time_series_results"][variable_to_plot]
                        timestamps = [p["timestamp"] for p in data_points]
                        values = [p["value"] for p in data_points]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=timestamps,
                            y=values,
                            mode='lines',
                            name=variable_to_plot.replace('_', ' ').title()
                        ))
                        
                        fig.update_layout(
                            title=f"{variable_to_plot.replace('_', ' ').title()} Over Time",
                            xaxis_title="Time",
                            yaxis_title=data_points[0]["unit"] if data_points else "",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("Community Analytics")
        
        days_for_analysis = st.selectbox("Analysis Period", [7, 30, 90], index=1)
        
        kpis_response = call_api(f"communities/{community_id}/analytics/kpis?days_back={days_for_analysis}")
        
        if kpis_response and kpis_response.get("success"):
            kpis = kpis_response["data"]
            
            st.subheader(f"KPIs for Last {days_for_analysis} Days")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Energy Consumed", f"{kpis.get('total_energy_consumed_kwh', 0):.0f} kWh")
                st.metric("Peak Demand", f"{kpis.get('peak_demand_kw', 0):.1f} kW")
            
            with col2:
                st.metric("PV Generated", f"{kpis.get('total_pv_generated_kwh', 0):.0f} kWh")
                st.metric("Average Demand", f"{kpis.get('average_demand_kw', 0):.1f} kW")
            
            with col3:
                st.metric("Grid Import", f"{kpis.get('total_grid_import_kwh', 0):.0f} kWh")
                st.metric("Self-Consumption", f"{kpis.get('self_consumption_rate', 0)*100:.1f}%")
            
            with col4:
                st.metric("Grid Export", f"{kpis.get('total_grid_export_kwh', 0):.0f} kWh")
                st.metric("Self-Sufficiency", f"{kpis.get('self_sufficiency_rate', 0)*100:.1f}%")
            
            energy_balance_data = {
                'Type': ['Consumption', 'PV Generation', 'Grid Import', 'Grid Export'],
                'Energy (kWh)': [
                    kpis.get('total_energy_consumed_kwh', 0),
                    kpis.get('total_pv_generated_kwh', 0),
                    kpis.get('total_grid_import_kwh', 0),
                    kpis.get('total_grid_export_kwh', 0)
                ]
            }
            
            fig = px.bar(
                energy_balance_data, 
                x='Type', 
                y='Energy (kWh)',
                title=f"Energy Balance - Last {days_for_analysis} Days",
                color='Type'
            )
            
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Please select a community from the sidebar to view the dashboard")
    
    st.markdown("## 🚀 Create New Community")
    
    with st.form("create_community"):
        st.subheader("Community Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            community_id = st.text_input("Community ID", placeholder="e.g., community_001")
            num_prosumers = st.number_input("Number of Prosumers", 10, 1000, 100)
            pv_capacity = st.number_input("Total PV Capacity (kW)", 50.0, 5000.0, 500.0)
        
        with col2:
            storage_capacity = st.number_input("Storage Capacity (kWh)", 100.0, 10000.0, 1000.0)
            max_storage_power = st.number_input("Max Storage Power (kW)", 50.0, 2000.0, 500.0)
            incentive_budget = st.number_input("Daily Incentive Budget ($)", 100.0, 10000.0, 1000.0)
        
        col3, col4 = st.columns(2)
        
        with col3:
            grid_import_limit = st.number_input("Grid Import Limit (kW)", 100.0, 5000.0, 1000.0)
        
        with col4:
            grid_export_limit = st.number_input("Grid Export Limit (kW)", 100.0, 5000.0, 800.0)
        
        submitted = st.form_submit_button("🏗️ Create Community", use_container_width=True)
        
        if submitted and community_id:
            community_config = {
                "community_id": community_id,
                "num_prosumers": num_prosumers,
                "total_pv_capacity_kw": pv_capacity,
                "total_storage_capacity_kwh": storage_capacity,
                "max_storage_power_kw": max_storage_power,
                "grid_import_limit_kw": grid_import_limit,
                "grid_export_limit_kw": grid_export_limit,
                "incentive_budget_daily": incentive_budget
            }
            
            with st.spinner("Creating community..."):
                result = call_api("communities", method="POST", data=community_config)
                
                if result and result.get("success"):
                    st.success(f"✅ Community {community_id} created successfully!")
                    st.experimental_rerun()
                else:
                    st.error("❌ Failed to create community")

st.markdown("---")
st.markdown("*Community Digital Twin Dashboard - Real-time Energy Management*")
