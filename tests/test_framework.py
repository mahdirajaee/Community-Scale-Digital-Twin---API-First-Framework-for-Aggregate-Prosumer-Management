"""
Comprehensive unit tests for the Digital Twin Framework
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio
from typing import List, Dict, Any

# Import the modules we're testing
from src.data.models import (
    CommunityState, MacroTwinState, ForecastRequest, 
    OptimizationRequest, ScenarioRequest
)
from src.macro_twin.core import MacroTwinCore
from src.forecasting.models import ForecastingService
from src.optimization.engine import OptimizationEngine
from src.simulation.engine import ScenarioSimulator


class TestDataModels:
    """Test suite for data models validation"""
    
    def test_community_state_validation(self):
        """Test CommunityState model validation"""
        # Valid community state
        valid_state = CommunityState(
            community_id="test_community",
            timestamp=datetime.utcnow(),
            total_generation_kw=100.0,
            total_consumption_kw=80.0,
            net_power_kw=20.0,
            grid_exchange_kw=-20.0,
            energy_price_per_kwh=0.25,
            grid_export_price_per_kwh=0.15,
            active_prosumers=50
        )
        assert valid_state.total_generation_kw == 100.0
        assert valid_state.active_prosumers == 50
        
    def test_community_state_invalid_data(self):
        """Test CommunityState validation with invalid data"""
        with pytest.raises(ValueError):
            CommunityState(
                community_id="test_community",
                timestamp=datetime.utcnow(),
                total_generation_kw=-10.0,  # Invalid: negative generation
                total_consumption_kw=80.0,
                net_power_kw=20.0,
                grid_exchange_kw=-20.0,
                energy_price_per_kwh=0.25,
                grid_export_price_per_kwh=0.15,
                active_prosumers=50
            )
    
    def test_macro_twin_state_validation(self):
        """Test MacroTwinState model validation"""
        state = MacroTwinState(
            timestamp=datetime.utcnow(),
            aggregate_generation=150.0,
            aggregate_consumption=120.0,
            grid_exchange=-30.0,
            battery_soc=75.0,
            solar_irradiance=800.0,
            temperature=25.0,
            wind_speed=5.0,
            energy_price=0.25,
            demand_flexibility=20.0,
            network_constraint=0.8,
            weather_forecast=0.9,
            market_signal=0.5
        )
        assert state.battery_soc == 75.0
        assert state.solar_irradiance == 800.0
        
    def test_forecast_request_validation(self):
        """Test ForecastRequest model validation"""
        request = ForecastRequest(
            community_id="test_community",
            forecast_horizon_hours=48,
            target_variables=["consumption", "generation"]
        )
        assert request.forecast_horizon_hours == 48
        assert "consumption" in request.target_variables


class TestMacroTwinCore:
    """Test suite for Macro Twin core functionality"""
    
    @pytest.fixture
    def mock_community_config(self):
        """Mock community configuration"""
        return {
            "community_id": "test_community",
            "num_prosumers": 100,
            "total_pv_capacity_kw": 500.0,
            "total_storage_capacity_kwh": 1000.0,
            "grid_import_limit_kw": 1000.0
        }
    
    @pytest.fixture
    def macro_twin(self, mock_community_config):
        """Create MacroTwinCore instance for testing"""
        return MacroTwinCore(mock_community_config)
    
    def test_initialization(self, macro_twin):
        """Test MacroTwinCore initialization"""
        assert macro_twin.state_dim == 12
        assert macro_twin.obs_dim == 7
        assert macro_twin.state_vector.shape == (12,)
        assert macro_twin.covariance_matrix.shape == (12, 12)
    
    def test_state_prediction(self, macro_twin):
        """Test state prediction functionality"""
        # Set initial state
        initial_state = np.array([
            100.0,  # aggregate_generation
            80.0,   # aggregate_consumption
            20.0,   # grid_exchange
            75.0,   # battery_soc
            800.0,  # solar_irradiance
            25.0,   # temperature
            5.0,    # wind_speed
            0.25,   # energy_price
            15.0,   # demand_flexibility
            0.8,    # network_constraint
            0.9,    # weather_forecast
            0.5     # market_signal
        ])
        
        predicted_state, predicted_covariance = macro_twin.predict_state(initial_state)
        
        assert predicted_state.shape == (12,)
        assert predicted_covariance.shape == (12, 12)
        assert np.all(np.isfinite(predicted_state))
        assert np.all(np.isfinite(predicted_covariance))
    
    def test_measurement_update(self, macro_twin):
        """Test measurement update (Kalman filter correction)"""
        # Mock measurement data
        measurements = np.array([100.0, 80.0, 20.0, 75.0, 800.0, 25.0, 5.0])
        
        updated_state, updated_covariance = macro_twin.update_state(measurements)
        
        assert updated_state.shape == (12,)
        assert updated_covariance.shape == (12, 12)
        assert np.all(np.isfinite(updated_state))
        assert np.all(np.isfinite(updated_covariance))
    
    def test_state_estimation_pipeline(self, macro_twin):
        """Test complete state estimation pipeline"""
        # Generate synthetic data
        timestamps = [datetime.utcnow() + timedelta(minutes=15*i) for i in range(10)]
        measurements = []
        
        for i in range(10):
            measurement = {
                "timestamp": timestamps[i],
                "net_load_kw": 80.0 + 10*np.sin(i*0.1),
                "pv_generation_kw": 100.0 + 20*np.sin(i*0.2),
                "grid_import_kw": 20.0 + 5*np.random.normal(),
                "storage_soc_kwh": 750.0 - i*10,
                "solar_irradiance": 800.0 + 100*np.sin(i*0.15),
                "temperature": 25.0 + 2*np.sin(i*0.1),
                "wind_speed": 5.0 + np.random.normal()
            }
            measurements.append(measurement)
        
        # Process measurements
        states = []
        for measurement in measurements:
            state = macro_twin.process_measurement(measurement)
            states.append(state)
        
        assert len(states) == 10
        assert all(isinstance(state, MacroTwinState) for state in states)


class TestForecastingService:
    """Test suite for forecasting service"""
    
    @pytest.fixture
    def forecasting_service(self):
        """Create ForecastingService instance for testing"""
        return ForecastingService()
    
    @pytest.fixture
    def sample_time_series_data(self):
        """Generate sample time series data for testing"""
        dates = pd.date_range(
            start=datetime.utcnow() - timedelta(days=30),
            end=datetime.utcnow(),
            freq='15T'
        )
        
        # Generate synthetic data with patterns
        consumption = 50 + 20*np.sin(np.arange(len(dates))*2*np.pi/96) + np.random.normal(0, 5, len(dates))
        generation = 30 + 40*np.sin(np.arange(len(dates))*2*np.pi/96 + np.pi/4) + np.random.normal(0, 8, len(dates))
        
        return pd.DataFrame({
            'timestamp': dates,
            'consumption': consumption,
            'generation': generation
        })
    
    def test_data_preprocessing(self, forecasting_service, sample_time_series_data):
        """Test data preprocessing functionality"""
        processed_data = forecasting_service.preprocess_data(sample_time_series_data)
        
        assert 'consumption_normalized' in processed_data.columns
        assert 'generation_normalized' in processed_data.columns
        assert processed_data['consumption_normalized'].mean() == pytest.approx(0, abs=0.1)
        assert processed_data['generation_normalized'].std() == pytest.approx(1, abs=0.1)
    
    @patch('src.forecasting.models.tf.keras.models.Sequential')
    def test_lstm_model_training(self, mock_sequential, forecasting_service, sample_time_series_data):
        """Test LSTM model training"""
        mock_model = Mock()
        mock_sequential.return_value = mock_model
        mock_model.fit.return_value = Mock()
        
        # Train model
        model = forecasting_service.train_lstm_model(
            sample_time_series_data,
            target_column='consumption',
            sequence_length=96,
            epochs=5
        )
        
        mock_sequential.assert_called_once()
        mock_model.compile.assert_called_once()
        mock_model.fit.assert_called_once()
    
    def test_prophet_model_training(self, forecasting_service, sample_time_series_data):
        """Test Prophet model training"""
        # Prepare data for Prophet
        prophet_data = sample_time_series_data[['timestamp', 'consumption']].copy()
        prophet_data.columns = ['ds', 'y']
        
        model = forecasting_service.train_prophet_model(prophet_data)
        
        assert model is not None
        # Prophet model should have fitted data
        assert hasattr(model, 'history')
    
    def test_forecast_generation(self, forecasting_service, sample_time_series_data):
        """Test forecast generation"""
        # Mock trained models
        forecasting_service.lstm_models = {'consumption': Mock()}
        forecasting_service.prophet_models = {'consumption': Mock()}
        
        # Mock predictions
        mock_lstm_pred = np.random.normal(50, 10, 96)  # 24 hours * 4 (15-min intervals)
        mock_prophet_pred = pd.DataFrame({
            'ds': pd.date_range(start=datetime.utcnow(), periods=96, freq='15T'),
            'yhat': np.random.normal(50, 10, 96),
            'yhat_lower': np.random.normal(40, 8, 96),
            'yhat_upper': np.random.normal(60, 12, 96)
        })
        
        forecasting_service.lstm_models['consumption'].predict.return_value = mock_lstm_pred.reshape(-1, 1)
        forecasting_service.prophet_models['consumption'].predict.return_value = mock_prophet_pred
        
        forecast = forecasting_service.generate_forecast(
            community_id="test_community",
            target_variable="consumption",
            horizon_hours=24
        )
        
        assert len(forecast['timestamps']) == 96
        assert len(forecast['predictions']) == 96
        assert 'confidence_intervals' in forecast


class TestOptimizationEngine:
    """Test suite for optimization engine"""
    
    @pytest.fixture
    def optimization_engine(self):
        """Create OptimizationEngine instance for testing"""
        return OptimizationEngine()
    
    @pytest.fixture
    def sample_optimization_data(self):
        """Generate sample data for optimization testing"""
        horizon = 24  # 24 hours
        
        return {
            'demand_forecast': 50 + 20*np.sin(np.arange(horizon)*2*np.pi/24) + np.random.normal(0, 2, horizon),
            'pv_forecast': 30 + 40*np.maximum(0, np.sin(np.arange(horizon)*2*np.pi/24)),
            'price_forecast': 0.2 + 0.1*np.sin(np.arange(horizon)*2*np.pi/24 + np.pi),
            'battery_capacity_kwh': 100.0,
            'battery_power_kw': 25.0,
            'initial_soc': 50.0
        }
    
    def test_optimization_problem_setup(self, optimization_engine, sample_optimization_data):
        """Test optimization problem formulation"""
        problem = optimization_engine.setup_optimization_problem(
            objective="cost_minimization",
            **sample_optimization_data
        )
        
        assert problem is not None
        assert hasattr(problem, 'objective')
        assert hasattr(problem, 'constraints')
    
    def test_cost_minimization_optimization(self, optimization_engine, sample_optimization_data):
        """Test cost minimization optimization"""
        result = optimization_engine.optimize(
            objective="cost_minimization",
            **sample_optimization_data
        )
        
        assert result['feasible'] is True
        assert 'optimal_battery_schedule' in result
        assert 'optimal_grid_exchange' in result
        assert 'total_cost' in result
        assert len(result['optimal_battery_schedule']) == 24
    
    def test_peak_minimization_optimization(self, optimization_engine, sample_optimization_data):
        """Test peak minimization optimization"""
        result = optimization_engine.optimize(
            objective="peak_minimization",
            **sample_optimization_data
        )
        
        assert result['feasible'] is True
        assert 'peak_reduction_percent' in result
        peak_demand = max(result['optimal_grid_exchange'])
        original_peak = max(sample_optimization_data['demand_forecast'])
        assert peak_demand <= original_peak
    
    def test_self_consumption_optimization(self, optimization_engine, sample_optimization_data):
        """Test self-consumption optimization"""
        result = optimization_engine.optimize(
            objective="self_consumption",
            **sample_optimization_data
        )
        
        assert result['feasible'] is True
        assert 'self_consumption_ratio' in result
        assert 0 <= result['self_consumption_ratio'] <= 1


class TestScenarioSimulator:
    """Test suite for scenario simulation"""
    
    @pytest.fixture
    def scenario_simulator(self):
        """Create ScenarioSimulator instance for testing"""
        return ScenarioSimulator()
    
    @pytest.fixture
    def sample_scenario_config(self):
        """Generate sample scenario configuration"""
        return {
            'community_id': 'test_community',
            'simulation_hours': 168,  # 1 week
            'monte_carlo_runs': 10,
            'solar_penetration_percent': 60.0,
            'battery_penetration_percent': 40.0,
            'ev_penetration_percent': 25.0,
            'tariff_structure': {
                'peak': 0.35,
                'off_peak': 0.15,
                'shoulder': 0.25
            }
        }
    
    def test_scenario_setup(self, scenario_simulator, sample_scenario_config):
        """Test scenario configuration setup"""
        scenario = scenario_simulator.setup_scenario(**sample_scenario_config)
        
        assert scenario['community_id'] == 'test_community'
        assert scenario['simulation_hours'] == 168
        assert 'prosumer_profiles' in scenario
        assert 'technology_mix' in scenario
    
    def test_monte_carlo_simulation(self, scenario_simulator, sample_scenario_config):
        """Test Monte Carlo simulation execution"""
        results = scenario_simulator.run_monte_carlo_simulation(**sample_scenario_config)
        
        assert 'mean_community_cost' in results
        assert 'std_community_cost' in results
        assert 'cost_percentiles' in results
        assert len(results['individual_runs']) == sample_scenario_config['monte_carlo_runs']
    
    def test_policy_impact_analysis(self, scenario_simulator, sample_scenario_config):
        """Test policy impact analysis"""
        baseline_config = sample_scenario_config.copy()
        policy_config = sample_scenario_config.copy()
        policy_config['tariff_structure']['peak'] = 0.45  # Increase peak rate
        
        impact_analysis = scenario_simulator.analyze_policy_impact(
            baseline_config,
            policy_config
        )
        
        assert 'cost_difference' in impact_analysis
        assert 'welfare_change' in impact_analysis
        assert 'demand_response_effect' in impact_analysis


class TestIntegration:
    """Integration tests for the complete framework"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # This would test the integration between all components
        # For now, we'll test a simplified version
        
        # 1. Generate synthetic community data
        community_data = {
            'community_id': 'integration_test',
            'prosumer_count': 50,
            'simulation_start': datetime.utcnow() - timedelta(days=1),
            'simulation_end': datetime.utcnow()
        }
        
        # 2. Create macro twin state
        macro_twin_state = MacroTwinState(
            timestamp=datetime.utcnow(),
            aggregate_generation=100.0,
            aggregate_consumption=80.0,
            grid_exchange=20.0,
            battery_soc=60.0,
            solar_irradiance=750.0,
            temperature=22.0,
            wind_speed=3.5,
            energy_price=0.22,
            demand_flexibility=15.0,
            network_constraint=0.85,
            weather_forecast=0.95,
            market_signal=0.3
        )
        
        # 3. Test forecast request and response
        forecast_request = ForecastRequest(
            community_id='integration_test',
            forecast_horizon_hours=24
        )
        
        # 4. Test optimization request
        optimization_request = OptimizationRequest(
            community_id='integration_test',
            objective='welfare_maximization',
            import_price_per_kwh=0.25,
            export_price_per_kwh=0.15
        )
        
        # 5. Test scenario request
        scenario_request = ScenarioRequest(
            community_id='integration_test',
            scenario_name='baseline',
            tariff_structure={'peak': 0.35, 'off_peak': 0.15}
        )
        
        # Assert all objects are created successfully
        assert macro_twin_state.community_id is not None
        assert forecast_request.community_id == 'integration_test'
        assert optimization_request.objective == 'welfare_maximization'
        assert scenario_request.scenario_name == 'baseline'


# Test fixtures and utilities
@pytest.fixture(scope="session")
def test_database():
    """Create test database for integration tests"""
    # This would set up a test database
    pass


@pytest.fixture
def mock_redis():
    """Mock Redis connection for testing"""
    return Mock()


@pytest.fixture
def sample_weather_data():
    """Generate sample weather data"""
    return {
        'temperature': 25.0,
        'humidity': 60.0,
        'wind_speed': 5.0,
        'solar_irradiance': 800.0,
        'cloud_cover': 20.0
    }


# Performance tests
class TestPerformance:
    """Performance and load testing"""
    
    def test_macro_twin_processing_speed(self):
        """Test that macro twin can process data within time limits"""
        import time
        
        macro_twin = MacroTwinCore({'community_id': 'perf_test'})
        measurements = np.random.normal(0, 1, 7)
        
        start_time = time.time()
        for _ in range(1000):
            macro_twin.update_state(measurements)
        end_time = time.time()
        
        processing_time_ms = (end_time - start_time) * 1000
        assert processing_time_ms < 1000  # Should process 1000 updates in < 1 second
    
    def test_optimization_scalability(self):
        """Test optimization performance with increasing problem size"""
        optimization_engine = OptimizationEngine()
        
        for horizon in [24, 48, 96, 168]:  # 1 day to 1 week
            data = {
                'demand_forecast': np.random.normal(50, 10, horizon),
                'pv_forecast': np.random.normal(30, 15, horizon),
                'price_forecast': np.random.normal(0.25, 0.05, horizon),
                'battery_capacity_kwh': 100.0,
                'battery_power_kw': 25.0,
                'initial_soc': 50.0
            }
            
            start_time = time.time()
            result = optimization_engine.optimize(objective="cost_minimization", **data)
            end_time = time.time()
            
            assert result['feasible'] is True
            assert (end_time - start_time) < 30  # Should complete within 30 seconds


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
