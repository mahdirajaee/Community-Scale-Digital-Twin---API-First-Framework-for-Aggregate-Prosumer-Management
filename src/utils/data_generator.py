import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import random
import logging

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    
    def __init__(self, community_id: str, num_prosumers: int, 
                 pv_capacity_kw: float, storage_capacity_kwh: float):
        self.community_id = community_id
        self.num_prosumers = num_prosumers
        self.pv_capacity_kw = pv_capacity_kw
        self.storage_capacity_kwh = storage_capacity_kwh
        
        self.base_load_profiles = self._generate_base_load_profiles()
        self.pv_profile_template = self._generate_pv_profile_template()
        
    def generate_historical_data(self, start_date: datetime, 
                                end_date: datetime, interval_minutes: int = 15) -> pd.DataFrame:
        
        timestamps = pd.date_range(start=start_date, end=end_date, freq=f'{interval_minutes}min')
        
        data = []
        current_soc = self.storage_capacity_kwh * 0.5  
        
        for timestamp in timestamps:
            
            day_of_year = timestamp.timetuple().tm_yday
            hour_of_day = timestamp.hour + timestamp.minute / 60
            day_of_week = timestamp.weekday()
            
            net_load = self._generate_load(hour_of_day, day_of_week, day_of_year)
            pv_generation = self._generate_pv(hour_of_day, day_of_year, timestamp)
            
            storage_action, current_soc = self._simulate_storage_operation(
                net_load, pv_generation, current_soc, hour_of_day
            )
            
            grid_import = max(0, net_load - pv_generation + storage_action)
            grid_export = max(0, pv_generation - net_load - storage_action)
            
            storage_charge = max(0, storage_action)
            storage_discharge = max(0, -storage_action)
            
            data.append({
                'timestamp': timestamp,
                'community_id': self.community_id,
                'net_load_kw': round(net_load, 2),
                'pv_generation_kw': round(pv_generation, 2),
                'grid_import_kw': round(grid_import, 2),
                'grid_export_kw': round(grid_export, 2),
                'storage_soc_kwh': round(current_soc, 2),
                'storage_charge_kw': round(storage_charge, 2),
                'storage_discharge_kw': round(storage_discharge, 2)
            })
            
        return pd.DataFrame(data)
        
    def generate_weather_data(self, start_date: datetime, 
                             end_date: datetime, interval_minutes: int = 15) -> pd.DataFrame:
        
        timestamps = pd.date_range(start=start_date, end=end_date, freq=f'{interval_minutes}min')
        
        data = []
        
        for timestamp in timestamps:
            day_of_year = timestamp.timetuple().tm_yday
            hour_of_day = timestamp.hour + timestamp.minute / 60
            
            temperature = self._generate_temperature(hour_of_day, day_of_year)
            irradiance = self._generate_irradiance(hour_of_day, day_of_year)
            humidity = self._generate_humidity(temperature, hour_of_day)
            wind_speed = self._generate_wind_speed()
            
            data.append({
                'timestamp': timestamp,
                'community_id': self.community_id,
                'irradiance_w_m2': round(irradiance, 1),
                'temperature_c': round(temperature, 1),
                'humidity_percent': round(humidity, 1),
                'wind_speed_m_s': round(wind_speed, 1)
            })
            
        return pd.DataFrame(data)
        
    def _generate_base_load_profiles(self) -> Dict[str, List[float]]:
        
        residential_profile = [
            0.6, 0.5, 0.4, 0.4, 0.4, 0.5, 0.7, 0.9,  
            0.8, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9,  
            1.0, 1.2, 1.4, 1.5, 1.3, 1.1, 0.9, 0.7   
        ]
        
        commercial_profile = [
            0.3, 0.3, 0.3, 0.3, 0.4, 0.6, 0.8, 1.0,  
            1.2, 1.3, 1.3, 1.2, 1.1, 1.2, 1.3, 1.2,  
            1.1, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.3   
        ]
        
        return {
            'residential': residential_profile,
            'commercial': commercial_profile
        }
        
    def _generate_pv_profile_template(self) -> List[float]:
        
        return [
            0, 0, 0, 0, 0, 0, 0.1, 0.3,  
            0.6, 0.8, 0.9, 1.0, 1.0, 0.9,  
            0.8, 0.6, 0.4, 0.2, 0.1, 0,   
            0, 0, 0, 0                     
        ]
        
    def _generate_load(self, hour_of_day: float, day_of_week: int, day_of_year: int) -> float:
        
        hour_index = int(hour_of_day)
        
        residential_factor = 0.7
        commercial_factor = 0.3
        
        base_load = (
            residential_factor * self.base_load_profiles['residential'][hour_index] +
            commercial_factor * self.base_load_profiles['commercial'][hour_index]
        )
        
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        weekend_factor = 0.8 if day_of_week >= 5 else 1.0
        
        load_per_prosumer = base_load * seasonal_factor * weekend_factor * 2.5  
        total_load = load_per_prosumer * self.num_prosumers
        
        noise = np.random.normal(0, 0.05 * total_load)
        
        return max(0, total_load + noise)
        
    def _generate_pv(self, hour_of_day: float, day_of_year: int, timestamp: datetime) -> float:
        
        hour_index = int(hour_of_day)
        
        base_generation = self.pv_profile_template[hour_index]
        
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        cloud_factor = np.random.uniform(0.7, 1.0) if base_generation > 0 else 1.0
        
        total_generation = base_generation * seasonal_factor * cloud_factor * self.pv_capacity_kw
        
        return max(0, total_generation)
        
    def _generate_temperature(self, hour_of_day: float, day_of_year: int) -> float:
        
        seasonal_avg = 20 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        daily_variation = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        noise = np.random.normal(0, 2)
        
        return seasonal_avg + daily_variation + noise
        
    def _generate_irradiance(self, hour_of_day: float, day_of_year: int) -> float:
        
        if hour_of_day < 6 or hour_of_day > 19:
            return 0
            
        max_irradiance = 1000  
        
        sun_elevation = np.sin(2 * np.pi * (hour_of_day - 6) / 13)  
        
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        cloud_factor = np.random.uniform(0.6, 1.0)
        
        irradiance = max_irradiance * sun_elevation * seasonal_factor * cloud_factor
        
        return max(0, irradiance)
        
    def _generate_humidity(self, temperature: float, hour_of_day: float) -> float:
        
        base_humidity = 60
        
        temp_effect = -0.5 * (temperature - 20)
        
        daily_variation = 10 * np.sin(2 * np.pi * (hour_of_day + 6) / 24)
        
        noise = np.random.normal(0, 5)
        
        humidity = base_humidity + temp_effect + daily_variation + noise
        
        return max(20, min(95, humidity))
        
    def _generate_wind_speed(self) -> float:
        
        base_speed = np.random.exponential(3)  
        
        return max(0, min(20, base_speed))
        
    def _simulate_storage_operation(self, net_load: float, pv_generation: float, 
                                   current_soc: float, hour_of_day: float) -> Tuple[float, float]:
        
        max_charge_rate = self.storage_capacity_kwh * 0.5  
        max_discharge_rate = self.storage_capacity_kwh * 0.5
        
        net_demand = net_load - pv_generation
        
        storage_action = 0
        
        if net_demand < 0:  
            available_storage = self.storage_capacity_kwh - current_soc
            charge_power = min(abs(net_demand), max_charge_rate, available_storage * 4)
            storage_action = charge_power
            
        elif net_demand > 0 and 17 <= hour_of_day <= 21:  
            available_discharge = current_soc
            discharge_power = min(net_demand * 0.7, max_discharge_rate, available_discharge * 4)
            storage_action = -discharge_power
            
        new_soc = current_soc + storage_action * 0.25  
        new_soc = max(0, min(self.storage_capacity_kwh, new_soc))
        
        return storage_action, new_soc


def populate_database_with_synthetic_data(community_id: str, days_of_data: int = 90):
    
    from ..data.database import get_db, Community, MeterData, WeatherData
    
    db = next(get_db())
    
    try:
        community = db.query(Community).filter(Community.id == community_id).first()
        
        if not community:
            logger.error(f"Community {community_id} not found")
            return False
            
        generator = SyntheticDataGenerator(
            community_id=community_id,
            num_prosumers=community.num_prosumers,
            pv_capacity_kw=community.total_pv_capacity_kw,
            storage_capacity_kwh=community.total_storage_capacity_kwh
        )
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_of_data)
        
        logger.info(f"Generating {days_of_data} days of synthetic data for community {community_id}")
        
        meter_df = generator.generate_historical_data(start_date, end_date)
        weather_df = generator.generate_weather_data(start_date, end_date)
        
        for _, row in meter_df.iterrows():
            meter_record = MeterData(**row.to_dict())
            db.add(meter_record)
            
        for _, row in weather_df.iterrows():
            weather_record = WeatherData(**row.to_dict())
            db.add(weather_record)
            
        db.commit()
        
        logger.info(f"Successfully populated database with {len(meter_df)} meter records and {len(weather_df)} weather records")
        
        return True
        
    except Exception as e:
        logger.error(f"Error populating database: {e}")
        db.rollback()
        return False
        
    finally:
        db.close()


def calculate_performance_metrics(predictions: List[float], actuals: List[float]) -> Dict[str, float]:
    
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have the same length")
        
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100 if np.all(actuals != 0) else float('inf')
    
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2),
        'n_samples': len(predictions)
    }


def validate_optimization_constraints(result_dict: Dict, community_config) -> List[str]:
    
    violations = []
    
    if 'battery_dispatch' in result_dict:
        max_charge = max([max(0, x) for x in result_dict['battery_dispatch']])
        max_discharge = max([max(0, -x) for x in result_dict['battery_dispatch']])
        
        if max_charge > community_config.max_storage_power_kw:
            violations.append(f"Battery charge rate violation: {max_charge} > {community_config.max_storage_power_kw}")
            
        if max_discharge > community_config.max_storage_power_kw:
            violations.append(f"Battery discharge rate violation: {max_discharge} > {community_config.max_storage_power_kw}")
            
    if 'grid_exchange' in result_dict:
        max_import = max([max(0, x) for x in result_dict['grid_exchange']])
        max_export = max([max(0, -x) for x in result_dict['grid_exchange']])
        
        if max_import > community_config.grid_import_limit_kw:
            violations.append(f"Grid import limit violation: {max_import} > {community_config.grid_import_limit_kw}")
            
        if max_export > community_config.grid_export_limit_kw:
            violations.append(f"Grid export limit violation: {max_export} > {community_config.grid_export_limit_kw}")
            
    return violations


class TimeSeriesValidator:
    
    @staticmethod
    def detect_anomalies(data: List[float], threshold_std: float = 3.0) -> List[int]:
        
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        
        anomaly_indices = []
        for i, value in enumerate(data_array):
            if abs(value - mean) > threshold_std * std:
                anomaly_indices.append(i)
                
        return anomaly_indices
        
    @staticmethod
    def check_data_continuity(timestamps: List[datetime], expected_interval_minutes: int = 15) -> List[Dict]:
        
        gaps = []
        expected_delta = timedelta(minutes=expected_interval_minutes)
        
        for i in range(1, len(timestamps)):
            actual_delta = timestamps[i] - timestamps[i-1]
            if actual_delta > expected_delta * 1.5:  
                gaps.append({
                    'start_time': timestamps[i-1],
                    'end_time': timestamps[i],
                    'gap_duration_minutes': actual_delta.total_seconds() / 60
                })
                
        return gaps
        
    @staticmethod
    def validate_physical_limits(data: Dict[str, List[float]], 
                                config) -> Dict[str, List[str]]:
        
        violations = {
            'net_load': [],
            'pv_generation': [],
            'storage_soc': [],
            'grid_import': [],
            'grid_export': []
        }
        
        if 'pv_generation' in data:
            for i, pv in enumerate(data['pv_generation']):
                if pv < 0:
                    violations['pv_generation'].append(f"Index {i}: Negative PV generation ({pv})")
                elif pv > config.total_pv_capacity_kw * 1.1:  
                    violations['pv_generation'].append(f"Index {i}: PV exceeds capacity ({pv} > {config.total_pv_capacity_kw})")
                    
        if 'storage_soc' in data:
            for i, soc in enumerate(data['storage_soc']):
                if soc < 0:
                    violations['storage_soc'].append(f"Index {i}: Negative SOC ({soc})")
                elif soc > config.total_storage_capacity_kwh:
                    violations['storage_soc'].append(f"Index {i}: SOC exceeds capacity ({soc} > {config.total_storage_capacity_kwh})")
                    
        return violations
