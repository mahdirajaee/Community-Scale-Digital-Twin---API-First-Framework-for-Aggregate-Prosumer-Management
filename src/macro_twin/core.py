import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import logging

from ..data.models import CommunityState, CommunityConfig, MacroTwinState
from ..data.database import get_db, MeterData, WeatherData

logger = logging.getLogger(__name__)


@dataclass
class StateSpaceMatrices:
    F: np.ndarray  
    H: np.ndarray  
    Q: np.ndarray  
    R: np.ndarray  


class MacroTwinCore:
    
    def __init__(self, community_config: CommunityConfig):
        self.community_config = community_config
        self.state_dim = 12  
        self.obs_dim = 7    
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        self.state_vector = np.zeros(self.state_dim)
        self.covariance_matrix = np.eye(self.state_dim) * 0.1
        
        self._initialize_state_space_matrices()
        
    def _initialize_state_space_matrices(self):
        dt = 0.25  
        
        self.F = np.array([
            [1, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.98, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, dt, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.85, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, dt, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, dt, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.99]
        ])
        
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        ])
        
        process_noise_var = 0.01
        self.Q = np.eye(self.state_dim) * process_noise_var
        
        measurement_noise_vars = [0.1, 0.05, 0.02, 0.1, 0.1, 0.05, 0.01]
        self.R = np.diag(measurement_noise_vars)
        
    def update_state(self, observation: CommunityState) -> MacroTwinState:
        
        obs_vector = np.array([
            observation.net_load_kw,
            observation.pv_generation_kw,
            observation.storage_soc_kwh,
            observation.grid_import_kw,
            observation.grid_export_kw,
            observation.ambient_temperature_c,
            observation.irradiance_w_m2
        ])
        
        if not self.is_fitted:
            self._initialize_state_from_observation(obs_vector)
            self.is_fitted = True
            
        self.state_vector, self.covariance_matrix = self._kalman_update(
            self.state_vector, self.covariance_matrix, obs_vector
        )
        
        return MacroTwinState(
            community_id=self.community_config.community_id,
            state_vector=self.state_vector.tolist(),
            covariance_matrix=self.covariance_matrix.tolist(),
            last_update=observation.timestamp,
            prediction_horizon=24
        )
        
    def _initialize_state_from_observation(self, obs_vector: np.ndarray):
        self.state_vector[0] = obs_vector[0]  
        self.state_vector[1] = 0              
        self.state_vector[2] = obs_vector[1]  
        self.state_vector[3] = 0              
        self.state_vector[4] = obs_vector[2]  
        self.state_vector[5] = obs_vector[3]  
        self.state_vector[6] = 0              
        self.state_vector[7] = obs_vector[4]  
        self.state_vector[8] = 0              
        self.state_vector[9] = obs_vector[5]  
        self.state_vector[10] = 0             
        self.state_vector[11] = obs_vector[6] 
        
    def _kalman_update(self, x_pred: np.ndarray, P_pred: np.ndarray, 
                       observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        x_pred = self.F @ x_pred
        P_pred = self.F @ P_pred @ self.F.T + self.Q
        
        innovation = observation - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        x_updated = x_pred + K @ innovation
        P_updated = (np.eye(self.state_dim) - K @ self.H) @ P_pred
        
        return x_updated, P_updated
        
    def predict_state(self, steps_ahead: int) -> np.ndarray:
        
        predicted_states = []
        x_temp = self.state_vector.copy()
        P_temp = self.covariance_matrix.copy()
        
        for _ in range(steps_ahead):
            x_temp = self.F @ x_temp
            P_temp = self.F @ P_temp @ self.F.T + self.Q
            predicted_states.append(x_temp.copy())
            
        return np.array(predicted_states)
        
    def get_aggregate_metrics(self) -> Dict[str, float]:
        
        return {
            "net_load_kw": self.state_vector[0],
            "net_load_trend": self.state_vector[1],
            "pv_generation_kw": self.state_vector[2],
            "pv_trend": self.state_vector[3],
            "storage_soc_kwh": self.state_vector[4],
            "grid_import_kw": self.state_vector[5],
            "grid_import_trend": self.state_vector[6],
            "grid_export_kw": self.state_vector[7],
            "grid_export_trend": self.state_vector[8],
            "temperature_c": self.state_vector[9],
            "temperature_trend": self.state_vector[10],
            "irradiance_w_m2": self.state_vector[11],
            "uncertainty_net_load": np.sqrt(self.covariance_matrix[0, 0]),
            "uncertainty_pv": np.sqrt(self.covariance_matrix[2, 2]),
            "uncertainty_soc": np.sqrt(self.covariance_matrix[4, 4])
        }
        
    def validate_physical_constraints(self) -> bool:
        
        metrics = self.get_aggregate_metrics()
        
        constraints_met = True
        
        if metrics["storage_soc_kwh"] < 0 or metrics["storage_soc_kwh"] > self.community_config.total_storage_capacity_kwh:
            logger.warning(f"Storage SOC constraint violated: {metrics['storage_soc_kwh']}")
            constraints_met = False
            
        if metrics["pv_generation_kw"] < 0 or metrics["pv_generation_kw"] > self.community_config.total_pv_capacity_kw:
            logger.warning(f"PV generation constraint violated: {metrics['pv_generation_kw']}")
            constraints_met = False
            
        if abs(metrics["grid_import_kw"]) > self.community_config.grid_import_limit_kw:
            logger.warning(f"Grid import limit violated: {metrics['grid_import_kw']}")
            constraints_met = False
            
        if abs(metrics["grid_export_kw"]) > self.community_config.grid_export_limit_kw:
            logger.warning(f"Grid export limit violated: {metrics['grid_export_kw']}")
            constraints_met = False
            
        return constraints_met
        
    def save_state(self, db_session):
        
        from ..data.database import MacroTwinStateDB
        
        state_record = MacroTwinStateDB(
            community_id=self.community_config.community_id,
            state_vector=self.state_vector.tolist(),
            covariance_matrix=self.covariance_matrix.tolist(),
            timestamp=datetime.utcnow(),
            prediction_horizon=24
        )
        
        db_session.add(state_record)
        db_session.commit()
        
    def load_state(self, db_session) -> bool:
        
        from ..data.database import MacroTwinStateDB
        
        latest_state = db_session.query(MacroTwinStateDB)\
            .filter(MacroTwinStateDB.community_id == self.community_config.community_id)\
            .order_by(MacroTwinStateDB.timestamp.desc())\
            .first()
            
        if latest_state:
            self.state_vector = np.array(latest_state.state_vector)
            self.covariance_matrix = np.array(latest_state.covariance_matrix)
            self.is_fitted = True
            return True
            
        return False
