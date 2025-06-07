from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np


class TimeSeriesDataPoint(BaseModel):
    timestamp: datetime
    value: float
    unit: str


class CommunityMetrics(BaseModel):
    timestamp: datetime
    net_load_kw: float
    pv_generation_kw: float
    grid_import_kw: float
    grid_export_kw: float
    storage_soc_kwh: float
    storage_charge_kw: float
    storage_discharge_kw: float


class WeatherData(BaseModel):
    timestamp: datetime
    irradiance_w_m2: float
    temperature_c: float
    humidity_percent: float
    wind_speed_m_s: float


class CommunityConfig(BaseModel):
    community_id: str
    num_prosumers: int
    total_pv_capacity_kw: float
    total_storage_capacity_kwh: float
    max_storage_power_kw: float
    grid_import_limit_kw: float
    grid_export_limit_kw: float
    incentive_budget_daily: float


class ForecastRequest(BaseModel):
    start_time: datetime
    end_time: datetime
    horizon_hours: int = Field(default=24, ge=1, le=168)
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)
    include_uncertainty: bool = True


class ForecastResult(BaseModel):
    variable: str
    predictions: List[TimeSeriesDataPoint]
    lower_bound: Optional[List[TimeSeriesDataPoint]] = None
    upper_bound: Optional[List[TimeSeriesDataPoint]] = None
    confidence_level: float
    mae: Optional[float] = None
    rmse: Optional[float] = None


class OptimizationObjective(str, Enum):
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_WELFARE = "maximize_welfare"
    MINIMIZE_PEAK = "minimize_peak"
    MAXIMIZE_SELF_CONSUMPTION = "maximize_self_consumption"


class TariffStructure(BaseModel):
    time_of_use_rates: Dict[str, float]
    feed_in_tariff: float
    demand_charge: float
    fixed_charge_daily: float


class OptimizationRequest(BaseModel):
    objective: OptimizationObjective
    horizon_hours: int = Field(default=24, ge=1, le=168)
    current_soc_kwh: float
    forecasted_load: List[float]
    forecasted_pv: List[float]
    current_tariff: TariffStructure
    constraints: Dict[str, Any] = {}


class OptimizationResult(BaseModel):
    objective_value: float
    battery_dispatch: List[float]
    grid_exchange: List[float]
    recommended_tariff: TariffStructure
    cost_savings: float
    peak_reduction_kw: float
    self_consumption_rate: float


class ScenarioDefinition(BaseModel):
    scenario_id: str
    name: str
    description: str
    parameter_changes: Dict[str, Any]
    simulation_duration_days: int = Field(default=30, ge=1, le=365)


class ScenarioResult(BaseModel):
    scenario_id: str
    kpis: Dict[str, float]
    time_series_results: Dict[str, List[TimeSeriesDataPoint]]
    summary_stats: Dict[str, Any]


class CommunityState(BaseModel):
    timestamp: datetime
    net_load_kw: float
    pv_generation_kw: float
    storage_soc_kwh: float
    grid_import_kw: float
    grid_export_kw: float
    ambient_temperature_c: float
    irradiance_w_m2: float
    
    
class MacroTwinState(BaseModel):
    community_id: str
    state_vector: List[float]
    covariance_matrix: List[List[float]]
    last_update: datetime
    prediction_horizon: int = 24


class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: datetime
    execution_time_ms: float


# Authentication and User Models
class UserRole(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    API_USER = "api_user"


class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    full_name: Optional[str] = None
    role: UserRole = UserRole.VIEWER
    is_active: bool = True


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)


class UserResponse(UserBase):
    id: int
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[UserRole] = None


class LoginRequest(BaseModel):
    username: str
    password: str
