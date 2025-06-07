from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./digital_twin.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Community(Base):
    __tablename__ = "communities"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    num_prosumers = Column(Integer, nullable=False)
    total_pv_capacity_kw = Column(Float, nullable=False)
    total_storage_capacity_kwh = Column(Float, nullable=False)
    max_storage_power_kw = Column(Float, nullable=False)
    grid_import_limit_kw = Column(Float, nullable=False)
    grid_export_limit_kw = Column(Float, nullable=False)
    incentive_budget_daily = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    meter_data = relationship("MeterData", back_populates="community")
    weather_data = relationship("WeatherData", back_populates="community")


class MeterData(Base):
    __tablename__ = "meter_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    community_id = Column(String, ForeignKey("communities.id"))
    timestamp = Column(DateTime, nullable=False)
    net_load_kw = Column(Float, nullable=False)
    pv_generation_kw = Column(Float, nullable=False)
    grid_import_kw = Column(Float, nullable=False)
    grid_export_kw = Column(Float, nullable=False)
    storage_soc_kwh = Column(Float, nullable=False)
    storage_charge_kw = Column(Float, nullable=False)
    storage_discharge_kw = Column(Float, nullable=False)
    
    community = relationship("Community", back_populates="meter_data")


class WeatherData(Base):
    __tablename__ = "weather_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    community_id = Column(String, ForeignKey("communities.id"))
    timestamp = Column(DateTime, nullable=False)
    irradiance_w_m2 = Column(Float, nullable=False)
    temperature_c = Column(Float, nullable=False)
    humidity_percent = Column(Float, nullable=False)
    wind_speed_m_s = Column(Float, nullable=False)
    
    community = relationship("Community", back_populates="weather_data")


class ForecastModel(Base):
    __tablename__ = "forecast_models"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    community_id = Column(String, ForeignKey("communities.id"))
    variable_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    model_path = Column(String, nullable=False)
    training_date = Column(DateTime, default=datetime.utcnow)
    performance_metrics = Column(JSON)
    is_active = Column(Boolean, default=True)


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    community_id = Column(String, ForeignKey("communities.id"))
    objective = Column(String, nullable=False)
    horizon_hours = Column(Integer, nullable=False)
    objective_value = Column(Float, nullable=False)
    execution_time_ms = Column(Float, nullable=False)
    results = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class SimulationScenario(Base):
    __tablename__ = "simulation_scenarios"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    scenario_id = Column(String, unique=True, nullable=False)
    community_id = Column(String, ForeignKey("communities.id"))
    name = Column(String, nullable=False)
    description = Column(String)
    parameter_changes = Column(JSON)
    simulation_duration_days = Column(Integer, nullable=False)
    results = Column(JSON)
    execution_time_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class MacroTwinStateDB(Base):
    __tablename__ = "macro_twin_states"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    community_id = Column(String, ForeignKey("communities.id"))
    state_vector = Column(JSON, nullable=False)
    covariance_matrix = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    prediction_horizon = Column(Integer, default=24)


# Authentication and User Tables
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    full_name = Column(String(100))
    hashed_password = Column(String(128), nullable=False)
    role = Column(String(20), nullable=False, default="viewer")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    api_keys = relationship("APIKey", back_populates="user")


class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    key_hash = Column(String(128), nullable=False)
    permissions = Column(JSON)  # List of permissions
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime)
    
    user = relationship("User", back_populates="api_keys")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    Base.metadata.create_all(bind=engine)
