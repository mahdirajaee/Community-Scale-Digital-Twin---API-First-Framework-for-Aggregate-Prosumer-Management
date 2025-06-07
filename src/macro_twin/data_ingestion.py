import asyncio
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import logging

from .core import MacroTwinCore
from ..data.models import CommunityState, CommunityConfig, CommunityMetrics
from ..data.database import get_db, MeterData, WeatherData

logger = logging.getLogger(__name__)


class DataIngestionEngine:
    
    def __init__(self, community_config: CommunityConfig):
        self.community_config = community_config
        self.macro_twin = MacroTwinCore(community_config)
        self.data_buffer = []
        self.max_buffer_size = 100
        self.ingestion_active = False
        
    async def start_real_time_ingestion(self):
        
        self.ingestion_active = True
        logger.info(f"Starting real-time data ingestion for community {self.community_config.community_id}")
        
        while self.ingestion_active:
            try:
                await self._ingest_meter_data()
                await self._ingest_weather_data()
                await self._process_data_buffer()
                await asyncio.sleep(15)  
            except Exception as e:
                logger.error(f"Error in data ingestion: {e}")
                await asyncio.sleep(60)  
                
    async def stop_real_time_ingestion(self):
        
        self.ingestion_active = False
        logger.info("Stopping real-time data ingestion")
        
    async def _ingest_meter_data(self):
        
        db = next(get_db())
        try:
            latest_data = db.query(MeterData)\
                .filter(MeterData.community_id == self.community_config.community_id)\
                .order_by(MeterData.timestamp.desc())\
                .first()
                
            if latest_data:
                community_state = CommunityState(
                    timestamp=latest_data.timestamp,
                    net_load_kw=latest_data.net_load_kw,
                    pv_generation_kw=latest_data.pv_generation_kw,
                    storage_soc_kwh=latest_data.storage_soc_kwh,
                    grid_import_kw=latest_data.grid_import_kw,
                    grid_export_kw=latest_data.grid_export_kw,
                    ambient_temperature_c=20.0,  
                    irradiance_w_m2=0.0  
                )
                
                self.data_buffer.append(community_state)
                
        except Exception as e:
            logger.error(f"Error ingesting meter data: {e}")
        finally:
            db.close()
            
    async def _ingest_weather_data(self):
        
        db = next(get_db())
        try:
            latest_weather = db.query(WeatherData)\
                .filter(WeatherData.community_id == self.community_config.community_id)\
                .order_by(WeatherData.timestamp.desc())\
                .first()
                
            if latest_weather and self.data_buffer:
                self.data_buffer[-1].ambient_temperature_c = latest_weather.temperature_c
                self.data_buffer[-1].irradiance_w_m2 = latest_weather.irradiance_w_m2
                
        except Exception as e:
            logger.error(f"Error ingesting weather data: {e}")
        finally:
            db.close()
            
    async def _process_data_buffer(self):
        
        if not self.data_buffer:
            return
            
        try:
            db = next(get_db())
            
            for state in self.data_buffer:
                macro_twin_state = self.macro_twin.update_state(state)
                
                if not self.macro_twin.validate_physical_constraints():
                    logger.warning(f"Physical constraints violated at {state.timestamp}")
                    
                self.macro_twin.save_state(db)
                
            self.data_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error processing data buffer: {e}")
        finally:
            db.close()
            
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer = self.data_buffer[-self.max_buffer_size//2:]
            
    def ingest_historical_data(self, start_date: datetime, end_date: datetime):
        
        db = next(get_db())
        try:
            meter_data = db.query(MeterData)\
                .filter(MeterData.community_id == self.community_config.community_id)\
                .filter(MeterData.timestamp >= start_date)\
                .filter(MeterData.timestamp <= end_date)\
                .order_by(MeterData.timestamp)\
                .all()
                
            weather_data = db.query(WeatherData)\
                .filter(WeatherData.community_id == self.community_config.community_id)\
                .filter(WeatherData.timestamp >= start_date)\
                .filter(WeatherData.timestamp <= end_date)\
                .order_by(WeatherData.timestamp)\
                .all()
                
            weather_dict = {wd.timestamp: wd for wd in weather_data}
            
            states_processed = 0
            for meter_point in meter_data:
                weather_point = weather_dict.get(meter_point.timestamp)
                
                community_state = CommunityState(
                    timestamp=meter_point.timestamp,
                    net_load_kw=meter_point.net_load_kw,
                    pv_generation_kw=meter_point.pv_generation_kw,
                    storage_soc_kwh=meter_point.storage_soc_kwh,
                    grid_import_kw=meter_point.grid_import_kw,
                    grid_export_kw=meter_point.grid_export_kw,
                    ambient_temperature_c=weather_point.temperature_c if weather_point else 20.0,
                    irradiance_w_m2=weather_point.irradiance_w_m2 if weather_point else 0.0
                )
                
                self.macro_twin.update_state(community_state)
                states_processed += 1
                
                if states_processed % 100 == 0:
                    self.macro_twin.save_state(db)
                    
            self.macro_twin.save_state(db)
            logger.info(f"Processed {states_processed} historical data points")
            
        except Exception as e:
            logger.error(f"Error processing historical data: {e}")
        finally:
            db.close()
            
    def get_current_state(self) -> Optional[Dict]:
        
        return self.macro_twin.get_aggregate_metrics()
        
    def get_state_predictions(self, steps_ahead: int = 24) -> np.ndarray:
        
        return self.macro_twin.predict_state(steps_ahead)
