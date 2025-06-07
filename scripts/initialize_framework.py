#!/usr/bin/env python3
"""
Initialization script for the Community-Scale Digital Twin Framework
This script sets up the database, creates sample data, and initializes services
"""

import os
import sys
import asyncio
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.database import create_tables, get_db, Community, MeterData, WeatherData
from src.data.models import CommunityConfig, CommunityState, MacroTwinState
from src.utils.data_generator import SyntheticDataGenerator
from src.utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


class FrameworkInitializer:
    """Initialize the Digital Twin Framework"""
    
    def __init__(self):
        pass
    
    async def initialize_database(self):
        """Create database tables and setup initial schema"""
        logger.info("Initializing database...")
        
        try:
            # Create all tables
            create_tables()
            logger.info("Database tables created successfully")
            
            # Create sample communities
            await self.create_sample_communities()
            logger.info("Sample communities created successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    async def create_sample_communities(self):
        """Create sample communities for testing and demonstration"""
        logger.info("Creating sample communities...")
        
        communities = [
            {
                "id": "residential_suburb",
                "name": "Green Valley Residential Community",
                "num_prosumers": 150,
                "total_pv_capacity_kw": 750.0,
                "total_storage_capacity_kwh": 300.0,
                "max_storage_power_kw": 100.0,
                "grid_import_limit_kw": 500.0,
                "grid_export_limit_kw": 400.0,
                "incentive_budget_daily": 1000.0
            },
            {
                "id": "mixed_urban",
                "name": "Downtown Mixed-Use District",
                "num_prosumers": 75,
                "total_pv_capacity_kw": 600.0,
                "total_storage_capacity_kwh": 800.0,
                "max_storage_power_kw": 200.0,
                "grid_import_limit_kw": 800.0,
                "grid_export_limit_kw": 300.0,
                "incentive_budget_daily": 1500.0
            },
            {
                "id": "rural_cooperative",
                "name": "Countryside Energy Cooperative",
                "num_prosumers": 50,
                "total_pv_capacity_kw": 400.0,
                "total_storage_capacity_kwh": 600.0,
                "max_storage_power_kw": 150.0,
                "grid_import_limit_kw": 300.0,
                "grid_export_limit_kw": 250.0,
                "incentive_budget_daily": 800.0
            }
        ]
        
        db = next(get_db())
        try:
            for community_data in communities:
                existing = db.query(Community).filter(Community.id == community_data["id"]).first()
                if not existing:
                    community = Community(**community_data)
                    db.add(community)
            
            db.commit()
            logger.info(f"Sample communities ready")
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create sample communities: {e}")
            raise
        finally:
            db.close()

    async def generate_historical_data(self, 
                                     community_id: str, 
                                     days: int = 30,
                                     interval_minutes: int = 15):
        """Generate historical data for a community"""
        logger.info(f"Generating {days} days of historical data for {community_id}")
        
        try:
            # Get community config from database
            db = next(get_db())
            community = db.query(Community).filter(Community.id == community_id).first()
            db.close()
            
            if not community:
                logger.error(f"Community {community_id} not found")
                return
            
            # Initialize data generator with community parameters
            data_generator = SyntheticDataGenerator(
                community_id=community_id,
                num_prosumers=community.num_prosumers,
                pv_capacity_kw=community.total_pv_capacity_kw,
                storage_capacity_kwh=community.total_storage_capacity_kwh
            )
            
            # Generate time series data
            start_time = datetime.utcnow() - timedelta(days=days)
            end_time = datetime.utcnow()
            
            # Generate historical data using the data generator
            historical_df = data_generator.generate_historical_data(
                start_date=start_time,
                end_date=end_time,
                interval_minutes=interval_minutes
            )
            
            # Generate weather data
            weather_df = data_generator.generate_weather_data(
                start_date=start_time,
                end_date=end_time,
                interval_minutes=interval_minutes
            )
            
            # Store in database
            await self.store_historical_data(community_id, historical_df, weather_df)
            
            logger.info(f"Generated and stored {len(historical_df)} historical data points")
            
        except Exception as e:
            logger.error(f"Failed to generate historical data: {e}")
            raise

    async def store_historical_data(self, community_id: str, historical_df: pd.DataFrame, weather_df: pd.DataFrame):
        """Store historical data in the database"""
        db = next(get_db())
        
        try:
            meter_data_records = []
            weather_data_records = []
            
            # Process meter data
            for _, row in historical_df.iterrows():
                meter_record = MeterData(
                    community_id=community_id,
                    timestamp=row['timestamp'],
                    net_load_kw=row['net_load_kw'],
                    pv_generation_kw=row['pv_generation_kw'],
                    grid_import_kw=row['grid_import_kw'],
                    grid_export_kw=row['grid_export_kw'],
                    storage_soc_kwh=row['storage_soc_kwh'],
                    storage_charge_kw=row['storage_charge_kw'],
                    storage_discharge_kw=row['storage_discharge_kw']
                )
                meter_data_records.append(meter_record)
            
            # Process weather data
            for _, row in weather_df.iterrows():
                weather_record = WeatherData(
                    community_id=community_id,
                    timestamp=row['timestamp'],
                    irradiance_w_m2=row['irradiance_w_m2'],
                    temperature_c=row['temperature_c'],
                    humidity_percent=row['humidity_percent'],
                    wind_speed_m_s=row['wind_speed_m_s']
                )
                weather_data_records.append(weather_record)
            
            # Batch insert
            if meter_data_records:
                db.bulk_save_objects(meter_data_records)
            if weather_data_records:
                db.bulk_save_objects(weather_data_records)
            db.commit()
            
            logger.info(f"Stored {len(meter_data_records)} meter records and {len(weather_data_records)} weather records")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to store historical data: {e}")
            raise
        finally:
            db.close()

    async def setup_redis_cache(self):
        """Initialize Redis cache with default values"""
        logger.info("Setting up Redis cache...")
        
        try:
            import redis
            
            redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=int(os.getenv('REDIS_DB', 0)),
                decode_responses=True
            )
            
            # Test connection
            redis_client.ping()
            
            # Set default cache values
            default_values = {
                'system:status': 'initialized',
                'system:last_update': datetime.utcnow().isoformat(),
                'system:version': '1.0.0'
            }
            
            for key, value in default_values.items():
                redis_client.set(key, value)
            
            logger.info("Redis cache setup completed")
            
        except Exception as e:
            logger.warning(f"Redis not available: {e}")

    async def run_health_checks(self):
        """Run system health checks"""
        logger.info("Running system health checks...")
        
        health_status = {
            'database': False,
            'redis': False
        }
        
        # Check database connection
        try:
            db = next(get_db())
            db.execute("SELECT 1")
            health_status['database'] = True
            db.close()
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
        
        # Check Redis connection
        try:
            import redis
            redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379))
            )
            redis_client.ping()
            health_status['redis'] = True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
        
        # Log health status
        logger.info(f"Health check results: {health_status}")
        
        return health_status

    async def initialize_framework(self, 
                                 create_sample_data: bool = True,
                                 days_of_data: int = 30):
        """Complete framework initialization"""
        logger.info("Starting Digital Twin Framework initialization...")
        
        try:
            # Step 1: Initialize database
            await self.initialize_database()
            
            # Step 2: Setup Redis cache
            await self.setup_redis_cache()
            
            # Step 3: Generate sample data (optional)
            if create_sample_data:
                communities = ["residential_suburb", "mixed_urban", "rural_cooperative"]
                for community_id in communities:
                    await self.generate_historical_data(
                        community_id=community_id,
                        days=days_of_data
                    )
            
            # Step 4: Run health checks
            health_status = await self.run_health_checks()
            
            if health_status.get('database', False):
                logger.info("Framework initialization completed successfully!")
                return True
            else:
                logger.error("Framework initialization completed with errors")
                return False
            
        except Exception as e:
            logger.error(f"Framework initialization failed: {e}")
            return False


async def main():
    """Main initialization function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Digital Twin Framework")
    parser.add_argument("--no-sample-data", action="store_true",
                        help="Skip creation of sample data")
    parser.add_argument("--days", type=int, default=7,
                       help="Number of days of historical data to generate")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging with specified level
    setup_logging(log_level=args.log_level)
    
    # Initialize framework
    initializer = FrameworkInitializer()
    
    success = await initializer.initialize_framework(
        create_sample_data=not args.no_sample_data,
        days_of_data=args.days
    )
    
    if success:
        logger.info("Initialization completed successfully!")
        logger.info("You can now start the services")
        return 0
    else:
        logger.error("Initialization failed!")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))