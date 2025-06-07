from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time
import logging
import asyncio

from .auth import (
    SecurityManager, security_manager, get_current_user, get_current_active_user,
    require_role, APIKeyManager, SecurityAuditLogger, rate_limit
)
from ..data.models import (
    APIResponse, CommunityConfig, CommunityMetrics, ForecastRequest, 
    ForecastResult, OptimizationRequest, OptimizationResult, 
    ScenarioDefinition, ScenarioResult, UserCreate, UserResponse, Token, LoginRequest, UserRole
)
from ..data.database import get_db, create_tables, Community
from ..macro_twin.core import MacroTwinCore
from ..macro_twin.data_ingestion import DataIngestionEngine
from ..forecasting.models import ForecastingService
from ..optimization.engine import OptimizationEngine
from ..simulation.engine import ScenarioSimulator, ParallelSimulationEngine
from .auth import (
    SecurityManager, security_manager, get_current_user, get_current_active_user,
    require_role, APIKeyManager, SecurityAuditLogger, rate_limit
)
from ..data.models import UserCreate, UserResponse, Token, LoginRequest, UserRole
from ..data.database import User

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Community Digital Twin API",
    description="API-First Framework for Aggregate Prosumer Management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

create_tables()

active_ingestion_engines: Dict[str, DataIngestionEngine] = {}


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Community Digital Twin API")


@app.on_event("shutdown")
async def shutdown_event():
    for engine in active_ingestion_engines.values():
        await engine.stop_real_time_ingestion()
    logger.info("Shutting down Community Digital Twin API")


def create_api_response(success: bool, message: str, data: Any = None, 
                       execution_time: float = 0) -> APIResponse:
    return APIResponse(
        success=success,
        message=message,
        data=data,
        timestamp=datetime.utcnow(),
        execution_time_ms=execution_time
    )


@app.get("/", response_model=APIResponse)
async def root():
    return create_api_response(
        success=True,
        message="Community Digital Twin API is running",
        data={
            "version": "1.0.0",
            "status": "operational",
            "endpoints": [
                "/communities",
                "/macro-twin",
                "/forecasting",
                "/optimization", 
                "/simulation"
            ]
        }
    )


@app.post("/communities", response_model=APIResponse)
async def create_community(
    community_config: CommunityConfig, 
    db: Session = Depends(get_db)
):
    start_time = time.time()
    
    try:
        existing_community = db.query(Community).filter(
            Community.id == community_config.community_id
        ).first()
        
        if existing_community:
            raise HTTPException(
                status_code=400, 
                detail=f"Community {community_config.community_id} already exists"
            )
        
        community = Community(
            id=community_config.community_id,
            name=f"Community {community_config.community_id}",
            num_prosumers=community_config.num_prosumers,
            total_pv_capacity_kw=community_config.total_pv_capacity_kw,
            total_storage_capacity_kwh=community_config.total_storage_capacity_kwh,
            max_storage_power_kw=community_config.max_storage_power_kw,
            grid_import_limit_kw=community_config.grid_import_limit_kw,
            grid_export_limit_kw=community_config.grid_export_limit_kw,
            incentive_budget_daily=community_config.incentive_budget_daily
        )
        
        db.add(community)
        db.commit()
        db.refresh(community)
        
        execution_time = (time.time() - start_time) * 1000
        
        return create_api_response(
            success=True,
            message=f"Community {community_config.community_id} created successfully",
            data={"community_id": community.id},
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error creating community: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/communities", response_model=APIResponse)
async def list_communities(db: Session = Depends(get_db)):
    start_time = time.time()
    
    try:
        communities = db.query(Community).all()
        
        community_list = [
            {
                "id": c.id,
                "name": c.name,
                "num_prosumers": c.num_prosumers,
                "total_pv_capacity_kw": c.total_pv_capacity_kw,
                "total_storage_capacity_kwh": c.total_storage_capacity_kwh,
                "created_at": c.created_at
            }
            for c in communities
        ]
        
        execution_time = (time.time() - start_time) * 1000
        
        return create_api_response(
            success=True,
            message=f"Found {len(community_list)} communities",
            data={"communities": community_list},
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error listing communities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/communities/{community_id}/macro-twin/start", response_model=APIResponse)
async def start_macro_twin(
    community_id: str, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    start_time = time.time()
    
    try:
        community = db.query(Community).filter(Community.id == community_id).first()
        if not community:
            raise HTTPException(status_code=404, detail="Community not found")
        
        community_config = CommunityConfig(
            community_id=community.id,
            num_prosumers=community.num_prosumers,
            total_pv_capacity_kw=community.total_pv_capacity_kw,
            total_storage_capacity_kwh=community.total_storage_capacity_kwh,
            max_storage_power_kw=community.max_storage_power_kw,
            grid_import_limit_kw=community.grid_import_limit_kw,
            grid_export_limit_kw=community.grid_export_limit_kw,
            incentive_budget_daily=community.incentive_budget_daily
        )
        
        if community_id in active_ingestion_engines:
            await active_ingestion_engines[community_id].stop_real_time_ingestion()
        
        ingestion_engine = DataIngestionEngine(community_config)
        active_ingestion_engines[community_id] = ingestion_engine
        
        background_tasks.add_task(ingestion_engine.start_real_time_ingestion)
        
        execution_time = (time.time() - start_time) * 1000
        
        return create_api_response(
            success=True,
            message=f"Macro-twin started for community {community_id}",
            data={"status": "running"},
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error starting macro-twin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/communities/{community_id}/macro-twin/state", response_model=APIResponse)
async def get_macro_twin_state(community_id: str):
    start_time = time.time()
    
    try:
        if community_id not in active_ingestion_engines:
            raise HTTPException(
                status_code=404, 
                detail="Macro-twin not running for this community"
            )
        
        engine = active_ingestion_engines[community_id]
        current_state = engine.get_current_state()
        
        if not current_state:
            raise HTTPException(
                status_code=404, 
                detail="No state data available"
            )
        
        execution_time = (time.time() - start_time) * 1000
        
        return create_api_response(
            success=True,
            message="Current macro-twin state retrieved",
            data=current_state,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error getting macro-twin state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/communities/{community_id}/forecasting/generate", response_model=APIResponse)
async def generate_forecast(
    community_id: str,
    request: ForecastRequest
):
    start_time = time.time()
    
    try:
        forecasting_service = ForecastingService(community_id)
        forecasts = await forecasting_service.generate_forecast(request)
        
        execution_time = (time.time() - start_time) * 1000
        
        return create_api_response(
            success=True,
            message=f"Generated forecasts for {len(forecasts)} variables",
            data={"forecasts": [f.dict() for f in forecasts]},
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/communities/{community_id}/forecasting/train", response_model=APIResponse)
async def train_forecasting_models(
    community_id: str,
    retrain_days: int = 90
):
    start_time = time.time()
    
    try:
        forecasting_service = ForecastingService(community_id)
        training_results = forecasting_service.train_models(retrain_days)
        
        execution_time = (time.time() - start_time) * 1000
        
        return create_api_response(
            success=True,
            message="Forecasting models trained successfully",
            data={"training_results": training_results},
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error training forecasting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/communities/{community_id}/optimization/optimize", response_model=APIResponse)
async def optimize_dispatch_and_tariff(
    community_id: str,
    request: OptimizationRequest,
    db: Session = Depends(get_db)
):
    start_time = time.time()
    
    try:
        community = db.query(Community).filter(Community.id == community_id).first()
        if not community:
            raise HTTPException(status_code=404, detail="Community not found")
        
        community_config = CommunityConfig(
            community_id=community.id,
            num_prosumers=community.num_prosumers,
            total_pv_capacity_kw=community.total_pv_capacity_kw,
            total_storage_capacity_kwh=community.total_storage_capacity_kwh,
            max_storage_power_kw=community.max_storage_power_kw,
            grid_import_limit_kw=community.grid_import_limit_kw,
            grid_export_limit_kw=community.grid_export_limit_kw,
            incentive_budget_daily=community.incentive_budget_daily
        )
        
        optimization_engine = OptimizationEngine(community_config)
        result = optimization_engine.optimize_dispatch_and_tariff(request)
        
        execution_time = (time.time() - start_time) * 1000
        
        return create_api_response(
            success=True,
            message="Optimization completed successfully",
            data=result.dict(),
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/communities/{community_id}/simulation/run-scenario", response_model=APIResponse)
async def run_scenario_simulation(
    community_id: str,
    scenario: ScenarioDefinition,
    db: Session = Depends(get_db)
):
    start_time = time.time()
    
    try:
        community = db.query(Community).filter(Community.id == community_id).first()
        if not community:
            raise HTTPException(status_code=404, detail="Community not found")
        
        community_config = CommunityConfig(
            community_id=community.id,
            num_prosumers=community.num_prosumers,
            total_pv_capacity_kw=community.total_pv_capacity_kw,
            total_storage_capacity_kwh=community.total_storage_capacity_kwh,
            max_storage_power_kw=community.max_storage_power_kw,
            grid_import_limit_kw=community.grid_import_limit_kw,
            grid_export_limit_kw=community.grid_export_limit_kw,
            incentive_budget_daily=community.incentive_budget_daily
        )
        
        simulator = ScenarioSimulator(community_config)
        result = await simulator.run_scenario_simulation(scenario)
        
        execution_time = (time.time() - start_time) * 1000
        
        return create_api_response(
            success=True,
            message=f"Scenario {scenario.scenario_id} simulation completed",
            data=result.dict(),
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error in scenario simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/communities/{community_id}/simulation/run-multiple", response_model=APIResponse)
async def run_multiple_scenarios(
    community_id: str,
    scenarios: List[ScenarioDefinition],
    db: Session = Depends(get_db)
):
    start_time = time.time()
    
    try:
        community = db.query(Community).filter(Community.id == community_id).first()
        if not community:
            raise HTTPException(status_code=404, detail="Community not found")
        
        community_config = CommunityConfig(
            community_id=community.id,
            num_prosumers=community.num_prosumers,
            total_pv_capacity_kw=community.total_pv_capacity_kw,
            total_storage_capacity_kwh=community.total_storage_capacity_kwh,
            max_storage_power_kw=community.max_storage_power_kw,
            grid_import_limit_kw=community.grid_import_limit_kw,
            grid_export_limit_kw=community.grid_export_limit_kw,
            incentive_budget_daily=community.incentive_budget_daily
        )
        
        parallel_engine = ParallelSimulationEngine(community_config)
        results = await parallel_engine.run_multiple_scenarios(scenarios)
        comparison = await parallel_engine.compare_scenarios(results)
        
        execution_time = (time.time() - start_time) * 1000
        
        return create_api_response(
            success=True,
            message=f"Completed {len(results)} scenario simulations",
            data={
                "results": [r.dict() for r in results],
                "comparison": comparison
            },
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error in multiple scenario simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/communities/{community_id}/analytics/kpis", response_model=APIResponse)
async def get_community_kpis(
    community_id: str,
    days_back: int = 30,
    db: Session = Depends(get_db)
):
    start_time = time.time()
    
    try:
        from ..data.database import MeterData
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        meter_data = db.query(MeterData)\
            .filter(MeterData.community_id == community_id)\
            .filter(MeterData.timestamp >= start_date)\
            .filter(MeterData.timestamp <= end_date)\
            .all()
        
        if not meter_data:
            raise HTTPException(status_code=404, detail="No data found for the specified period")
        
        total_load = sum(m.net_load_kw for m in meter_data) * 0.25
        total_pv = sum(m.pv_generation_kw for m in meter_data) * 0.25
        total_import = sum(m.grid_import_kw for m in meter_data) * 0.25
        total_export = sum(m.grid_export_kw for m in meter_data) * 0.25
        
        peak_demand = max(m.grid_import_kw for m in meter_data)
        avg_demand = sum(m.grid_import_kw for m in meter_data) / len(meter_data)
        
        self_consumption_rate = (total_pv - total_export) / total_pv if total_pv > 0 else 0
        self_sufficiency_rate = 1 - (total_import / total_load) if total_load > 0 else 0
        
        kpis = {
            "period_days": days_back,
            "total_energy_consumed_kwh": total_load,
            "total_pv_generated_kwh": total_pv,
            "total_grid_import_kwh": total_import,
            "total_grid_export_kwh": total_export,
            "peak_demand_kw": peak_demand,
            "average_demand_kw": avg_demand,
            "self_consumption_rate": self_consumption_rate,
            "self_sufficiency_rate": self_sufficiency_rate,
            "data_points": len(meter_data)
        }
        
        execution_time = (time.time() - start_time) * 1000
        
        return create_api_response(
            success=True,
            message=f"KPIs calculated for {days_back} days",
            data=kpis,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error calculating KPIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Authentication Endpoints
@app.post("/auth/register", response_model=APIResponse)
async def register_user(
    user_create: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user"""
    start_time = time.time()
    
    try:
        user = security_manager.create_user(db, user_create)
        
        execution_time = (time.time() - start_time) * 1000
        
        return create_api_response(
            success=True,
            message="User registered successfully",
            data={
                "user_id": user.id,
                "username": user.username,
                "role": user.role
            },
            execution_time=execution_time
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/auth/login", response_model=APIResponse)
@rate_limit(requests_per_minute=10)  # Limit login attempts
async def login(
    request: Request,
    login_request: LoginRequest,
    db: Session = Depends(get_db)
):
    """Authenticate user and return access token"""
    start_time = time.time()
    client_ip = request.client.host
    
    try:
        user = security_manager.authenticate_user(
            db, login_request.username, login_request.password
        )
        
        if not user:
            SecurityAuditLogger.log_login_attempt(
                login_request.username, False, client_ip
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        access_token = security_manager.create_access_token(
            data={"sub": user.username, "role": user.role}
        )
        
        SecurityAuditLogger.log_login_attempt(
            login_request.username, True, client_ip
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        return create_api_response(
            success=True,
            message="Login successful",
            data={
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": security_manager.access_token_expire_minutes * 60,
                "user": {
                    "username": user.username,
                    "role": user.role,
                    "full_name": user.full_name
                }
            },
            execution_time=execution_time
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/auth/me", response_model=APIResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information"""
    return create_api_response(
        success=True,
        message="User information retrieved",
        data={
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "full_name": current_user.full_name,
            "role": current_user.role,
            "is_active": current_user.is_active,
            "created_at": current_user.created_at,
            "last_login": current_user.last_login
        }
    )


@app.post("/auth/api-keys", response_model=APIResponse)
async def create_api_key(
    name: str,
    permissions: List[str],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new API key (Admin only)"""
    if current_user.role != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can create API keys"
        )
    
    api_key = APIKeyManager.create_api_key(db, name, permissions)
    
    return create_api_response(
        success=True,
        message="API key created successfully",
        data={
            "api_key": api_key.raw_key,  # Only shown once
            "name": api_key.name,
            "permissions": api_key.permissions,
            "created_at": api_key.created_at
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
