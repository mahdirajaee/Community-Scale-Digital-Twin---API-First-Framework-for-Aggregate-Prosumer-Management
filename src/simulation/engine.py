import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import concurrent.futures
import logging
from dataclasses import dataclass
import json

from ..data.models import (
    ScenarioDefinition, ScenarioResult, CommunityConfig, 
    TimeSeriesDataPoint, OptimizationRequest, OptimizationObjective,
    TariffStructure
)
from ..macro_twin.core import MacroTwinCore
from ..forecasting.models import ForecastingService
from ..optimization.engine import OptimizationEngine
from ..data.database import get_db

logger = logging.getLogger(__name__)


@dataclass
class SimulationParameters:
    scenario_id: str
    duration_days: int
    time_step_minutes: int = 15
    monte_carlo_runs: int = 100
    include_uncertainty: bool = True


class ScenarioSimulator:
    
    def __init__(self, community_config: CommunityConfig):
        self.community_config = community_config
        self.macro_twin = MacroTwinCore(community_config)
        self.forecasting_service = ForecastingService(community_config.community_id)
        self.optimization_engine = OptimizationEngine(community_config)
        
    async def run_scenario_simulation(self, scenario: ScenarioDefinition) -> ScenarioResult:
        
        start_time = datetime.utcnow()
        
        try:
            modified_config = self._apply_scenario_parameters(scenario)
            
            simulation_results = await self._execute_simulation(
                scenario, modified_config
            )
            
            kpis = self._calculate_kpis(simulation_results)
            time_series_results = self._extract_time_series(simulation_results)
            summary_stats = self._calculate_summary_statistics(simulation_results)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return ScenarioResult(
                scenario_id=scenario.scenario_id,
                kpis=kpis,
                time_series_results=time_series_results,
                summary_stats=summary_stats
            )
            
        except Exception as e:
            logger.error(f"Scenario simulation failed: {e}")
            raise
            
    def _apply_scenario_parameters(self, scenario: ScenarioDefinition) -> CommunityConfig:
        
        config_dict = self.community_config.dict()
        
        for param_name, param_value in scenario.parameter_changes.items():
            if param_name in config_dict:
                config_dict[param_name] = param_value
            else:
                logger.warning(f"Unknown parameter: {param_name}")
                
        return CommunityConfig(**config_dict)
        
    async def _execute_simulation(self, scenario: ScenarioDefinition, 
                                 config: CommunityConfig) -> Dict[str, Any]:
        
        simulation_steps = scenario.simulation_duration_days * 24 * 4  
        
        macro_twin = MacroTwinCore(config)
        
        db = next(get_db())
        try:
            macro_twin.load_state(db)
        finally:
            db.close()
            
        results = {
            'timestamps': [],
            'net_load': [],
            'pv_generation': [],
            'storage_soc': [],
            'grid_import': [],
            'grid_export': [],
            'battery_dispatch': [],
            'costs': [],
            'peak_demands': [],
            'self_consumption_rates': []
        }
        
        current_time = datetime.utcnow()
        
        for step in range(simulation_steps):
            step_time = current_time + timedelta(minutes=15 * step)
            
            forecasted_state = macro_twin.predict_state(1)[0]
            
            net_load = forecasted_state[0] + np.random.normal(0, 0.1)
            pv_generation = max(0, forecasted_state[2] + np.random.normal(0, 0.05))
            storage_soc = max(0, min(config.total_storage_capacity_kwh, 
                                   forecasted_state[4] + np.random.normal(0, 0.02)))
            
            if step % 96 == 0:  
                try:
                    optimization_request = OptimizationRequest(
                        objective=OptimizationObjective.MAXIMIZE_WELFARE,
                        horizon_hours=24,
                        current_soc_kwh=storage_soc,
                        forecasted_load=[net_load] * 96,
                        forecasted_pv=[pv_generation] * 96,
                        current_tariff=TariffStructure(
                            time_of_use_rates={"peak": 0.3, "off_peak": 0.15},
                            feed_in_tariff=0.08,
                            demand_charge=10.0,
                            fixed_charge_daily=5.0
                        )
                    )
                    
                    optimization_result = self.optimization_engine.optimize_dispatch_and_tariff(
                        optimization_request
                    )
                    
                    daily_battery_dispatch = optimization_result.battery_dispatch
                    daily_costs = optimization_result.cost_savings
                    
                except Exception as e:
                    logger.warning(f"Optimization failed at step {step}: {e}")
                    daily_battery_dispatch = [0] * 96
                    daily_costs = 0
                    
            step_in_day = step % 96
            battery_dispatch = daily_battery_dispatch[step_in_day] if step_in_day < len(daily_battery_dispatch) else 0
            
            grid_import = max(0, net_load - pv_generation + battery_dispatch)
            grid_export = max(0, pv_generation - net_load - battery_dispatch)
            
            peak_demand = max(results['grid_import'][-24:] + [grid_import]) if len(results['grid_import']) >= 24 else grid_import
            
            if pv_generation > 0:
                self_consumption = min(net_load, pv_generation) / pv_generation
            else:
                self_consumption = 0
                
            results['timestamps'].append(step_time)
            results['net_load'].append(net_load)
            results['pv_generation'].append(pv_generation)
            results['storage_soc'].append(storage_soc)
            results['grid_import'].append(grid_import)
            results['grid_export'].append(grid_export)
            results['battery_dispatch'].append(battery_dispatch)
            results['costs'].append(daily_costs / 96 if 'daily_costs' in locals() else 0)
            results['peak_demands'].append(peak_demand)
            results['self_consumption_rates'].append(self_consumption)
            
            if step % 100 == 0:
                logger.info(f"Simulation progress: {step}/{simulation_steps} steps completed")
                
        return results
        
    def _calculate_kpis(self, simulation_results: Dict[str, Any]) -> Dict[str, float]:
        
        kpis = {}
        
        total_load = sum(simulation_results['net_load'])
        total_pv = sum(simulation_results['pv_generation'])
        total_grid_import = sum(simulation_results['grid_import'])
        total_grid_export = sum(simulation_results['grid_export'])
        
        kpis['total_energy_consumed_kwh'] = total_load * 0.25  
        kpis['total_pv_generated_kwh'] = total_pv * 0.25
        kpis['total_grid_import_kwh'] = total_grid_import * 0.25
        kpis['total_grid_export_kwh'] = total_grid_export * 0.25
        
        kpis['peak_demand_kw'] = max(simulation_results['grid_import'])
        kpis['average_demand_kw'] = np.mean(simulation_results['grid_import'])
        
        if total_pv > 0:
            kpis['self_consumption_rate'] = np.mean(simulation_results['self_consumption_rates'])
        else:
            kpis['self_consumption_rate'] = 0
            
        if total_load > 0:
            kpis['self_sufficiency_rate'] = 1 - (total_grid_import / total_load)
        else:
            kpis['self_sufficiency_rate'] = 0
            
        kpis['total_cost_savings'] = sum(simulation_results['costs'])
        
        storage_cycles = self._calculate_storage_cycles(simulation_results['storage_soc'])
        kpis['storage_cycle_count'] = storage_cycles
        
        kpis['grid_export_revenue'] = total_grid_export * 0.08 * 0.25  
        
        kpis['peak_reduction_percentage'] = self._calculate_peak_reduction_percentage(simulation_results)
        
        return kpis
        
    def _extract_time_series(self, simulation_results: Dict[str, Any]) -> Dict[str, List[TimeSeriesDataPoint]]:
        
        time_series = {}
        
        variables = ['net_load', 'pv_generation', 'storage_soc', 'grid_import', 'grid_export']
        units = {'net_load': 'kW', 'pv_generation': 'kW', 'storage_soc': 'kWh', 
                'grid_import': 'kW', 'grid_export': 'kW'}
        
        for var in variables:
            time_series[var] = [
                TimeSeriesDataPoint(
                    timestamp=ts, 
                    value=val, 
                    unit=units[var]
                ) 
                for ts, val in zip(simulation_results['timestamps'], simulation_results[var])
            ]
            
        return time_series
        
    def _calculate_summary_statistics(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        
        stats = {}
        
        for variable in ['net_load', 'pv_generation', 'storage_soc', 'grid_import', 'grid_export']:
            data = np.array(simulation_results[variable])
            stats[variable] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'p25': float(np.percentile(data, 25)),
                'p50': float(np.percentile(data, 50)),
                'p75': float(np.percentile(data, 75)),
                'p95': float(np.percentile(data, 95))
            }
            
        return stats
        
    def _calculate_storage_cycles(self, soc_data: List[float]) -> float:
        
        if len(soc_data) < 2:
            return 0
            
        cycles = 0
        direction = 0  
        
        for i in range(1, len(soc_data)):
            delta = soc_data[i] - soc_data[i-1]
            
            if abs(delta) < 0.01:  
                continue
                
            if delta > 0 and direction <= 0:  
                direction = 1
            elif delta < 0 and direction >= 0:  
                if direction == 1:
                    cycles += 0.5  
                direction = -1
                
        return cycles
        
    def _calculate_peak_reduction_percentage(self, simulation_results: Dict[str, Any]) -> float:
        
        baseline_peak = max(simulation_results['net_load'])
        optimized_peak = max(simulation_results['grid_import'])
        
        if baseline_peak > 0:
            return ((baseline_peak - optimized_peak) / baseline_peak) * 100
        return 0


class ParallelSimulationEngine:
    
    def __init__(self, community_config: CommunityConfig, max_workers: int = 4):
        self.community_config = community_config
        self.max_workers = max_workers
        
    async def run_multiple_scenarios(self, scenarios: List[ScenarioDefinition]) -> List[ScenarioResult]:
        
        logger.info(f"Running {len(scenarios)} scenarios in parallel")
        
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = []
            
            for scenario in scenarios:
                simulator = ScenarioSimulator(self.community_config)
                task = loop.run_in_executor(
                    executor, 
                    self._run_scenario_sync, 
                    simulator, 
                    scenario
                )
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Scenario {scenarios[i].scenario_id} failed: {result}")
            else:
                successful_results.append(result)
                
        return successful_results
        
    def _run_scenario_sync(self, simulator: ScenarioSimulator, scenario: ScenarioDefinition) -> ScenarioResult:
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(simulator.run_scenario_simulation(scenario))
        finally:
            loop.close()
            
    async def compare_scenarios(self, results: List[ScenarioResult]) -> Dict[str, Any]:
        
        if not results:
            return {}
            
        comparison = {
            'scenario_count': len(results),
            'kpi_comparison': {},
            'ranking': {},
            'summary': {}
        }
        
        kpi_names = list(results[0].kpis.keys())
        
        for kpi in kpi_names:
            values = [result.kpis[kpi] for result in results]
            scenario_ids = [result.scenario_id for result in results]
            
            comparison['kpi_comparison'][kpi] = {
                'values': dict(zip(scenario_ids, values)),
                'best_scenario': scenario_ids[np.argmax(values) if 'cost' not in kpi else np.argmin(values)],
                'worst_scenario': scenario_ids[np.argmin(values) if 'cost' not in kpi else np.argmax(values)],
                'mean': np.mean(values),
                'std': np.std(values)
            }
            
        primary_kpis = ['total_cost_savings', 'self_consumption_rate', 'peak_reduction_percentage']
        
        for kpi in primary_kpis:
            if kpi in comparison['kpi_comparison']:
                values = list(comparison['kpi_comparison'][kpi]['values'].values())
                scenario_ids = list(comparison['kpi_comparison'][kpi]['values'].keys())
                
                if 'cost' in kpi:
                    sorted_indices = np.argsort(values)
                else:
                    sorted_indices = np.argsort(values)[::-1]
                    
                comparison['ranking'][kpi] = [scenario_ids[i] for i in sorted_indices]
                
        comparison['summary']['best_overall'] = self._find_best_overall_scenario(results)
        comparison['summary']['total_execution_time'] = sum(
            result.summary_stats.get('execution_time_ms', 0) for result in results
        )
        
        return comparison
        
    def _find_best_overall_scenario(self, results: List[ScenarioResult]) -> str:
        
        scores = {}
        
        for result in results:
            score = 0
            score += result.kpis.get('total_cost_savings', 0) * 0.3
            score += result.kpis.get('self_consumption_rate', 0) * 100 * 0.3
            score += result.kpis.get('peak_reduction_percentage', 0) * 0.4
            
            scores[result.scenario_id] = score
            
        return max(scores, key=scores.get)
