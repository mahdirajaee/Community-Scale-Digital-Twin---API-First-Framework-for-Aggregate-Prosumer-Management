import numpy as np
import cvxpy as cp
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

from ..data.models import (
    OptimizationRequest, OptimizationResult, OptimizationObjective,
    TariffStructure, CommunityConfig
)

logger = logging.getLogger(__name__)


class OptimizationEngine:
    
    def __init__(self, community_config: CommunityConfig):
        self.community_config = community_config
        
    def optimize_dispatch_and_tariff(self, request: OptimizationRequest) -> OptimizationResult:
        
        try:
            if request.objective == OptimizationObjective.MAXIMIZE_WELFARE:
                return self._optimize_welfare_maximization(request)
            elif request.objective == OptimizationObjective.MINIMIZE_COST:
                return self._optimize_cost_minimization(request)
            elif request.objective == OptimizationObjective.MINIMIZE_PEAK:
                return self._optimize_peak_minimization(request)
            elif request.objective == OptimizationObjective.MAXIMIZE_SELF_CONSUMPTION:
                return self._optimize_self_consumption(request)
            else:
                raise ValueError(f"Unknown optimization objective: {request.objective}")
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
            
    def _optimize_welfare_maximization(self, request: OptimizationRequest) -> OptimizationResult:
        
        T = request.horizon_hours * 4  
        
        battery_charge = cp.Variable(T, nonneg=True)
        battery_discharge = cp.Variable(T, nonneg=True)
        grid_import = cp.Variable(T, nonneg=True)
        grid_export = cp.Variable(T, nonneg=True)
        soc = cp.Variable(T + 1, nonneg=True)
        
        tou_rate_peak = cp.Variable(nonneg=True)
        tou_rate_offpeak = cp.Variable(nonneg=True)
        feed_in_rate = cp.Variable(nonneg=True)
        
        forecasted_load = np.array(request.forecasted_load[:T])
        forecasted_pv = np.array(request.forecasted_pv[:T])
        
        dt = 0.25  
        battery_efficiency = 0.95
        
        constraints = []
        
        constraints.append(soc[0] == request.current_soc_kwh)
        for t in range(T):
            constraints.append(
                soc[t + 1] == soc[t] + dt * (battery_efficiency * battery_charge[t] - battery_discharge[t] / battery_efficiency)
            )
            
        constraints.extend([
            soc <= self.community_config.total_storage_capacity_kwh,
            battery_charge <= self.community_config.max_storage_power_kw,
            battery_discharge <= self.community_config.max_storage_power_kw,
            grid_import <= self.community_config.grid_import_limit_kw,
            grid_export <= self.community_config.grid_export_limit_kw
        ])
        
        peak_hours = self._get_peak_hours(T)
        energy_balance = []
        for t in range(T):
            net_load = forecasted_load[t] - forecasted_pv[t] + battery_charge[t] - battery_discharge[t]
            if net_load >= 0:
                energy_balance.append(grid_import[t] == net_load)
                energy_balance.append(grid_export[t] == 0)
            else:
                energy_balance.append(grid_export[t] == -net_load)
                energy_balance.append(grid_import[t] == 0)
                
        constraints.extend(energy_balance)
        
        constraints.extend([
            tou_rate_peak >= 0.1,  
            tou_rate_peak <= 0.5,  
            tou_rate_offpeak >= 0.05,
            tou_rate_offpeak <= 0.3,
            feed_in_rate >= 0.02,
            feed_in_rate <= 0.15,
            tou_rate_peak >= tou_rate_offpeak  
        ])
        
        consumer_cost = 0
        producer_revenue = 0
        
        for t in range(T):
            if peak_hours[t]:
                consumer_cost += tou_rate_peak * grid_import[t] * dt
            else:
                consumer_cost += tou_rate_offpeak * grid_import[t] * dt
            producer_revenue += feed_in_rate * grid_export[t] * dt
            
        welfare = producer_revenue - 0.8 * consumer_cost  
        
        problem = cp.Problem(cp.Maximize(welfare), constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"]:
            battery_dispatch = (battery_discharge.value - battery_charge.value).tolist()
            grid_exchange = (grid_import.value - grid_export.value).tolist()
            
            recommended_tariff = TariffStructure(
                time_of_use_rates={
                    "peak": float(tou_rate_peak.value),
                    "off_peak": float(tou_rate_offpeak.value)
                },
                feed_in_tariff=float(feed_in_rate.value),
                demand_charge=0.0,
                fixed_charge_daily=5.0
            )
            
            original_cost = self._calculate_original_cost(request, forecasted_load, forecasted_pv)
            optimized_cost = float(consumer_cost.value) if consumer_cost.value is not None else 0
            cost_savings = original_cost - optimized_cost
            
            peak_reduction = self._calculate_peak_reduction(grid_exchange, forecasted_load)
            self_consumption_rate = self._calculate_self_consumption_rate(forecasted_pv, grid_exchange)
            
            return OptimizationResult(
                objective_value=float(problem.value),
                battery_dispatch=battery_dispatch,
                grid_exchange=grid_exchange,
                recommended_tariff=recommended_tariff,
                cost_savings=cost_savings,
                peak_reduction_kw=peak_reduction,
                self_consumption_rate=self_consumption_rate
            )
        else:
            raise ValueError(f"Optimization problem status: {problem.status}")
            
    def _optimize_cost_minimization(self, request: OptimizationRequest) -> OptimizationResult:
        
        T = request.horizon_hours * 4
        
        battery_charge = cp.Variable(T, nonneg=True)
        battery_discharge = cp.Variable(T, nonneg=True)
        grid_import = cp.Variable(T, nonneg=True)
        soc = cp.Variable(T + 1, nonneg=True)
        
        forecasted_load = np.array(request.forecasted_load[:T])
        forecasted_pv = np.array(request.forecasted_pv[:T])
        
        dt = 0.25
        battery_efficiency = 0.95
        
        constraints = []
        
        constraints.append(soc[0] == request.current_soc_kwh)
        for t in range(T):
            constraints.append(
                soc[t + 1] == soc[t] + dt * (battery_efficiency * battery_charge[t] - battery_discharge[t] / battery_efficiency)
            )
            
        constraints.extend([
            soc <= self.community_config.total_storage_capacity_kwh,
            battery_charge <= self.community_config.max_storage_power_kw,
            battery_discharge <= self.community_config.max_storage_power_kw,
            grid_import <= self.community_config.grid_import_limit_kw
        ])
        
        for t in range(T):
            net_load = forecasted_load[t] - forecasted_pv[t] + battery_charge[t] - battery_discharge[t]
            constraints.append(grid_import[t] >= net_load)
            
        peak_hours = self._get_peak_hours(T)
        total_cost = 0
        
        for t in range(T):
            if peak_hours[t]:
                rate = request.current_tariff.time_of_use_rates.get("peak", 0.3)
            else:
                rate = request.current_tariff.time_of_use_rates.get("off_peak", 0.15)
            total_cost += rate * grid_import[t] * dt
            
        problem = cp.Problem(cp.Minimize(total_cost), constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"]:
            battery_dispatch = (battery_discharge.value - battery_charge.value).tolist()
            grid_exchange = grid_import.value.tolist()
            
            original_cost = self._calculate_original_cost(request, forecasted_load, forecasted_pv)
            optimized_cost = float(total_cost.value)
            cost_savings = original_cost - optimized_cost
            
            peak_reduction = self._calculate_peak_reduction(grid_exchange, forecasted_load)
            self_consumption_rate = self._calculate_self_consumption_rate(forecasted_pv, grid_exchange)
            
            return OptimizationResult(
                objective_value=float(problem.value),
                battery_dispatch=battery_dispatch,
                grid_exchange=grid_exchange,
                recommended_tariff=request.current_tariff,
                cost_savings=cost_savings,
                peak_reduction_kw=peak_reduction,
                self_consumption_rate=self_consumption_rate
            )
        else:
            raise ValueError(f"Optimization problem status: {problem.status}")
            
    def _optimize_peak_minimization(self, request: OptimizationRequest) -> OptimizationResult:
        
        T = request.horizon_hours * 4
        
        battery_charge = cp.Variable(T, nonneg=True)
        battery_discharge = cp.Variable(T, nonneg=True)
        grid_import = cp.Variable(T, nonneg=True)
        soc = cp.Variable(T + 1, nonneg=True)
        peak_demand = cp.Variable(nonneg=True)
        
        forecasted_load = np.array(request.forecasted_load[:T])
        forecasted_pv = np.array(request.forecasted_pv[:T])
        
        dt = 0.25
        battery_efficiency = 0.95
        
        constraints = []
        
        constraints.append(soc[0] == request.current_soc_kwh)
        for t in range(T):
            constraints.append(
                soc[t + 1] == soc[t] + dt * (battery_efficiency * battery_charge[t] - battery_discharge[t] / battery_efficiency)
            )
            
        constraints.extend([
            soc <= self.community_config.total_storage_capacity_kwh,
            battery_charge <= self.community_config.max_storage_power_kw,
            battery_discharge <= self.community_config.max_storage_power_kw,
            grid_import <= self.community_config.grid_import_limit_kw
        ])
        
        for t in range(T):
            net_load = forecasted_load[t] - forecasted_pv[t] + battery_charge[t] - battery_discharge[t]
            constraints.append(grid_import[t] >= net_load)
            constraints.append(peak_demand >= grid_import[t])
            
        problem = cp.Problem(cp.Minimize(peak_demand), constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"]:
            battery_dispatch = (battery_discharge.value - battery_charge.value).tolist()
            grid_exchange = grid_import.value.tolist()
            
            original_peak = max(forecasted_load)
            optimized_peak = float(peak_demand.value)
            peak_reduction = original_peak - optimized_peak
            
            self_consumption_rate = self._calculate_self_consumption_rate(forecasted_pv, grid_exchange)
            
            return OptimizationResult(
                objective_value=float(problem.value),
                battery_dispatch=battery_dispatch,
                grid_exchange=grid_exchange,
                recommended_tariff=request.current_tariff,
                cost_savings=0.0,
                peak_reduction_kw=peak_reduction,
                self_consumption_rate=self_consumption_rate
            )
        else:
            raise ValueError(f"Optimization problem status: {problem.status}")
            
    def _optimize_self_consumption(self, request: OptimizationRequest) -> OptimizationResult:
        
        T = request.horizon_hours * 4
        
        battery_charge = cp.Variable(T, nonneg=True)
        battery_discharge = cp.Variable(T, nonneg=True)
        grid_import = cp.Variable(T, nonneg=True)
        grid_export = cp.Variable(T, nonneg=True)
        soc = cp.Variable(T + 1, nonneg=True)
        
        forecasted_load = np.array(request.forecasted_load[:T])
        forecasted_pv = np.array(request.forecasted_pv[:T])
        
        dt = 0.25
        battery_efficiency = 0.95
        
        constraints = []
        
        constraints.append(soc[0] == request.current_soc_kwh)
        for t in range(T):
            constraints.append(
                soc[t + 1] == soc[t] + dt * (battery_efficiency * battery_charge[t] - battery_discharge[t] / battery_efficiency)
            )
            
        constraints.extend([
            soc <= self.community_config.total_storage_capacity_kwh,
            battery_charge <= self.community_config.max_storage_power_kw,
            battery_discharge <= self.community_config.max_storage_power_kw,
            grid_import <= self.community_config.grid_import_limit_kw,
            grid_export <= self.community_config.grid_export_limit_kw
        ])
        
        for t in range(T):
            constraints.append(
                forecasted_load[t] + battery_charge[t] == 
                forecasted_pv[t] + battery_discharge[t] + grid_import[t] - grid_export[t]
            )
            
        self_consumed_pv = cp.sum(forecasted_pv) - cp.sum(grid_export)
        
        problem = cp.Problem(cp.Maximize(self_consumed_pv), constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"]:
            battery_dispatch = (battery_discharge.value - battery_charge.value).tolist()
            grid_exchange = (grid_import.value - grid_export.value).tolist()
            
            total_pv = np.sum(forecasted_pv)
            self_consumption_rate = float(self_consumed_pv.value) / total_pv if total_pv > 0 else 0
            
            peak_reduction = self._calculate_peak_reduction(grid_exchange, forecasted_load)
            
            return OptimizationResult(
                objective_value=float(problem.value),
                battery_dispatch=battery_dispatch,
                grid_exchange=grid_exchange,
                recommended_tariff=request.current_tariff,
                cost_savings=0.0,
                peak_reduction_kw=peak_reduction,
                self_consumption_rate=self_consumption_rate
            )
        else:
            raise ValueError(f"Optimization problem status: {problem.status}")
            
    def _get_peak_hours(self, T: int) -> List[bool]:
        
        peak_hours = []
        for t in range(T):
            hour = (t * 0.25) % 24
            is_peak = (7 <= hour <= 11) or (17 <= hour <= 21)
            peak_hours.append(is_peak)
        return peak_hours
        
    def _calculate_original_cost(self, request: OptimizationRequest, 
                                forecasted_load: np.ndarray, forecasted_pv: np.ndarray) -> float:
        
        T = len(forecasted_load)
        original_cost = 0.0
        peak_hours = self._get_peak_hours(T)
        dt = 0.25
        
        for t in range(T):
            net_load = max(0, forecasted_load[t] - forecasted_pv[t])
            if peak_hours[t]:
                rate = request.current_tariff.time_of_use_rates.get("peak", 0.3)
            else:
                rate = request.current_tariff.time_of_use_rates.get("off_peak", 0.15)
            original_cost += rate * net_load * dt
            
        return original_cost
        
    def _calculate_peak_reduction(self, grid_exchange: List[float], forecasted_load: np.ndarray) -> float:
        
        original_peak = max(forecasted_load)
        optimized_peak = max([max(0, ge) for ge in grid_exchange])
        return original_peak - optimized_peak
        
    def _calculate_self_consumption_rate(self, forecasted_pv: np.ndarray, grid_exchange: List[float]) -> float:
        
        total_pv = np.sum(forecasted_pv)
        total_export = sum([max(0, -ge) for ge in grid_exchange])
        if total_pv > 0:
            return max(0, (total_pv - total_export) / total_pv)
        return 0.0
