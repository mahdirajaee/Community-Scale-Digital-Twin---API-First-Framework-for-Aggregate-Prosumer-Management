import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import joblib
import logging
from pathlib import Path

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import prophet

from ..data.models import ForecastRequest, ForecastResult, TimeSeriesDataPoint
from ..data.database import get_db, MeterData, WeatherData

logger = logging.getLogger(__name__)


class LSTMForecaster:
    
    def __init__(self, variable_name: str, lookback_window: int = 96):
        self.variable_name = variable_name
        self.lookback_window = lookback_window
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        
    def prepare_sequences(self, data: np.ndarray, target_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.lookback_window, len(scaled_data)):
            X.append(scaled_data[i-self.lookback_window:i, 0])
            if target_data is not None:
                y.append(target_data[i])
            else:
                y.append(scaled_data[i, 0])
                
        return np.array(X), np.array(y)
        
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
        
    def train(self, training_data: np.ndarray, validation_split: float = 0.2) -> Dict[str, float]:
        
        X, y = self.prepare_sequences(training_data)
        
        if len(X) < self.lookback_window * 2:
            raise ValueError("Insufficient training data")
            
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        self.model = self.build_model((X.shape[1], 1))
        
        history = self.model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=validation_split,
            verbose=0,
            shuffle=True
        )
        
        self.is_trained = True
        
        val_loss = min(history.history['val_loss'])
        val_mae = min(history.history['val_mae'])
        
        return {
            'val_loss': val_loss,
            'val_mae': val_mae,
            'training_samples': len(X)
        }
        
    def predict(self, last_sequence: np.ndarray, steps_ahead: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        scaled_sequence = self.scaler.transform(last_sequence.reshape(-1, 1))
        current_sequence = scaled_sequence[-self.lookback_window:].flatten()
        
        predictions = []
        uncertainties = []
        
        for _ in range(steps_ahead):
            X_pred = current_sequence.reshape(1, self.lookback_window, 1)
            pred_scaled = self.model.predict(X_pred, verbose=0)[0, 0]
            predictions.append(pred_scaled)
            
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred_scaled
            
            uncertainty = 0.1 * abs(pred_scaled)  
            uncertainties.append(uncertainty)
            
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        uncertainties = np.array(uncertainties)
        lower_bound = predictions - 1.96 * uncertainties
        upper_bound = predictions + 1.96 * uncertainties
        
        return predictions, lower_bound, upper_bound
        
    def save_model(self, model_path: str):
        
        if self.model:
            self.model.save(f"{model_path}_lstm.h5")
            joblib.dump(self.scaler, f"{model_path}_scaler.pkl")
            
    def load_model(self, model_path: str):
        
        try:
            self.model = load_model(f"{model_path}_lstm.h5")
            self.scaler = joblib.load(f"{model_path}_scaler.pkl")
            self.is_trained = True
            return True
        except:
            return False


class ProphetForecaster:
    
    def __init__(self, variable_name: str):
        self.variable_name = variable_name
        self.model = None
        self.is_trained = False
        
    def train(self, timestamps: List[datetime], values: List[float]) -> Dict[str, float]:
        
        df = pd.DataFrame({
            'ds': timestamps,
            'y': values
        })
        
        self.model = prophet.Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            uncertainty_samples=1000
        )
        
        self.model.fit(df)
        self.is_trained = True
        
        future = self.model.make_future_dataframe(periods=24, freq='15min')
        forecast = self.model.predict(future)
        
        actual_values = df['y'].values
        predicted_values = forecast['yhat'].values[:len(actual_values)]
        
        mae = mean_absolute_error(actual_values, predicted_values)
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'training_samples': len(df)
        }
        
    def predict(self, steps_ahead: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        future = self.model.make_future_dataframe(periods=steps_ahead, freq='15min')
        forecast = self.model.predict(future)
        
        predictions = forecast['yhat'].values[-steps_ahead:]
        lower_bound = forecast['yhat_lower'].values[-steps_ahead:]
        upper_bound = forecast['yhat_upper'].values[-steps_ahead:]
        
        return predictions, lower_bound, upper_bound
        
    def save_model(self, model_path: str):
        
        if self.model:
            joblib.dump(self.model, f"{model_path}_prophet.pkl")
            
    def load_model(self, model_path: str):
        
        try:
            self.model = joblib.load(f"{model_path}_prophet.pkl")
            self.is_trained = True
            return True
        except:
            return False


class ForecastingService:
    
    def __init__(self, community_id: str):
        self.community_id = community_id
        self.models = {}
        self.model_dir = Path(f"./models/{community_id}")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def train_models(self, retrain_days: int = 90) -> Dict[str, Dict[str, float]]:
        
        db = next(get_db())
        training_results = {}
        
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=retrain_days)
            
            meter_data = db.query(MeterData)\
                .filter(MeterData.community_id == self.community_id)\
                .filter(MeterData.timestamp >= start_date)\
                .filter(MeterData.timestamp <= end_date)\
                .order_by(MeterData.timestamp)\
                .all()
                
            if len(meter_data) < 96:  
                raise ValueError("Insufficient training data")
                
            timestamps = [m.timestamp for m in meter_data]
            
            variables = {
                'net_load_kw': [m.net_load_kw for m in meter_data],
                'pv_generation_kw': [m.pv_generation_kw for m in meter_data],
                'storage_soc_kwh': [m.storage_soc_kwh for m in meter_data]
            }
            
            for var_name, values in variables.items():
                lstm_model = LSTMForecaster(var_name)
                prophet_model = ProphetForecaster(var_name)
                
                lstm_results = lstm_model.train(np.array(values))
                prophet_results = prophet_model.train(timestamps, values)
                
                lstm_model.save_model(str(self.model_dir / f"{var_name}_lstm"))
                prophet_model.save_model(str(self.model_dir / f"{var_name}_prophet"))
                
                self.models[f"{var_name}_lstm"] = lstm_model
                self.models[f"{var_name}_prophet"] = prophet_model
                
                training_results[var_name] = {
                    'lstm': lstm_results,
                    'prophet': prophet_results
                }
                
            logger.info(f"Trained models for community {self.community_id}")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
        finally:
            db.close()
            
        return training_results
        
    def load_models(self) -> bool:
        
        variables = ['net_load_kw', 'pv_generation_kw', 'storage_soc_kwh']
        models_loaded = 0
        
        for var_name in variables:
            lstm_model = LSTMForecaster(var_name)
            prophet_model = ProphetForecaster(var_name)
            
            if lstm_model.load_model(str(self.model_dir / f"{var_name}_lstm")):
                self.models[f"{var_name}_lstm"] = lstm_model
                models_loaded += 1
                
            if prophet_model.load_model(str(self.model_dir / f"{var_name}_prophet")):
                self.models[f"{var_name}_prophet"] = prophet_model
                models_loaded += 1
                
        return models_loaded > 0
        
    async def generate_forecast(self, request: ForecastRequest) -> List[ForecastResult]:
        
        if not self.models:
            if not self.load_models():
                await self._train_if_needed()
                
        results = []
        
        variables = ['net_load_kw', 'pv_generation_kw', 'storage_soc_kwh']
        
        for var_name in variables:
            lstm_key = f"{var_name}_lstm"
            prophet_key = f"{var_name}_prophet"
            
            if lstm_key in self.models and prophet_key in self.models:
                lstm_model = self.models[lstm_key]
                prophet_model = self.models[prophet_key]
                
                try:
                    recent_data = await self._get_recent_data(var_name, lstm_model.lookback_window)
                    
                    lstm_pred, lstm_lower, lstm_upper = lstm_model.predict(recent_data, request.horizon_hours * 4)
                    prophet_pred, prophet_lower, prophet_upper = prophet_model.predict(request.horizon_hours * 4)
                    
                    ensemble_pred = 0.6 * lstm_pred + 0.4 * prophet_pred
                    ensemble_lower = 0.6 * lstm_lower + 0.4 * prophet_lower
                    ensemble_upper = 0.6 * lstm_upper + 0.4 * prophet_upper
                    
                    timestamps = [request.start_time + timedelta(minutes=15*i) for i in range(len(ensemble_pred))]
                    
                    predictions = [TimeSeriesDataPoint(timestamp=ts, value=val, unit="kW" if "kw" in var_name else "kWh") 
                                 for ts, val in zip(timestamps, ensemble_pred)]
                    
                    if request.include_uncertainty:
                        lower_bound = [TimeSeriesDataPoint(timestamp=ts, value=val, unit="kW" if "kw" in var_name else "kWh") 
                                     for ts, val in zip(timestamps, ensemble_lower)]
                        upper_bound = [TimeSeriesDataPoint(timestamp=ts, value=val, unit="kW" if "kw" in var_name else "kWh") 
                                     for ts, val in zip(timestamps, ensemble_upper)]
                    else:
                        lower_bound = None
                        upper_bound = None
                    
                    result = ForecastResult(
                        variable=var_name,
                        predictions=predictions,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        confidence_level=request.confidence_level
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error generating forecast for {var_name}: {e}")
                    
        return results
        
    async def _get_recent_data(self, variable_name: str, num_points: int) -> np.ndarray:
        
        db = next(get_db())
        try:
            recent_data = db.query(MeterData)\
                .filter(MeterData.community_id == self.community_id)\
                .order_by(MeterData.timestamp.desc())\
                .limit(num_points)\
                .all()
                
            if variable_name == 'net_load_kw':
                values = [m.net_load_kw for m in reversed(recent_data)]
            elif variable_name == 'pv_generation_kw':
                values = [m.pv_generation_kw for m in reversed(recent_data)]
            elif variable_name == 'storage_soc_kwh':
                values = [m.storage_soc_kwh for m in reversed(recent_data)]
            else:
                raise ValueError(f"Unknown variable: {variable_name}")
                
            return np.array(values)
            
        finally:
            db.close()
            
    async def _train_if_needed(self):
        
        logger.info("No trained models found, training new models...")
        self.train_models()
