-- Digital Twin Framework Database Initialization Script
-- Creates all necessary tables, indexes, and initial data

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create custom types
CREATE TYPE user_role AS ENUM ('admin', 'operator', 'viewer', 'api_user');
CREATE TYPE optimization_objective AS ENUM ('minimize_cost', 'maximize_welfare', 'minimize_emissions');
CREATE TYPE model_type AS ENUM ('lstm', 'prophet', 'linear_regression', 'ensemble');

-- Communities table
CREATE TABLE IF NOT EXISTS communities (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    num_prosumers INTEGER NOT NULL CHECK (num_prosumers > 0),
    total_pv_capacity_kw DECIMAL(10,3) NOT NULL CHECK (total_pv_capacity_kw >= 0),
    total_storage_capacity_kwh DECIMAL(10,3) NOT NULL CHECK (total_storage_capacity_kwh >= 0),
    max_storage_power_kw DECIMAL(10,3) NOT NULL CHECK (max_storage_power_kw >= 0),
    grid_import_limit_kw DECIMAL(10,3) NOT NULL CHECK (grid_import_limit_kw >= 0),
    grid_export_limit_kw DECIMAL(10,3) NOT NULL CHECK (grid_export_limit_kw >= 0),
    incentive_budget_daily DECIMAL(10,2) NOT NULL CHECK (incentive_budget_daily >= 0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Meter data table
CREATE TABLE IF NOT EXISTS meter_data (
    id SERIAL PRIMARY KEY,
    community_id VARCHAR(255) NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    net_load_kw DECIMAL(10,3) NOT NULL,
    pv_generation_kw DECIMAL(10,3) NOT NULL CHECK (pv_generation_kw >= 0),
    grid_import_kw DECIMAL(10,3) NOT NULL CHECK (grid_import_kw >= 0),
    grid_export_kw DECIMAL(10,3) NOT NULL CHECK (grid_export_kw >= 0),
    storage_soc_kwh DECIMAL(10,3) NOT NULL CHECK (storage_soc_kwh >= 0),
    storage_charge_kw DECIMAL(10,3) NOT NULL CHECK (storage_charge_kw >= 0),
    storage_discharge_kw DECIMAL(10,3) NOT NULL CHECK (storage_discharge_kw >= 0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Weather data table
CREATE TABLE IF NOT EXISTS weather_data (
    id SERIAL PRIMARY KEY,
    community_id VARCHAR(255) NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    irradiance_w_m2 DECIMAL(8,3) NOT NULL CHECK (irradiance_w_m2 >= 0),
    temperature_c DECIMAL(5,2) NOT NULL,
    humidity_percent DECIMAL(5,2) NOT NULL CHECK (humidity_percent >= 0 AND humidity_percent <= 100),
    wind_speed_m_s DECIMAL(5,2) NOT NULL CHECK (wind_speed_m_s >= 0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Users table for authentication
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(100),
    hashed_password VARCHAR(128) NOT NULL,
    role user_role NOT NULL DEFAULT 'viewer',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE
);

-- API keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(128) NOT NULL,
    permissions TEXT[], -- Array of permission strings
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Forecasts table
CREATE TABLE IF NOT EXISTS forecasts (
    id SERIAL PRIMARY KEY,
    community_id VARCHAR(255) NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    model_type model_type NOT NULL,
    forecast_type VARCHAR(50) NOT NULL, -- 'load', 'pv', 'price'
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    forecast_data JSONB NOT NULL, -- Stores forecast values and metadata
    confidence_intervals JSONB, -- Uncertainty bounds
    accuracy_metrics JSONB, -- MAPE, RMSE, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Optimization results table
CREATE TABLE IF NOT EXISTS optimization_results (
    id SERIAL PRIMARY KEY,
    community_id VARCHAR(255) NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    objective optimization_objective NOT NULL,
    horizon_hours INTEGER NOT NULL CHECK (horizon_hours > 0),
    input_parameters JSONB NOT NULL,
    solution_data JSONB NOT NULL, -- Dispatch schedules, tariff structures
    objective_value DECIMAL(12,4),
    solver_status VARCHAR(50),
    solve_time_seconds DECIMAL(8,3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Simulation scenarios table
CREATE TABLE IF NOT EXISTS simulation_scenarios (
    id SERIAL PRIMARY KEY,
    scenario_id VARCHAR(255) UNIQUE NOT NULL,
    community_id VARCHAR(255) NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    scenario_name VARCHAR(255) NOT NULL,
    description TEXT,
    parameter_changes JSONB NOT NULL, -- Scenario parameter modifications
    simulation_duration_days INTEGER NOT NULL CHECK (simulation_duration_days > 0),
    monte_carlo_runs INTEGER DEFAULT 100 CHECK (monte_carlo_runs > 0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Simulation results table
CREATE TABLE IF NOT EXISTS simulation_results (
    id SERIAL PRIMARY KEY,
    scenario_id VARCHAR(255) NOT NULL REFERENCES simulation_scenarios(scenario_id) ON DELETE CASCADE,
    run_number INTEGER NOT NULL,
    kpis JSONB NOT NULL, -- Key performance indicators
    time_series_data JSONB, -- Detailed time series results
    summary_statistics JSONB, -- Statistical summaries
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Macro twin states table (for state history)
CREATE TABLE IF NOT EXISTS macro_twin_states (
    id SERIAL PRIMARY KEY,
    community_id VARCHAR(255) NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    state_vector DECIMAL(10,6)[] NOT NULL, -- Kalman filter state
    covariance_matrix DECIMAL(10,8)[] NOT NULL, -- State covariance
    prediction_accuracy JSONB, -- Accuracy metrics
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(255),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_meter_data_community_timestamp ON meter_data(community_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_weather_data_community_timestamp ON weather_data(community_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_forecasts_community_model_time ON forecasts(community_id, model_type, start_time);
CREATE INDEX IF NOT EXISTS idx_optimization_results_community_time ON optimization_results(community_id, created_at);
CREATE INDEX IF NOT EXISTS idx_simulation_results_scenario ON simulation_results(scenario_id);
CREATE INDEX IF NOT EXISTS idx_macro_twin_states_community_time ON macro_twin_states(community_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_audit_log_user_action ON audit_log(user_id, action);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(created_at);

-- GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_forecasts_data_gin ON forecasts USING GIN (forecast_data);
CREATE INDEX IF NOT EXISTS idx_optimization_input_gin ON optimization_results USING GIN (input_parameters);
CREATE INDEX IF NOT EXISTS idx_optimization_solution_gin ON optimization_results USING GIN (solution_data);
CREATE INDEX IF NOT EXISTS idx_simulation_kpis_gin ON simulation_results USING GIN (kpis);

-- Unique constraints
ALTER TABLE meter_data ADD CONSTRAINT IF NOT EXISTS unique_meter_data_community_timestamp 
    UNIQUE (community_id, timestamp);
ALTER TABLE weather_data ADD CONSTRAINT IF NOT EXISTS unique_weather_data_community_timestamp 
    UNIQUE (community_id, timestamp);

-- Triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_communities_updated_at BEFORE UPDATE ON communities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Views for commonly used queries
CREATE OR REPLACE VIEW community_summary AS
SELECT 
    c.id,
    c.name,
    c.num_prosumers,
    c.total_pv_capacity_kw,
    c.total_storage_capacity_kwh,
    COUNT(md.id) as meter_data_points,
    MAX(md.timestamp) as last_meter_reading,
    COUNT(wd.id) as weather_data_points,
    MAX(wd.timestamp) as last_weather_reading,
    c.created_at
FROM communities c
LEFT JOIN meter_data md ON c.id = md.community_id
LEFT JOIN weather_data wd ON c.id = wd.community_id
GROUP BY c.id, c.name, c.num_prosumers, c.total_pv_capacity_kw, 
         c.total_storage_capacity_kwh, c.created_at;

-- View for recent forecasting performance
CREATE OR REPLACE VIEW forecast_performance AS
SELECT 
    community_id,
    model_type,
    forecast_type,
    AVG((accuracy_metrics->>'mape')::DECIMAL) as avg_mape,
    AVG((accuracy_metrics->>'rmse')::DECIMAL) as avg_rmse,
    COUNT(*) as forecast_count,
    MAX(created_at) as last_forecast
FROM forecasts 
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY community_id, model_type, forecast_type;

-- View for optimization performance
CREATE OR REPLACE VIEW optimization_performance AS
SELECT 
    community_id,
    objective,
    AVG(solve_time_seconds) as avg_solve_time,
    AVG(objective_value) as avg_objective_value,
    COUNT(*) as optimization_count,
    COUNT(CASE WHEN solver_status = 'optimal' THEN 1 END) as optimal_solutions
FROM optimization_results 
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY community_id, objective;

-- Insert default admin user (password: admin123 - change in production!)
INSERT INTO users (username, email, full_name, hashed_password, role, is_active, created_at)
VALUES (
    'admin',
    'admin@digitaltwin.com',
    'System Administrator',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LFNf.eC7E8A8F9G0K', -- bcrypt hash of "admin123"
    'admin',
    true,
    CURRENT_TIMESTAMP
) ON CONFLICT (username) DO NOTHING;

-- Insert sample community for testing
INSERT INTO communities (
    id, name, num_prosumers, total_pv_capacity_kw, total_storage_capacity_kwh,
    max_storage_power_kw, grid_import_limit_kw, grid_export_limit_kw,
    incentive_budget_daily, created_at
) VALUES (
    'sample_community',
    'Sample Community',
    100,
    500.0,
    1000.0,
    200.0,
    800.0,
    300.0,
    1000.0,
    CURRENT_TIMESTAMP
) ON CONFLICT (id) DO NOTHING;

-- Create partitioning for large tables (optional, for high-volume deployments)
-- This can be uncommented for production deployments with high data volumes

/*
-- Partition meter_data by month
CREATE TABLE meter_data_template (LIKE meter_data INCLUDING ALL);
ALTER TABLE meter_data_template ADD CONSTRAINT meter_data_timestamp_check 
    CHECK (false) NO INHERIT;

-- Create function to create monthly partitions
CREATE OR REPLACE FUNCTION create_monthly_partition(table_name text, start_date date)
RETURNS void AS $$
DECLARE
    partition_name text;
    end_date date;
BEGIN
    partition_name := table_name || '_' || to_char(start_date, 'YYYY_MM');
    end_date := start_date + interval '1 month';
    
    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF %I 
                    FOR VALUES FROM (%L) TO (%L)',
                   partition_name, table_name, start_date, end_date);
END;
$$ LANGUAGE plpgsql;
*/

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO postgres;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Create database statistics
ANALYZE;

-- Log completion
INSERT INTO audit_log (action, resource_type, details, created_at)
VALUES (
    'database_initialized',
    'system',
    '{"version": "1.0", "tables_created": true, "indexes_created": true}',
    CURRENT_TIMESTAMP
);

-- Display summary
DO $$
DECLARE
    table_count INTEGER;
    index_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO table_count FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
    
    SELECT COUNT(*) INTO index_count FROM pg_indexes 
    WHERE schemaname = 'public';
    
    RAISE NOTICE 'Database initialization completed successfully!';
    RAISE NOTICE 'Created % tables and % indexes', table_count, index_count;
    RAISE NOTICE 'Default admin user created (username: admin, password: admin123)';
    RAISE NOTICE 'Sample community created (ID: sample_community)';
END $$;
