# Digital Twin Framework - Complete Guide

## 🏢 Community-Scale Digital Twin & API-First Framework

A comprehensive solution for aggregate prosumer management with real-time optimization, forecasting, and scenario simulation capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Security](#security)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The Digital Twin Framework provides:

- **Macro-Twin Modeling**: Lightweight state-space representation using Kalman filters
- **Aggregate Forecasting**: ML models (LSTM/Prophet) for demand and generation prediction
- **Optimization Engine**: Welfare-maximizing tariff schedules and dispatch plans
- **Scenario Simulation**: "What-if" policy evaluation with Monte Carlo analysis
- **Real-time Processing**: Data ingestion and state updates
- **API-First Design**: Microservices architecture with comprehensive REST APIs

### Key Capabilities

✅ **Scale**: Handle communities from tens to thousands of prosumers  
✅ **Performance**: Complete optimizations within minutes on a single VM  
✅ **Real-time**: Continuous data ingestion and state updates  
✅ **Flexibility**: Rapid scenario evaluation and policy testing  
✅ **Production-Ready**: Security, monitoring, and deployment automation  

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │    │   API Gateway   │    │   Monitoring    │
│  (Streamlit)    │    │   (FastAPI)     │    │(Prometheus/Graf)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Forecasting    │    │  Optimization   │    │   Simulation    │
│   Service       │    │    Service      │    │    Service      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     Redis       │    │  Macro Twin     │
│   Database      │    │     Cache       │    │     Core        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

1. **API Gateway**: Central entry point with authentication and routing
2. **Macro Twin Core**: Kalman filter-based state estimation
3. **Forecasting Service**: LSTM and Prophet models for predictions
4. **Optimization Engine**: CVXPY-based dispatch and tariff optimization
5. **Simulation Service**: Monte Carlo scenario analysis
6. **Dashboard**: Real-time visualization and control interface

## 🚀 Features

### Forecasting & Prediction
- Multi-model ensemble forecasting (LSTM, Prophet)
- Uncertainty quantification with confidence intervals
- Weather-aware generation forecasting
- Load pattern recognition and prediction

### Optimization & Control
- Multi-objective optimization (cost, welfare, emissions)
- Dynamic tariff design and dispatch optimization
- Storage and DER coordination
- Grid constraint management

### Scenario Analysis
- Monte Carlo simulation for policy evaluation
- Technology mix scenario comparison
- Economic impact analysis
- Risk assessment and sensitivity analysis

### Real-time Operations
- Continuous data ingestion from meters and weather stations
- Real-time state estimation and prediction
- Automated optimization execution
- Alert and notification systems

### Security & Authentication
- JWT-based user authentication
- Role-based access control (RBAC)
- API key management for service integration
- Rate limiting and security monitoring

### Monitoring & Observability
- Prometheus metrics collection
- Grafana dashboards for visualization
- Performance monitoring and alerting
- Audit logging and compliance tracking

## 🚀 Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM minimum
- 50GB disk space

### 1. Clone and Setup

```bash
git clone <repository-url>
cd digital-twin-framework

# Copy environment configuration
cp config/.env.example config/.env

# Edit configuration as needed
nano config/.env
```

### 2. Deploy with One Command

```bash
# Development deployment
./scripts/deployment/deploy.sh

# Production deployment
ENVIRONMENT=production ./scripts/deployment/deploy_production.sh
```

### 3. Access Services

- **Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Monitoring**: http://localhost:3000 (admin/admin)

### 4. Create Your First Community

```bash
curl -X POST "http://localhost:8000/communities" \
  -H "Content-Type: application/json" \
  -d '{
    "community_id": "my_community",
    "num_prosumers": 100,
    "total_pv_capacity_kw": 500.0,
    "total_storage_capacity_kwh": 1000.0,
    "max_storage_power_kw": 200.0,
    "grid_import_limit_kw": 800.0,
    "grid_export_limit_kw": 300.0,
    "incentive_budget_daily": 1000.0
  }'
```

## 📚 API Documentation

### Authentication

All API endpoints (except public ones) require authentication:

```bash
# Register a new user
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "operator1",
    "email": "operator@example.com",
    "password": "secure_password",
    "role": "operator"
  }'

# Login and get access token
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "operator1",
    "password": "secure_password"
  }'

# Use token in subsequent requests
curl -H "Authorization: Bearer <your_token>" \
  "http://localhost:8000/auth/me"
```

### Core Endpoints

#### Communities
- `POST /communities` - Create new community
- `GET /communities` - List all communities
- `GET /communities/{id}` - Get community details
- `PUT /communities/{id}` - Update community configuration

#### Forecasting
- `POST /communities/{id}/forecasting/generate` - Generate forecasts
- `GET /communities/{id}/forecasting/models` - List available models
- `POST /communities/{id}/forecasting/train` - Train new models

#### Optimization
- `POST /communities/{id}/optimization/dispatch` - Optimize dispatch
- `POST /communities/{id}/optimization/tariff` - Optimize tariffs
- `GET /communities/{id}/optimization/results` - Get results

#### Simulation
- `POST /communities/{id}/simulation/run-scenario` - Run single scenario
- `POST /communities/{id}/simulation/run-multiple` - Run multiple scenarios
- `GET /communities/{id}/simulation/results/{id}` - Get simulation results

### Data Models

#### Community Configuration
```json
{
  "community_id": "string",
  "num_prosumers": 100,
  "total_pv_capacity_kw": 500.0,
  "total_storage_capacity_kwh": 1000.0,
  "max_storage_power_kw": 200.0,
  "grid_import_limit_kw": 800.0,
  "grid_export_limit_kw": 300.0,
  "incentive_budget_daily": 1000.0
}
```

#### Forecast Request
```json
{
  "start_time": "2025-01-01T00:00:00Z",
  "end_time": "2025-01-02T00:00:00Z",
  "horizon_hours": 24,
  "confidence_level": 0.95,
  "include_uncertainty": true
}
```

## ⚙️ Configuration

### Environment Variables

Key configuration options in `config/.env`:

```bash
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/digital_twin

# Redis Cache
REDIS_URL=redis://localhost:6379

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your-secret-key

# JWT Authentication
JWT_SECRET_KEY=your-jwt-secret
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Performance
MAX_PROSUMERS_PER_COMMUNITY=10000
OPTIMIZATION_TIMEOUT_MINUTES=5
BATCH_SIZE=100

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### Service Configuration

Each service can be configured independently:

- **Forecasting**: Model parameters, training schedules
- **Optimization**: Solver settings, timeout values
- **Simulation**: Monte Carlo parameters, scenario definitions
- **Cache**: TTL values, memory limits

## 🚀 Deployment

### Development Deployment

```bash
# Quick start for development
./scripts/deployment/deploy.sh

# View logs
./scripts/deployment/deploy.sh logs

# Stop services
./scripts/deployment/deploy.sh stop
```

### Production Deployment

```bash
# Full production setup with SSL and monitoring
DOMAIN=yourdomain.com LETSENCRYPT_EMAIL=admin@yourdomain.com \
  ./scripts/deployment/deploy_production.sh production

# Features included:
# - SSL certificates (Let's Encrypt)
# - Nginx reverse proxy
# - Security hardening
# - Automated backups
# - Production monitoring
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes cluster
kubectl apply -f deployment/kubernetes/

# Monitor deployment
kubectl get pods -n digital-twin
```

### Cloud Deployment

Supported platforms:
- **AWS**: ECS, EKS, EC2
- **Azure**: Container Instances, AKS
- **Google Cloud**: Cloud Run, GKE
- **DigitalOcean**: App Platform, Kubernetes

## 📊 Monitoring

### Metrics

The framework collects comprehensive metrics:

- **API Performance**: Request latency, throughput, error rates
- **Forecasting**: Model accuracy, prediction intervals, training time
- **Optimization**: Solution time, objective values, constraint violations
- **System**: CPU, memory, disk usage, database connections

### Dashboards

Pre-configured Grafana dashboards:

1. **Overview**: System health and key metrics
2. **API Monitoring**: Request patterns and performance
3. **Forecasting**: Model performance and accuracy
4. **Optimization**: Solution quality and timing
5. **Infrastructure**: Resource utilization

### Alerting

Automated alerts for:
- High API latency (>500ms)
- Database connection issues
- Memory usage >90%
- Service downtime
- Model accuracy degradation

## 🔒 Security

### Authentication & Authorization

- **JWT Tokens**: Secure, stateless authentication
- **Role-Based Access**: Admin, Operator, Viewer, API User roles
- **API Keys**: Service-to-service authentication
- **Rate Limiting**: Protection against abuse

### Data Protection

- **Encryption**: All data encrypted in transit and at rest
- **Sensitive Data**: Passwords hashed with bcrypt
- **Audit Logging**: All actions logged for compliance
- **Input Validation**: Comprehensive request validation

### Network Security

- **HTTPS Only**: All traffic encrypted
- **CORS Configuration**: Restricted origins
- **Firewall Rules**: Minimal exposed ports
- **Security Headers**: HSTS, CSP, X-Frame-Options

## 🛠️ Development

### Local Development Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Start local services
docker-compose up -d db redis

# Run API in development mode
cd src
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v --cov=src
```

### Code Structure

```
src/
├── api/           # FastAPI application and routes
├── data/          # Data models and database
├── forecasting/   # ML models and forecasting logic
├── macro_twin/    # Kalman filter implementation
├── optimization/  # Optimization engines
├── simulation/    # Scenario simulation
└── utils/         # Shared utilities
```

### Testing

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/test_forecasting.py
pytest tests/test_optimization.py
pytest tests/test_simulation.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 🐛 Troubleshooting

### Common Issues

#### Services Won't Start
```bash
# Check logs
docker-compose logs

# Check system resources
docker stats

# Clean and rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

#### Database Connection Issues
```bash
# Check database status
docker-compose exec db pg_isready

# Check connection string
echo $DATABASE_URL

# Reset database
docker-compose down db
docker volume rm digital-twin_postgres_data
docker-compose up -d db
```

#### High Memory Usage
```bash
# Check Redis memory
docker-compose exec redis redis-cli info memory

# Clear cache
docker-compose exec redis redis-cli flushall

# Adjust cache settings in config/.env
```

### Performance Optimization

1. **Database**: Add indexes, optimize queries
2. **Cache**: Increase Redis memory, tune TTL values
3. **API**: Enable connection pooling, optimize serialization
4. **Models**: Use model compression, batch predictions

### Getting Help

- **Documentation**: Check this README and API docs
- **Logs**: Review application and service logs
- **Monitoring**: Check Grafana dashboards
- **Issues**: Open GitHub issue with logs and configuration

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

For support and questions:
- Open an issue on GitHub
- Check the documentation
- Review existing issues and solutions

## 🔄 Updates

The framework is actively maintained with regular updates for:
- Security patches
- Performance improvements
- New features
- Bug fixes

Stay updated by watching the repository and checking releases.
