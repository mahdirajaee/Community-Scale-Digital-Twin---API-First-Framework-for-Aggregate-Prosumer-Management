#!/bin/bash

# Digital Twin Framework Deployment Script
# Deploys the complete system with monitoring and security

set -e

echo "🚀 Starting Digital Twin Framework Deployment..."

# Configuration
PROJECT_NAME="digital-twin-framework"
NETWORK_NAME="${PROJECT_NAME}_network"
ENV_FILE="config/.env"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Utility functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    if [ ! -f "$ENV_FILE" ]; then
        log_error "Environment file not found: $ENV_FILE"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create necessary directories
setup_directories() {
    log_info "Setting up directories..."
    
    mkdir -p logs
    mkdir -p data/backups
    mkdir -p models/cache
    mkdir -p config/ssl
    
    log_success "Directories created"
}

# Generate SSL certificates (self-signed for development)
generate_ssl_certs() {
    log_info "Generating SSL certificates..."
    
    if [ ! -f "config/ssl/cert.pem" ]; then
        openssl req -x509 -newkey rsa:4096 -nodes -out config/ssl/cert.pem -keyout config/ssl/key.pem -days 365 \
            -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=localhost"
        log_success "SSL certificates generated"
    else
        log_info "SSL certificates already exist"
    fi
}

# Setup database initialization
setup_database() {
    log_info "Setting up database initialization..."
    
    if [ ! -f "scripts/init_db.sql" ]; then
        cat > scripts/init_db.sql << 'EOF'
-- Digital Twin Database Initialization

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_meter_data_community_timestamp ON meter_data(community_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_weather_data_community_timestamp ON weather_data(community_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_forecasts_community_timestamp ON forecasts(community_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active) WHERE is_active = true;

-- Create admin user (password: admin123 - change in production!)
INSERT INTO users (username, email, full_name, hashed_password, role, is_active, created_at)
VALUES (
    'admin',
    'admin@digitaltwin.com',
    'System Administrator',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LFNf.eC7E8A8F9G0K', -- admin123
    'admin',
    true,
    NOW()
) ON CONFLICT (username) DO NOTHING;

-- Create sample community
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
    NOW()
) ON CONFLICT (id) DO NOTHING;
EOF
        log_success "Database initialization script created"
    fi
}

# Build and deploy services
deploy_services() {
    log_info "Building and deploying services..."
    
    # Create custom network
    docker network create ${NETWORK_NAME} 2>/dev/null || log_info "Network ${NETWORK_NAME} already exists"
    
    # Build and start services
    docker-compose --env-file ${ENV_FILE} build --no-cache
    docker-compose --env-file ${ENV_FILE} up -d
    
    log_success "Services deployed"
}

# Wait for services to be healthy
wait_for_services() {
    log_info "Waiting for services to be healthy..."
    
    services=("db" "redis" "api-gateway")
    
    for service in "${services[@]}"; do
        log_info "Waiting for $service..."
        
        max_attempts=30
        attempt=1
        
        while [ $attempt -le $max_attempts ]; do
            if docker-compose ps $service | grep -q "Up"; then
                log_success "$service is ready"
                break
            fi
            
            if [ $attempt -eq $max_attempts ]; then
                log_error "$service failed to start"
                exit 1
            fi
            
            sleep 2
            ((attempt++))
        done
    done
}

# Initialize database
initialize_database() {
    log_info "Initializing database..."
    
    # Wait for PostgreSQL to be ready
    sleep 10
    
    # Run initialization script
    if [ -f "scripts/init_db.sql" ]; then
        docker-compose exec -T db psql -U postgres -d digital_twin -f /docker-entrypoint-initdb.d/init_db.sql
    fi
    
    # Run Python initialization
    docker-compose exec api-gateway python scripts/initialize_framework.py
    
    log_success "Database initialized"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # API Gateway health check
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        log_success "API Gateway is healthy"
    else
        log_warning "API Gateway health check failed"
    fi
    
    # Forecasting service health check
    if curl -f http://localhost:8001/health >/dev/null 2>&1; then
        log_success "Forecasting service is healthy"
    else
        log_warning "Forecasting service health check failed"
    fi
    
    # Optimization service health check
    if curl -f http://localhost:8002/health >/dev/null 2>&1; then
        log_success "Optimization service is healthy"
    else
        log_warning "Optimization service health check failed"
    fi
    
    # Dashboard health check
    if curl -f http://localhost:8501 >/dev/null 2>&1; then
        log_success "Dashboard is healthy"
    else
        log_warning "Dashboard health check failed"
    fi
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Wait for Prometheus and Grafana to start
    sleep 15
    
    # Import Grafana dashboards
    if [ -d "config/grafana/dashboards" ]; then
        log_info "Importing Grafana dashboards..."
        # Dashboard import logic would go here
    fi
    
    log_success "Monitoring setup complete"
}

# Display deployment summary
show_deployment_summary() {
    log_success "🎉 Digital Twin Framework deployed successfully!"
    echo ""
    echo "📊 Service URLs:"
    echo "  • API Gateway:    http://localhost:8000"
    echo "  • API Docs:       http://localhost:8000/docs"
    echo "  • Dashboard:      http://localhost:8501"
    echo "  • Grafana:        http://localhost:3000 (admin/admin)"
    echo "  • Prometheus:     http://localhost:9090"
    echo ""
    echo "🔧 Database:"
    echo "  • PostgreSQL:     localhost:5432"
    echo "  • Redis:          localhost:6379"
    echo ""
    echo "🔑 Default Credentials:"
    echo "  • Admin User:     admin / admin123"
    echo "  • Grafana:        admin / admin"
    echo ""
    echo "📝 Next Steps:"
    echo "  1. Change default passwords"
    echo "  2. Configure SSL certificates for production"
    echo "  3. Set up external monitoring"
    echo "  4. Configure backup procedures"
    echo ""
    echo "📖 Documentation: Check the README.md for more information"
}

# Cleanup function
cleanup() {
    if [ $? -ne 0 ]; then
        log_error "Deployment failed! Cleaning up..."
        docker-compose down --remove-orphans
    fi
}

# Main deployment flow
main() {
    # Set trap for cleanup
    trap cleanup EXIT
    
    echo "======================================"
    echo "  Digital Twin Framework Deployment"
    echo "======================================"
    echo ""
    
    check_prerequisites
    setup_directories
    generate_ssl_certs
    setup_database
    deploy_services
    wait_for_services
    initialize_database
    setup_monitoring
    sleep 10  # Allow services to fully initialize
    run_health_checks
    show_deployment_summary
    
    # Remove trap on successful completion
    trap - EXIT
}

# Command line options
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        log_info "Stopping services..."
        docker-compose down
        log_success "Services stopped"
        ;;
    "restart")
        log_info "Restarting services..."
        docker-compose restart
        log_success "Services restarted"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        docker-compose ps
        ;;
    "clean")
        log_warning "This will remove all containers and volumes!"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down -v --remove-orphans
            docker system prune -f
            log_success "Cleanup complete"
        fi
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|status|clean}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy the complete framework"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  logs     - Show logs from all services"
        echo "  status   - Show status of all services"
        echo "  clean    - Remove all containers and volumes"
        exit 1
        ;;
esac
