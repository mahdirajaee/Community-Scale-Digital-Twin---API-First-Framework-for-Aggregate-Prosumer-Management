#!/bin/bash

# Production Deployment Script for Digital Twin Framework
# Optimized for production environments with security and monitoring

set -e

echo "🏭 Starting Production Deployment..."

# Production Configuration
PROJECT_NAME="digital-twin-prod"
DOMAIN="${DOMAIN:-digitaltwin.example.com}"
EMAIL="${LETSENCRYPT_EMAIL:-admin@example.com}"
ENV_FILE="config/.env.production"

# Source common functions
source "$(dirname "$0")/deploy.sh"

# Production-specific setup
setup_production_env() {
    log_info "Setting up production environment..."
    
    if [ ! -f "$ENV_FILE" ]; then
        log_info "Creating production environment file..."
        cp config/.env.example config/.env.production
        
        # Generate secure random keys
        JWT_SECRET=$(openssl rand -hex 32)
        API_SECRET=$(openssl rand -hex 32)
        
        # Update production settings
        sed -i.bak \
            -e "s/your-jwt-secret-key-change-in-production/$JWT_SECRET/" \
            -e "s/your-secret-key-change-in-production/$API_SECRET/" \
            -e "s/DEVELOPMENT_MODE=true/DEVELOPMENT_MODE=false/" \
            -e "s/API_DEBUG=false/API_DEBUG=false/" \
            -e "s/LOG_LEVEL=INFO/LOG_LEVEL=WARNING/" \
            "$ENV_FILE"
        
        log_success "Production environment file created"
    fi
}

# Setup SSL with Let's Encrypt
setup_letsencrypt() {
    log_info "Setting up Let's Encrypt SSL certificates..."
    
    if [ ! -d "config/ssl/letsencrypt" ]; then
        mkdir -p config/ssl/letsencrypt
        
        # Install certbot if not present
        if ! command -v certbot &> /dev/null; then
            log_info "Installing certbot..."
            sudo apt-get update
            sudo apt-get install -y certbot
        fi
        
        # Generate SSL certificate
        sudo certbot certonly --standalone \
            --email "$EMAIL" \
            --agree-tos \
            --no-eff-email \
            -d "$DOMAIN" \
            --cert-path config/ssl/letsencrypt/
        
        log_success "SSL certificates obtained"
    else
        log_info "SSL certificates already exist"
    fi
}

# Setup reverse proxy with Nginx
setup_nginx() {
    log_info "Setting up Nginx reverse proxy..."
    
    cat > config/nginx.conf << EOF
upstream api_backend {
    server api-gateway:8000;
}

upstream dashboard_backend {
    server dashboard:8501;
}

upstream grafana_backend {
    server grafana:3000;
}

server {
    listen 80;
    server_name ${DOMAIN};
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ${DOMAIN};

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

    # API endpoints
    location /api/ {
        proxy_pass http://api_backend/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Rate limiting
        limit_req zone=api burst=20 nodelay;
    }

    # Dashboard
    location /dashboard/ {
        proxy_pass http://dashboard_backend/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Monitoring (restricted access)
    location /monitoring/ {
        auth_basic "Monitoring";
        auth_basic_user_file /etc/nginx/.htpasswd;
        
        proxy_pass http://grafana_backend/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Static files and documentation
    location / {
        root /var/www/html;
        index index.html;
        try_files \$uri \$uri/ =404;
    }
}

# Rate limiting
http {
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
}
EOF

    log_success "Nginx configuration created"
}

# Setup database backup
setup_backup() {
    log_info "Setting up database backup..."
    
    mkdir -p scripts/backup
    
    cat > scripts/backup/backup_db.sh << 'EOF'
#!/bin/bash

# Database backup script
BACKUP_DIR="/app/data/backups"
DB_NAME="digital_twin"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/backup_${TIMESTAMP}.sql"

# Create backup
docker-compose exec -T db pg_dump -U postgres -d ${DB_NAME} > ${BACKUP_FILE}

# Compress backup
gzip ${BACKUP_FILE}

# Keep only last 30 days of backups
find ${BACKUP_DIR} -name "backup_*.sql.gz" -mtime +30 -delete

echo "Backup completed: ${BACKUP_FILE}.gz"
EOF

    chmod +x scripts/backup/backup_db.sh
    
    # Setup cron job for daily backups
    echo "0 2 * * * /path/to/scripts/backup/backup_db.sh" | crontab -
    
    log_success "Backup system configured"
}

# Setup monitoring and alerting
setup_monitoring() {
    log_info "Setting up production monitoring..."
    
    # Prometheus configuration for production
    cat > config/prometheus.prod.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    environment: 'production'
    cluster: 'digital-twin'

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'digital-twin-api'
    static_configs:
      - targets: ['api-gateway:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'digital-twin-forecasting'
    static_configs:
      - targets: ['forecasting-service:8001']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'digital-twin-optimization'
    static_configs:
      - targets: ['optimization-service:8002']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF

    # Alert rules
    cat > config/alert_rules.yml << EOF
groups:
  - name: digital_twin_alerts
    rules:
      - alert: HighAPILatency
        expr: histogram_quantile(0.95, http_request_duration_seconds_bucket) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
          description: "95th percentile latency is {{ \$value }}s"

      - alert: DatabaseConnectionHigh
        expr: pg_stat_activity_count > 80
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High database connection count"
          description: "{{ \$value }} active connections"

      - alert: RedisMemoryUsageHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis memory usage high"
          description: "Memory usage is {{ \$value | humanizePercentage }}"

      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "{{ \$labels.job }} service is down"
EOF

    log_success "Production monitoring configured"
}

# Security hardening
security_hardening() {
    log_info "Applying security hardening..."
    
    # Create security configuration
    cat > config/security.yml << EOF
# Security Configuration for Production

security:
  # Authentication
  jwt:
    secret_rotation_days: 30
    token_expiry_minutes: 15
    refresh_token_expiry_days: 7
  
  # Rate limiting
  rate_limits:
    login_attempts: 5
    api_requests_per_minute: 60
    burst_requests: 10
  
  # CORS
  cors:
    allowed_origins:
      - "https://${DOMAIN}"
    allowed_methods: ["GET", "POST", "PUT", "DELETE"]
    allowed_headers: ["Authorization", "Content-Type"]
  
  # API Security
  api:
    require_api_key: true
    enforce_https: true
    validate_content_type: true
  
  # Database Security
  database:
    connection_encryption: true
    query_logging: true
    sensitive_data_masking: true
EOF

    # Setup fail2ban for additional security
    if command -v fail2ban-client &> /dev/null; then
        cat > /etc/fail2ban/jail.d/digital-twin.conf << EOF
[digital-twin-api]
enabled = true
port = 443,80
filter = digital-twin-api
logpath = /var/log/nginx/access.log
maxretry = 5
bantime = 3600
EOF
    fi
    
    log_success "Security hardening applied"
}

# Production health checks
run_production_health_checks() {
    log_info "Running production health checks..."
    
    # SSL certificate check
    if openssl s_client -connect "${DOMAIN}:443" -servername "${DOMAIN}" </dev/null 2>/dev/null | openssl x509 -checkend 2592000 -noout; then
        log_success "SSL certificate is valid and not expiring soon"
    else
        log_warning "SSL certificate check failed or expiring soon"
    fi
    
    # Security headers check
    if curl -s -I "https://${DOMAIN}" | grep -q "Strict-Transport-Security"; then
        log_success "Security headers are properly configured"
    else
        log_warning "Security headers check failed"
    fi
    
    # Database backup check
    if [ -f "data/backups/backup_$(date +%Y%m%d)*.sql.gz" ]; then
        log_success "Recent database backup found"
    else
        log_warning "No recent database backup found"
    fi
    
    # Monitoring check
    if curl -f "http://localhost:9090/-/healthy" >/dev/null 2>&1; then
        log_success "Prometheus monitoring is healthy"
    else
        log_warning "Prometheus monitoring check failed"
    fi
}

# Main production deployment
main_production() {
    echo "=========================================="
    echo "  Digital Twin Production Deployment"
    echo "=========================================="
    echo ""
    
    check_prerequisites
    setup_production_env
    setup_directories
    setup_letsencrypt
    setup_nginx
    setup_backup
    setup_monitoring
    security_hardening
    
    # Use production docker-compose file
    export COMPOSE_FILE="docker-compose.prod.yml"
    
    deploy_services
    wait_for_services
    initialize_database
    sleep 15
    run_production_health_checks
    
    log_success "🎉 Production deployment completed successfully!"
    echo ""
    echo "🔐 Production URLs:"
    echo "  • Main Site:      https://${DOMAIN}"
    echo "  • API:            https://${DOMAIN}/api/"
    echo "  • Dashboard:      https://${DOMAIN}/dashboard/"
    echo "  • Monitoring:     https://${DOMAIN}/monitoring/"
    echo ""
    echo "⚠️  Important:"
    echo "  1. Change default admin password immediately"
    echo "  2. Configure external monitoring"
    echo "  3. Set up log aggregation"
    echo "  4. Review security configuration"
    echo "  5. Test backup and recovery procedures"
}

# Check if running in production mode
if [ "${ENVIRONMENT}" = "production" ] || [ "${1}" = "production" ]; then
    main_production
else
    echo "Use 'ENVIRONMENT=production $0' or '$0 production' for production deployment"
    exit 1
fi
