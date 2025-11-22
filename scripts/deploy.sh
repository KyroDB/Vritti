#!/bin/bash
# Deployment script for EpisodicMemory
# Manual deployment automation
#
# Usage:
#   ./scripts/deploy.sh [OPTIONS]
#
# Options:
#   --env ENV: Target environment (staging|production, required)
#   --tag TAG: Image tag to deploy (default: environment-specific)
#   --dry-run: Show what would be deployed without applying
#   --rollback: Rollback to previous deployment
#   --status: Show deployment status
#   -h, --help: Show this help message

set -euo pipefail

# Default values
ENVIRONMENT=""
IMAGE_TAG=""
DRY_RUN=false
ROLLBACK=false
STATUS=false
REGISTRY="docker.io"
IMAGE_NAME="episodic-memory"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --rollback)
            ROLLBACK=true
            shift
            ;;
        --status)
            STATUS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env ENV        Target environment (staging|production, required)"
            echo "  --tag TAG        Image tag to deploy (default: environment-specific)"
            echo "  --dry-run        Show what would be deployed without applying"
            echo "  --rollback       Rollback to previous deployment"
            echo "  --status         Show deployment status"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate environment
if [ -z "$ENVIRONMENT" ]; then
    log_error "Environment is required. Use --env staging or --env production"
    exit 1
fi

if [ "$ENVIRONMENT" != "staging" ] && [ "$ENVIRONMENT" != "production" ]; then
    log_error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
    exit 1
fi

# Set default image tag based on environment
if [ -z "$IMAGE_TAG" ]; then
    if [ "$ENVIRONMENT" = "staging" ]; then
        IMAGE_TAG="develop"
    else
        IMAGE_TAG="latest"
    fi
fi

# Kubernetes namespace
NAMESPACE="episodic-memory-$ENVIRONMENT"
DEPLOYMENT_NAME="episodic-memory-api"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if kustomize is available
if ! command -v kustomize &> /dev/null; then
    log_error "kustomize is not installed. Please install kustomize first."
    exit 1
fi

# Show deployment status
if [ "$STATUS" = true ]; then
    log_info "Deployment status for $ENVIRONMENT:"
    echo ""

    kubectl get deployment $DEPLOYMENT_NAME -n $NAMESPACE
    echo ""

    kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE
    echo ""

    log_info "Pods:"
    kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT_NAME
    echo ""

    log_info "Recent events:"
    kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' | tail -10

    exit 0
fi

# Rollback deployment
if [ "$ROLLBACK" = true ]; then
    log_warn "Rolling back deployment in $ENVIRONMENT..."

    kubectl rollout undo deployment/$DEPLOYMENT_NAME -n $NAMESPACE

    log_info "Waiting for rollback to complete..."
    kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=10m

    log_info "Rollback completed successfully"

    kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT_NAME

    exit 0
fi

# Deploy new version
log_info "==========================================="
log_info "Deploying EpisodicMemory"
log_info "==========================================="
log_info "Environment: $ENVIRONMENT"
log_info "Namespace: $NAMESPACE"
log_info "Image tag: $IMAGE_TAG"
log_info "Dry run: $DRY_RUN"
log_info "==========================================="
echo ""

# Confirm production deployment
if [ "$ENVIRONMENT" = "production" ] && [ "$DRY_RUN" = false ]; then
    log_warn "You are about to deploy to PRODUCTION"
    read -p "Are you sure? (yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        log_info "Deployment cancelled"
        exit 0
    fi
fi

# Build kustomization
cd "$(dirname "$0")/.."
KUSTOMIZE_DIR="k8s/$ENVIRONMENT"

if [ ! -d "$KUSTOMIZE_DIR" ]; then
    log_error "Kustomization directory not found: $KUSTOMIZE_DIR"
    exit 1
fi

log_info "Building kustomization from $KUSTOMIZE_DIR..."

# Update image tag in kustomization
cd "$KUSTOMIZE_DIR"
kustomize edit set image episodic-memory="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
cd ../..

# Generate manifests
MANIFESTS=$(kustomize build "$KUSTOMIZE_DIR")

if [ "$DRY_RUN" = true ]; then
    log_info "Dry run mode - manifests that would be applied:"
    echo ""
    echo "$MANIFESTS"
    exit 0
fi

# Create namespace if it doesn't exist
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    log_info "Creating namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE"
fi

# Pre-deployment backup
log_info "Creating pre-deployment backup..."
BACKUP_FILE="deployment-backup-$(date +%Y%m%d-%H%M%S).yaml"
kubectl get deployment $DEPLOYMENT_NAME -n $NAMESPACE -o yaml > "$BACKUP_FILE" 2>/dev/null || true

if [ -f "$BACKUP_FILE" ]; then
    log_info "Backup saved to: $BACKUP_FILE"
fi

# Apply manifests
log_info "Applying manifests to $NAMESPACE..."
echo "$MANIFESTS" | kubectl apply -f -

# Wait for rollout
log_info "Waiting for rollout to complete..."
kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=10m

# Verify deployment
log_info "Verifying deployment health..."

# Get pod name
POD_NAME=$(kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT_NAME -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POD_NAME" ]; then
    log_error "No pods found for deployment"
    exit 1
fi

log_info "Testing health endpoints on pod: $POD_NAME"

# Port forward and test
kubectl port-forward $POD_NAME 8000:8000 -n $NAMESPACE &
PF_PID=$!

# Wait for port forward
sleep 5

# Test liveness endpoint
if curl -f http://localhost:8000/health/liveness > /dev/null 2>&1; then
    log_info "Liveness check: PASSED"
else
    log_error "Liveness check: FAILED"
    kill $PF_PID
    exit 1
fi

# Test readiness endpoint
if curl -f http://localhost:8000/health/readiness > /dev/null 2>&1; then
    log_info "Readiness check: PASSED"
else
    log_error "Readiness check: FAILED"
    kill $PF_PID
    exit 1
fi

# Cleanup port forward
kill $PF_PID

# Show deployment status
log_info ""
log_info "Deployment completed successfully!"
log_info ""
kubectl get deployment $DEPLOYMENT_NAME -n $NAMESPACE
echo ""
kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT_NAME

log_info ""
log_info "==========================================="
log_info "Deployment Summary"
log_info "==========================================="
log_info "Environment: $ENVIRONMENT"
log_info "Image: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
log_info "Namespace: $NAMESPACE"
log_info "Backup: $BACKUP_FILE"
log_info "==========================================="
