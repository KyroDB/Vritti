# CI/CD Pipeline Documentation
**- Automated Testing and Deployment**

Complete guide to the EpisodicMemory CI/CD pipeline built with GitHub Actions.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [CI Pipeline](#ci-pipeline)
- [CD Pipeline](#cd-pipeline)
- [Secrets Configuration](#secrets-configuration)
- [Manual Deployment](#manual-deployment)
- [Rollback Procedures](#rollback-procedures)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Overview

EpisodicMemory uses a two-pipeline CI/CD approach:

- **CI (Continuous Integration)**: Runs on every push, validates code quality, tests, and security
- **CD (Continuous Deployment)**: Deploys to staging (on develop branch) and production (on version tags)

### Pipeline Triggers

| Trigger | CI | CD |
|---------|----|----|
| Push to any branch | ✓ | - |
| Push to develop | ✓ | ✓ (staging) |
| Pull request to main/develop | ✓ | - |
| Tag push (v*.*.*) | ✓ | ✓ (production) |

### Pipeline Duration

- **CI Pipeline**: ~8-12 minutes
  - Lint: 1-2 minutes
  - Test: 3-5 minutes
  - Security: 2-3 minutes
  - Build: 2-3 minutes

- **CD Pipeline**: ~15-20 minutes
  - Build: 3-5 minutes
  - Scan: 2-3 minutes
  - Deploy: 8-10 minutes

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GitHub Repository                     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ├─ Push to any branch
                 │  └─> CI Pipeline
                 │      ├─ Lint (black, ruff)
                 │      ├─ Test (pytest + coverage)
                 │      ├─ Security (safety, bandit)
                 │      └─ Build Check (Docker)
                 │
                 ├─ Push to develop
                 │  └─> CD Pipeline (Staging)
                 │      ├─ Build & Push Image
                 │      ├─ Security Scan (Trivy)
                 │      └─ Deploy to Staging
                 │
                 └─ Tag push (v*.*.*)
                    └─> CD Pipeline (Production)
                        ├─ Build & Push Image
                        ├─ Security Scan (Trivy)
                        └─ Deploy to Production
                            └─ Manual Approval Required
```

---

## CI Pipeline

### Jobs

#### 1. Lint (Code Quality)

Validates code formatting and style:

```yaml
- black --check src/ tests/
- ruff check src/ tests/
```

**Failure Criteria**:
- Code not formatted with black
- Ruff linting errors

#### 2. Test (Test Suite)

Runs full test suite with coverage:

```yaml
- pytest tests/ -v --tb=short --cov=src --cov-report=xml
```

**Artifacts**:
- Coverage report (uploaded to Codecov)

**Failure Criteria**:
- Any test failures
- Coverage below threshold (not enforced yet)

#### 3. Security (Security Scanning)

Scans dependencies and code for vulnerabilities:

```yaml
- safety check --json  # Known CVEs in dependencies
- bandit -r src/       # Security issues in code
```

**Artifacts**:
- Bandit JSON report

**Failure Criteria**:
- Currently set to `continue-on-error: true` for visibility
- Will be enforced in production with thresholds

#### 4. Build Check (Docker Build)

Verifies Docker image builds and health checks work:

```yaml
- docker build -t episodic-memory:test .
- docker run -d --name test-container episodic-memory:test
- curl -f http://localhost:8000/health/liveness
```

**Failure Criteria**:
- Docker build fails
- Container won't start
- Health endpoints fail

#### 5. CI Passed (Aggregator)

Checks all jobs succeeded:

```yaml
if: always()
needs: [lint, test, security, build-check]
```

---

## CD Pipeline

### Jobs

#### 1. Build (Build & Push Image)

Builds multi-arch Docker image and pushes to registry:

```yaml
- Docker Buildx with cache
- Multi-platform build (linux/amd64)
- Generate SBOM (Software Bill of Materials)
- Push to Docker Hub
```

**Outputs**:
- `image_tag`: Full image reference
- `image_digest`: SHA256 digest

**Artifacts**:
- SBOM (SPDX JSON format, 30-day retention)

#### 2. Scan (Security Scan)

Scans container image for vulnerabilities:

```yaml
- Trivy vulnerability scanner
- SARIF report upload to GitHub Security
- Fail on CRITICAL or HIGH severity
```

**Artifacts**:
- Trivy SARIF report (30-day retention)

**Failure Criteria**:
- CRITICAL or HIGH severity vulnerabilities

#### 3. Deploy-Staging (Staging Deployment)

Deploys to staging environment (develop branch only):

```yaml
- kubectl set image deployment/episodic-memory-api
- kubectl rollout status (5-minute timeout)
- Health check verification
- Smoke tests
- Auto-rollback on failure
```

**Environment**:
- Name: `staging`
- URL: `https://staging-api.episodicmemory.dev`
- Namespace: `episodic-memory-staging`

**Health Checks**:
1. Liveness probe
2. Readiness probe
3. Smoke tests (health, metrics)

**Rollback**:
- Automatic on any failure
- Restores previous deployment

#### 4. Deploy-Production (Production Deployment)

Deploys to production (version tags only):

```yaml
- Manual approval required (GitHub Environment)
- Pre-deployment backup
- kubectl set image deployment/episodic-memory-api
- kubectl rollout status (10-minute timeout)
- Health check verification
- 5-minute error rate monitoring
- Auto-rollback on failure
- GitHub Release creation
```

**Environment**:
- Name: `production`
- URL: `https://api.episodicmemory.com`
- Namespace: `episodic-memory-production`
- **Requires manual approval**

**Safety Checks**:
1. Pre-deployment backup (90-day retention)
2. Gradual rollout (rolling update)
3. Health verification
4. Error rate monitoring (5 minutes)
5. Auto-rollback if error rate > 1%

---

## Secrets Configuration

### Required Secrets

Configure these secrets in GitHub repository settings:

#### Docker Registry

```
DOCKER_USERNAME     # Docker Hub username
DOCKER_PASSWORD     # Docker Hub access token (NOT password)
```

**Setup**:
1. Create Docker Hub account
2. Generate access token: Account Settings → Security → New Access Token
3. Add to GitHub: Settings → Secrets and variables → Actions → New repository secret

#### Kubernetes Staging

```
KUBECONFIG_STAGING  # Base64-encoded kubeconfig for staging cluster
```

**Setup**:
```bash
# 1. Get kubeconfig for staging cluster
kubectl config view --minify --flatten > staging-kubeconfig.yaml

# 2. Base64 encode
cat staging-kubeconfig.yaml | base64 | pbcopy

# 3. Add to GitHub secrets as KUBECONFIG_STAGING
```

#### Kubernetes Production

```
KUBECONFIG_PRODUCTION  # Base64-encoded kubeconfig for production cluster
```

**Setup**:
```bash
# 1. Get kubeconfig for production cluster
kubectl config view --minify --flatten > production-kubeconfig.yaml

# 2. Base64 encode
cat production-kubeconfig.yaml | base64 | pbcopy

# 3. Add to GitHub secrets as KUBECONFIG_PRODUCTION
```

### Optional Secrets

```
CODECOV_TOKEN       # For private repository coverage uploads
SLACK_WEBHOOK       # For deployment notifications
```

---

## Manual Deployment

Use the `scripts/deploy.sh` script for manual deployments:

### Deploy to Staging

```bash
./scripts/deploy.sh --env staging --tag develop
```

### Deploy to Production

```bash
./scripts/deploy.sh --env production --tag v1.0.0
```

### Dry Run

Preview what would be deployed without applying:

```bash
./scripts/deploy.sh --env staging --dry-run
```

### Check Deployment Status

```bash
./scripts/deploy.sh --env production --status
```

### Script Options

```bash
./scripts/deploy.sh [OPTIONS]

Options:
  --env ENV        Target environment (staging|production, required)
  --tag TAG        Image tag to deploy (default: environment-specific)
  --dry-run        Show what would be deployed without applying
  --rollback       Rollback to previous deployment
  --status         Show deployment status
  -h, --help       Show this help message
```

---

## Rollback Procedures

### Automated Rollback

CD pipeline automatically rolls back on:
- Health check failures
- Deployment timeout (>10 minutes)
- High error rate (>1% for 5 minutes in production)

### Manual Rollback via Script

```bash
./scripts/deploy.sh --env production --rollback
```

### Manual Rollback via kubectl

```bash
# Undo last deployment
kubectl rollout undo deployment/episodic-memory-api \
  --namespace=episodic-memory-production

# Rollback to specific revision
kubectl rollout undo deployment/episodic-memory-api \
  --namespace=episodic-memory-production \
  --to-revision=3

# Check rollout status
kubectl rollout status deployment/episodic-memory-api \
  --namespace=episodic-memory-production
```

### View Rollout History

```bash
kubectl rollout history deployment/episodic-memory-api \
  --namespace=episodic-memory-production
```

---

## Monitoring

### GitHub Actions

View pipeline runs:
```
Repository → Actions → Select workflow
```

**Key Metrics**:
- Success rate
- Duration trends
- Failure patterns

### Deployment Status

Check deployment health:

```bash
# Staging
kubectl get pods -n episodic-memory-staging
kubectl logs -n episodic-memory-staging -l app=episodic-memory-api --tail=100

# Production
kubectl get pods -n episodic-memory-production
kubectl logs -n episodic-memory-production -l app=episodic-memory-api --tail=100
```

### Service Health

```bash
# Port forward to pod
kubectl port-forward -n episodic-memory-production \
  deployment/episodic-memory-api 8000:8000

# Check endpoints
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

### GitHub Security Alerts

Trivy scan results appear in:
```
Repository → Security → Code scanning alerts
```

---

## Troubleshooting

### CI Pipeline Failures

#### Lint Failures

**Error**: `black --check src/ tests/` fails

**Fix**:
```bash
black src/ tests/
git add .
git commit -m "Format code with black"
```

#### Test Failures

**Error**: `pytest` fails

**Fix**:
```bash
# Run tests locally
pytest tests/ -v

# Debug specific test
pytest tests/test_specific.py::test_function -v -s
```

#### Security Scan Failures

**Error**: `safety check` finds vulnerabilities

**Fix**:
```bash
# View vulnerabilities
safety check

# Update dependencies
pip install --upgrade package-name
pip freeze > requirements.txt
```

#### Build Check Failures

**Error**: Docker build fails

**Fix**:
```bash
# Build locally to debug
docker build -t episodic-memory:test .

# Check logs
docker logs test-container
```

### CD Pipeline Failures

#### Build Failures

**Error**: Image build fails

**Fix**:
1. Check Dockerfile syntax
2. Verify base image availability
3. Check build context size
4. Review build logs in Actions

#### Trivy Scan Failures

**Error**: CRITICAL or HIGH vulnerabilities found

**Fix**:
```bash
# Scan locally
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image episodic-memory:latest

# Update base image or dependencies
# Rebuild and rescan
```

#### Deployment Failures

**Error**: `kubectl rollout status` times out

**Fix**:
```bash
# Check pod status
kubectl get pods -n episodic-memory-staging

# Describe pod for events
kubectl describe pod <pod-name> -n episodic-memory-staging

# Check logs
kubectl logs <pod-name> -n episodic-memory-staging

# Common issues:
# - Image pull errors (check credentials)
# - Resource limits (check node capacity)
# - Health probe failures (check endpoints)
```

#### Health Check Failures

**Error**: Liveness/readiness probes fail

**Fix**:
```bash
# Port forward to pod
kubectl port-forward <pod-name> 8000:8000 -n episodic-memory-staging

# Test endpoints manually
curl -v http://localhost:8000/health/liveness
curl -v http://localhost:8000/health/readiness

# Check application logs
kubectl logs <pod-name> -n episodic-memory-staging
```

### Secret Issues

#### Invalid Kubeconfig

**Error**: `kubectl` authentication fails

**Fix**:
```bash
# Regenerate kubeconfig
kubectl config view --minify --flatten > kubeconfig.yaml

# Verify it works locally
KUBECONFIG=kubeconfig.yaml kubectl get pods

# Re-encode and update secret
cat kubeconfig.yaml | base64 | pbcopy
# Update in GitHub secrets
```

#### Docker Registry Authentication

**Error**: `docker push` fails

**Fix**:
1. Verify Docker Hub credentials
2. Regenerate access token
3. Update `DOCKER_PASSWORD` secret
4. Ensure token has write permissions

### Performance Issues

#### Slow Builds

**Optimization**:
- Build cache is enabled (GitHub Actions cache)
- Multi-stage builds reduce image size
- Buildx with layer caching

**Check**:
```bash
# Locally verify cache usage
docker build --cache-from episodic-memory:latest .
```

#### Slow Deployments

**Optimization**:
- Readiness probe reduces downtime
- Rolling update strategy (maxSurge: 1, maxUnavailable: 0)
- Image already cached on nodes after first pull

**Check**:
```bash
# View rollout progress
kubectl rollout status deployment/episodic-memory-api -n <namespace> -w
```

---

## Best Practices

### Development Workflow

1. **Create feature branch**:
   ```bash
   git checkout -b feature/new-feature
   ```

2. **Make changes and test locally**:
   ```bash
   pytest tests/
   black src/ tests/
   ruff check src/ tests/
   ```

3. **Push to trigger CI**:
   ```bash
   git push origin feature/new-feature
   ```

4. **Create pull request**:
   - CI runs automatically
   - Address any failures
   - Request review

5. **Merge to develop**:
   - Merging triggers CD to staging
   - Verify in staging environment

6. **Create release tag for production**:
   ```bash
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   ```

### Release Workflow

1. **Prepare release**:
   - Ensure all features merged to develop
   - Verify staging deployment successful
   - Run manual testing in staging

2. **Create release tag**:
   ```bash
   git checkout main
   git merge develop
   git tag -a v1.0.0 -m "Release v1.0.0: Feature X, Bug fix Y"
   git push origin main --tags
   ```

3. **Monitor deployment**:
   - CD pipeline triggers automatically
   - Manual approval required for production
   - Approve in GitHub Actions UI
   - Monitor rollout progress

4. **Verify production**:
   - Check health endpoints
   - Monitor error rates in Grafana
   - Review logs for issues

5. **Document release**:
   - GitHub Release created automatically
   - Add release notes
   - Update CHANGELOG.md

### Emergency Hotfix

1. **Create hotfix branch**:
   ```bash
   git checkout -b hotfix/critical-bug main
   ```

2. **Fix and test**:
   ```bash
   # Make fix
   pytest tests/
   git commit -m "Fix critical bug"
   ```

3. **Fast-track to production**:
   ```bash
   git checkout main
   git merge hotfix/critical-bug
   git tag -a v1.0.1 -m "Hotfix: Critical bug"
   git push origin main --tags
   ```

4. **Backport to develop**:
   ```bash
   git checkout develop
   git merge hotfix/critical-bug
   git push origin develop
   ```

---

## Security Considerations

### Image Scanning

- **Trivy**: Scans for CVEs in base images and dependencies
- **Frequency**: Every build
- **Threshold**: Fail on CRITICAL or HIGH
- **Reporting**: GitHub Security tab

### Secrets Management

- **Never commit secrets**: Use .gitignore for .env files
- **Rotate regularly**: Update kubeconfigs and tokens quarterly
- **Least privilege**: Service accounts with minimal permissions
- **Audit**: Review GitHub Actions logs for secret access

### Supply Chain Security

- **SBOM Generation**: Software Bill of Materials for each build
- **Dependency Scanning**: Safety checks for known vulnerabilities
- **Pinned Versions**: requirements.txt with exact versions
- **Reproducible Builds**: Multi-stage Dockerfile with checksums

---

## Future Enhancements

### Planned Improvements

1. **Blue-Green Deployment**: Zero-downtime with instant rollback
2. **Canary Releases**: Gradual rollout to subset of users
3. **Performance Testing**: Automated load testing in CD pipeline
4. **Slack Notifications**: Real-time deployment status
5. **ArgoCD Integration**: GitOps-based deployment
6. **Multi-region**: Deploy to multiple Kubernetes clusters

### Metrics to Track

- Deployment frequency
- Lead time for changes
- Mean time to recovery (MTTR)
- Change failure rate
- CI/CD pipeline success rate

---

## Support

For issues with CI/CD pipeline:

1. Check this documentation
2. Review GitHub Actions logs
3. Check Kubernetes events and logs
4. Contact DevOps team
