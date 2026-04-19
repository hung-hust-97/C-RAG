# Migration Runbook: UUID-Based Document ID System

## Overview

This runbook provides step-by-step procedures for migrating from the legacy hash-based document identification system with track_id to the new UUID-based system with content_hash.

## Migration Summary

**What's Changing**:
- doc_id format: `doc-{md5_hash}` → `doc-{uuid4}`
- doc_id availability: After extraction → Immediately on upload
- Batch tracking: track_id → doc_ids array
- Duplicate detection: doc_id comparison → content_hash comparison

**Database Changes**:
- Add `content_hash` column (VARCHAR(32), nullable)
- Create index on `(workspace, content_hash)`
- Remove `track_id` column (optional)
- Delete track documents (`track-{track_id}`)
- Migrate existing doc_ids to UUID format

**API Changes**:
- Remove `track_id` from all responses
- Remove `/track_status/{track_id}` endpoint
- Add `doc_ids` array to batch responses
- Add `content_hash` field to document status

## Pre-Migration Checklist

### 1. Environment Preparation

- [ ] **Backup database**: Create full database backup
- [ ] **Backup configuration**: Save `.env` and config files
- [ ] **Document current state**: Record document counts and statistics
- [ ] **Test environment**: Set up staging environment for testing
- [ ] **Rollback plan**: Prepare rollback procedures
- [ ] **Maintenance window**: Schedule downtime if needed
- [ ] **Stakeholder notification**: Inform users of upcoming changes

### 2. Validation Checks

```bash
# Check database connectivity
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT 1;"

# Count existing documents
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT COUNT(*) FROM LIGHTRAG_DOC_STATUS;"

# Count track documents
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT COUNT(*) FROM LIGHTRAG_DOC_STATUS WHERE id LIKE 'track-%';"

# Check for documents without content_hash
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT COUNT(*) FROM LIGHTRAG_DOC_STATUS WHERE content_hash IS NULL;"

# Verify disk space (need ~2x current database size)
df -h /var/lib/postgresql/data
```

### 3. Backup Procedures

```bash
# Full database backup
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME \
  -F c -f lightrag_backup_$(date +%Y%m%d_%H%M%S).dump

# Verify backup
pg_restore --list lightrag_backup_*.dump | head -20

# Backup configuration files
tar -czf config_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
  .env config.ini docker-compose.yml

# Store backups securely
aws s3 cp lightrag_backup_*.dump s3://backups/lightrag/
aws s3 cp config_backup_*.tar.gz s3://backups/lightrag/
```

### 4. Dependency Checks

```bash
# Check Python version (3.8+)
python --version

# Check required packages
pip list | grep -E "(lightrag|psycopg2|asyncpg)"

# Check database version (PostgreSQL 12+)
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT version();"

# Check available disk space
du -sh rag_storage/
df -h
```

## Migration Steps

### Step 1: Stop Services

```bash
# Stop API server
systemctl stop lightrag-api
# or
docker-compose stop lightrag-api

# Stop Celery workers
systemctl stop lightrag-worker
# or
docker-compose stop celery-worker

# Verify services stopped
ps aux | grep -E "(lightrag|celery)"
```

### Step 2: Run Database Migration Script

The migration script is located at `.kiro/scripts/migrate_to_uuid_doc_id.py`.

```bash
# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export LIGHTRAG_DB_HOST=localhost
export LIGHTRAG_DB_PORT=5432
export LIGHTRAG_DB_NAME=lightrag
export LIGHTRAG_DB_USER=lightrag_user
export LIGHTRAG_DB_PASSWORD=your_password

# Run migration script
python .kiro/scripts/migrate_to_uuid_doc_id.py \
  --workspace default \
  --batch-size 1000 \
  --dry-run

# Review dry-run output, then run actual migration
python .kiro/scripts/migrate_to_uuid_doc_id.py \
  --workspace default \
  --batch-size 1000
```

**Migration Script Output**:
```
Starting migration for workspace: default
Step 1: Adding content_hash column... ✓
Step 2: Creating index on (workspace, content_hash)... ✓
Step 3: Removing track documents... ✓
  - Removed 150 track documents
Step 4: Migrating documents to UUID format...
  - Batch 1/5: Migrated 1000 documents
  - Batch 2/5: Migrated 1000 documents
  - Batch 3/5: Migrated 1000 documents
  - Batch 4/5: Migrated 1000 documents
  - Batch 5/5: Migrated 500 documents
Step 5: Updating chunk references... ✓
Step 6: Removing track_id column... ✓

Migration Summary:
  Total documents: 4500
  Migrated: 4500
  Errors: 0
  Track documents removed: 150
  Duration: 45.2 seconds
```

### Step 3: Verify Migration

```bash
# Check content_hash column exists
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT column_name, data_type, is_nullable 
   FROM information_schema.columns 
   WHERE table_name = 'lightrag_doc_status' 
   AND column_name = 'content_hash';"

# Check index exists
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT indexname, indexdef 
   FROM pg_indexes 
   WHERE tablename = 'lightrag_doc_status' 
   AND indexname LIKE '%content_hash%';"

# Verify all documents have UUID-based doc_ids
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT COUNT(*) FROM LIGHTRAG_DOC_STATUS 
   WHERE id ~ '^doc-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$';"

# Verify all documents have content_hash
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT COUNT(*) FROM LIGHTRAG_DOC_STATUS 
   WHERE content_hash IS NOT NULL;"

# Verify no track documents remain
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT COUNT(*) FROM LIGHTRAG_DOC_STATUS 
   WHERE id LIKE 'track-%';"

# Check chunk references updated
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT COUNT(DISTINCT full_doc_id) FROM LIGHTRAG_DOC_CHUNKS 
   WHERE full_doc_id ~ '^doc-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$';"
```

### Step 4: Deploy New Code

```bash
# Pull latest code
git pull origin main

# Install dependencies
pip install -e .[api]

# Verify installation
python -c "from lightrag import LightRAG; print('Import successful')"

# Update configuration if needed
cp .env.example .env.new
# Review and merge changes

# Build Docker images (if using Docker)
docker-compose build
```

### Step 5: Start Services

```bash
# Start API server
systemctl start lightrag-api
# or
docker-compose up -d lightrag-api

# Wait for API to be ready
curl http://localhost:8020/health

# Start Celery workers
systemctl start lightrag-worker
# or
docker-compose up -d celery-worker

# Verify services running
systemctl status lightrag-api lightrag-worker
# or
docker-compose ps
```

### Step 6: Post-Migration Validation

```bash
# Test single file upload
curl -X POST http://localhost:8020/upload \
  -F "file=@test.txt" \
  -F "workspace_id=default"

# Expected response:
# {
#   "status": "queued",
#   "doc_id": "doc-550e8400-e29b-41d4-a716-446655440000",
#   "content_hash": null
# }

# Test document status query
curl http://localhost:8020/documents/doc-550e8400-e29b-41d4-a716-446655440000

# Test batch upload
curl -X POST http://localhost:8020/scan \
  -H "Content-Type: application/json" \
  -d '{"workspace_id": "default"}'

# Expected response:
# {
#   "status": "queued",
#   "doc_ids": ["doc-uuid1", "doc-uuid2", ...],
#   "total_files": 3,
#   "queued_files": 3,
#   "failed_files": 0
# }

# Verify track_id not in response
curl -X POST http://localhost:8020/upload \
  -F "file=@test.txt" \
  -F "workspace_id=default" | grep -q "track_id" && echo "FAIL: track_id found" || echo "PASS: track_id removed"

# Test querying old hash-based doc_id (if any remain)
curl http://localhost:8020/documents/doc-a1b2c3d4e5f6789012345678901234567
```

## Rollback Procedures

### When to Rollback

Rollback if:
- Migration script fails with errors
- Data integrity issues detected
- Services fail to start after migration
- Critical functionality broken
- Performance degradation observed

### Rollback Steps

#### Option 1: Database Restore (Recommended)

```bash
# Stop services
systemctl stop lightrag-api lightrag-worker
# or
docker-compose down

# Restore database from backup
pg_restore -h $DB_HOST -U $DB_USER -d $DB_NAME \
  --clean --if-exists \
  lightrag_backup_YYYYMMDD_HHMMSS.dump

# Verify restoration
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT COUNT(*) FROM LIGHTRAG_DOC_STATUS;"

# Restore configuration
tar -xzf config_backup_YYYYMMDD_HHMMSS.tar.gz

# Checkout previous code version
git checkout <previous-commit-hash>

# Reinstall dependencies
pip install -e .[api]

# Start services
systemctl start lightrag-api lightrag-worker
# or
docker-compose up -d
```

#### Option 2: Manual Rollback (If Backup Unavailable)

```bash
# Stop services
systemctl stop lightrag-api lightrag-worker

# Run rollback script
python .kiro/scripts/rollback_uuid_migration.py \
  --workspace default

# Verify rollback
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT COUNT(*) FROM LIGHTRAG_DOC_STATUS 
   WHERE id LIKE 'doc-%' AND LENGTH(id) = 36;"

# Checkout previous code
git checkout <previous-commit-hash>

# Start services
systemctl start lightrag-api lightrag-worker
```

### Rollback Verification

```bash
# Verify hash-based doc_ids restored
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT id, content_hash FROM LIGHTRAG_DOC_STATUS LIMIT 5;"

# Test API with old format
curl -X POST http://localhost:8020/upload \
  -F "file=@test.txt" \
  -F "workspace_id=default"

# Should return track_id in old system
```

## Post-Migration Tasks

### 1. Monitoring Setup

```bash
# Monitor error logs
tail -f /var/log/lightrag/api.log | grep -i error

# Monitor migration metrics
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT 
     COUNT(*) as total_docs,
     COUNT(CASE WHEN id ~ '^doc-[0-9a-f]{8}-' THEN 1 END) as uuid_docs,
     COUNT(CASE WHEN content_hash IS NOT NULL THEN 1 END) as docs_with_hash
   FROM LIGHTRAG_DOC_STATUS;"

# Monitor API performance
curl http://localhost:8020/metrics
```

### 2. Client Updates

- [ ] **Notify clients**: Send migration announcement
- [ ] **Update documentation**: Publish API changes
- [ ] **Provide examples**: Share migration code samples
- [ ] **Support period**: Offer extended support for migration issues

### 3. Cleanup Tasks

```bash
# Remove old backups (after verification period)
find /backups -name "lightrag_backup_*.dump" -mtime +30 -delete

# Archive migration logs
tar -czf migration_logs_$(date +%Y%m%d).tar.gz /var/log/lightrag/migration/

# Update monitoring dashboards
# - Remove track_id metrics
# - Add content_hash metrics
# - Update doc_id format validation
```

### 4. Performance Tuning

```sql
-- Analyze tables for query optimization
ANALYZE LIGHTRAG_DOC_STATUS;
ANALYZE LIGHTRAG_DOC_CHUNKS;

-- Check index usage
SELECT 
  schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE tablename = 'lightrag_doc_status'
ORDER BY idx_scan DESC;

-- Vacuum tables
VACUUM ANALYZE LIGHTRAG_DOC_STATUS;
VACUUM ANALYZE LIGHTRAG_DOC_CHUNKS;
```

## Troubleshooting

### Issue 1: Migration Script Fails

**Symptoms**:
- Script exits with error
- Partial migration completed
- Database in inconsistent state

**Diagnosis**:
```bash
# Check migration logs
cat /var/log/lightrag/migration.log

# Check database state
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT 
     COUNT(*) as total,
     COUNT(CASE WHEN content_hash IS NOT NULL THEN 1 END) as with_hash,
     COUNT(CASE WHEN id LIKE 'track-%' THEN 1 END) as track_docs
   FROM LIGHTRAG_DOC_STATUS;"
```

**Resolution**:
1. Restore from backup
2. Fix underlying issue (disk space, permissions, etc.)
3. Retry migration with smaller batch size
4. Contact support if issue persists

### Issue 2: Services Won't Start

**Symptoms**:
- API server fails to start
- Celery workers crash
- Import errors

**Diagnosis**:
```bash
# Check service logs
journalctl -u lightrag-api -n 100
journalctl -u lightrag-worker -n 100

# Check Python imports
python -c "from lightrag import LightRAG"

# Check database connectivity
python -c "from lightrag.kg.postgres_impl import PGDocStatusStorage; print('OK')"
```

**Resolution**:
1. Verify all dependencies installed
2. Check environment variables
3. Verify database schema changes
4. Rollback if necessary

### Issue 3: Duplicate Detection Not Working

**Symptoms**:
- Duplicate documents not detected
- Multiple documents with same content
- content_hash not populated

**Diagnosis**:
```bash
# Check content_hash population
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT status, COUNT(*), 
   COUNT(CASE WHEN content_hash IS NULL THEN 1 END) as null_hash
   FROM LIGHTRAG_DOC_STATUS 
   GROUP BY status;"

# Check for duplicates
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT content_hash, COUNT(*) 
   FROM LIGHTRAG_DOC_STATUS 
   WHERE content_hash IS NOT NULL 
   GROUP BY content_hash 
   HAVING COUNT(*) > 1;"
```

**Resolution**:
1. Verify content_hash index exists
2. Check Celery worker logs for extraction errors
3. Manually trigger content_hash computation for affected documents
4. Verify duplicate detection logic in code

### Issue 4: Performance Degradation

**Symptoms**:
- Slow query responses
- High database CPU usage
- Timeout errors

**Diagnosis**:
```bash
# Check slow queries
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT query, calls, total_time, mean_time 
   FROM pg_stat_statements 
   WHERE query LIKE '%LIGHTRAG_DOC_STATUS%' 
   ORDER BY mean_time DESC 
   LIMIT 10;"

# Check index usage
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
  "SELECT * FROM pg_stat_user_indexes 
   WHERE tablename = 'lightrag_doc_status';"
```

**Resolution**:
1. Run ANALYZE on tables
2. Verify indexes exist and are used
3. Increase database resources if needed
4. Optimize query patterns

### Issue 5: Client Compatibility Issues

**Symptoms**:
- Client errors after migration
- Missing track_id errors
- Unexpected response format

**Diagnosis**:
```bash
# Check API responses
curl -v http://localhost:8020/upload \
  -F "file=@test.txt" \
  -F "workspace_id=default"

# Check for track_id references in logs
grep -r "track_id" /var/log/lightrag/
```

**Resolution**:
1. Review client migration guide
2. Update client code to use doc_id
3. Provide compatibility layer if needed
4. Extend support period for client updates

## Migration Checklist

### Pre-Migration
- [ ] Database backup completed
- [ ] Configuration backup completed
- [ ] Staging environment tested
- [ ] Rollback plan prepared
- [ ] Stakeholders notified
- [ ] Maintenance window scheduled

### Migration
- [ ] Services stopped
- [ ] Migration script executed
- [ ] Database changes verified
- [ ] New code deployed
- [ ] Services started
- [ ] Post-migration tests passed

### Post-Migration
- [ ] Monitoring configured
- [ ] Client notifications sent
- [ ] Documentation updated
- [ ] Support team briefed
- [ ] Performance metrics baseline established
- [ ] Cleanup tasks scheduled

### Verification
- [ ] UUID-based doc_ids generated
- [ ] content_hash populated
- [ ] track_id removed from responses
- [ ] Duplicate detection working
- [ ] Old documents queryable
- [ ] No track documents remain
- [ ] Performance acceptable

## Support Contacts

- **Technical Lead**: [contact info]
- **Database Admin**: [contact info]
- **DevOps Team**: [contact info]
- **On-Call Support**: [contact info]

## Additional Resources

- [API Documentation](./APIDocumentation.md)
- [Backward Compatibility Guide](./BackwardCompatibility.md)
- [Migration Script Source](.kiro/scripts/migrate_to_uuid_doc_id.py)
- [Rollback Script Source](.kiro/scripts/rollback_uuid_migration.py)

## Appendix A: Migration Script Reference

### Command-Line Options

```bash
python .kiro/scripts/migrate_to_uuid_doc_id.py --help

Options:
  --workspace TEXT        Workspace to migrate (default: all)
  --batch-size INTEGER    Batch size for processing (default: 1000)
  --dry-run              Run without making changes
  --skip-track-removal   Keep track documents
  --skip-column-removal  Keep track_id column
  --verbose              Enable verbose logging
```

### Environment Variables

```bash
# Database connection
LIGHTRAG_DB_HOST=localhost
LIGHTRAG_DB_PORT=5432
LIGHTRAG_DB_NAME=lightrag
LIGHTRAG_DB_USER=lightrag_user
LIGHTRAG_DB_PASSWORD=your_password

# Migration options
LIGHTRAG_MIGRATION_BATCH_SIZE=1000
LIGHTRAG_MIGRATION_TIMEOUT=3600
```

## Appendix B: SQL Queries

### Check Migration Status

```sql
-- Count documents by format
SELECT 
  CASE 
    WHEN id ~ '^doc-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$' 
      THEN 'UUID'
    WHEN id ~ '^doc-[0-9a-f]{32}$' 
      THEN 'Hash'
    WHEN id LIKE 'track-%' 
      THEN 'Track'
    ELSE 'Other'
  END as doc_type,
  COUNT(*) as count
FROM LIGHTRAG_DOC_STATUS
GROUP BY doc_type;
```

### Find Documents Without content_hash

```sql
SELECT id, status, created_at, updated_at
FROM LIGHTRAG_DOC_STATUS
WHERE content_hash IS NULL
  AND status NOT IN ('EXTRACTING', 'FAILED')
ORDER BY created_at DESC
LIMIT 100;
```

### Check Duplicate Detection

```sql
SELECT content_hash, COUNT(*) as duplicate_count, 
       ARRAY_AGG(id) as doc_ids
FROM LIGHTRAG_DOC_STATUS
WHERE content_hash IS NOT NULL
GROUP BY content_hash
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC;
```

## Appendix C: Testing Procedures

### Integration Test Suite

```bash
# Run full integration tests
python -m pytest tests/integration/test_uuid_migration.py -v

# Test specific scenarios
python -m pytest tests/integration/test_uuid_migration.py::test_single_upload -v
python -m pytest tests/integration/test_uuid_migration.py::test_batch_upload -v
python -m pytest tests/integration/test_uuid_migration.py::test_duplicate_detection -v
```

### Manual Test Cases

1. **Single File Upload**
   - Upload file
   - Verify doc_id format (UUID)
   - Verify no track_id in response
   - Query document status
   - Wait for extraction
   - Verify content_hash populated

2. **Batch Upload**
   - Upload multiple files
   - Verify doc_ids array returned
   - Verify no track_id in response
   - Query each document
   - Verify all processed

3. **Duplicate Detection**
   - Upload same file twice
   - Verify unique doc_ids
   - Wait for extraction
   - Verify second marked as DUPLICATED
   - Verify reference to original

4. **Old Document Query**
   - Query hash-based doc_id
   - Verify document returned
   - Verify content_hash present

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-04-17 | Migration Team | Initial runbook |
| 1.1 | TBD | | Post-migration updates |
