# Backward Compatibility Guide

## Overview

This guide describes the backward compatibility strategy for migrating from the legacy track_id system to the new UUID-based document identification system.

## Compatibility Matrix

| Feature | Legacy System | Current System | Compatibility Status |
|---------|--------------|----------------|---------------------|
| Hash-based doc_id queries | `doc-{md5_hash}` | `doc-{uuid4}` | ✅ Supported during transition |
| UUID-based doc_id queries | Not supported | `doc-{uuid4}` | ✅ Fully supported |
| track_id in responses | Returned | Not returned | ⚠️ Deprecated (removed) |
| `/track_status/{track_id}` endpoint | Available | Not available | ⚠️ Deprecated (removed) |
| content_hash field | Not present | Present | ✅ Backward compatible (nullable) |
| Immediate doc_id availability | No | Yes | ✅ Enhancement |

## Deprecation Timeline

### Phase 1: Introduction (Version X.X.X)

**Status**: Current

**Changes**:
- UUID-based doc_id system introduced
- content_hash field added to database schema
- New uploads receive UUID-based doc_ids
- Old hash-based doc_ids continue to work

**Compatibility**:
- ✅ Old documents queryable with hash-based doc_ids
- ✅ New documents use UUID-based doc_ids
- ✅ Both formats supported in all query endpoints
- ⚠️ track_id removed from API responses
- ⚠️ `/track_status/{track_id}` endpoint removed

**Client Impact**:
- **Breaking**: Clients using track_id must migrate to doc_id tracking
- **Non-breaking**: Clients querying existing documents continue to work

### Phase 2: Transition Period (Version X.X+1 to X.X+3)

**Status**: Future

**Changes**:
- Migration tools provided for bulk doc_id updates
- Monitoring and logging for hash-based doc_id usage
- Documentation and migration guides published

**Compatibility**:
- ✅ Hash-based doc_ids continue to work
- ✅ Warnings logged for hash-based doc_id queries
- ✅ Migration utilities available

**Client Impact**:
- **Recommended**: Update client code to handle UUID-based doc_ids
- **Optional**: Migrate stored hash-based doc_ids to UUID format

### Phase 3: Deprecation Notice (Version X.X+4)

**Status**: Future (TBD)

**Changes**:
- Formal deprecation notice for hash-based doc_id support
- Deprecation warnings in API responses
- Extended support timeline announced

**Compatibility**:
- ✅ Hash-based doc_ids still work
- ⚠️ Deprecation warnings in responses
- ⚠️ Clients encouraged to migrate

**Client Impact**:
- **Required**: Plan migration from hash-based doc_ids
- **Timeline**: Minimum 6 months notice before removal

### Phase 4: Removal (Version X.X+6+)

**Status**: Future (TBD, minimum 6 months after Phase 3)

**Changes**:
- Hash-based doc_id support removed
- Only UUID-based doc_ids accepted

**Compatibility**:
- ❌ Hash-based doc_ids no longer work
- ✅ UUID-based doc_ids fully supported

**Client Impact**:
- **Required**: All clients must use UUID-based doc_ids
- **Migration**: Complete before this version

## Querying Old Documents

### Hash-Based doc_id Support

Documents created before migration retain their hash-based doc_ids and remain queryable:

```python
# Old document with hash-based doc_id
doc_id = "doc-a1b2c3d4e5f6789012345678901234567"

# Query still works
status = await storage.get_by_id(doc_id)
# Returns: {
#   "doc_id": "doc-a1b2c3d4e5f6789012345678901234567",
#   "status": "PROCESSED",
#   "content_hash": "a1b2c3d4e5f6789012345678901234567",
#   ...
# }
```

### Mixed Format Support

Systems can have both hash-based and UUID-based doc_ids:

```python
# Query multiple documents with mixed formats
doc_ids = [
    "doc-a1b2c3d4e5f6789012345678901234567",  # Hash-based (old)
    "doc-550e8400-e29b-41d4-a716-446655440000"  # UUID-based (new)
]

statuses = await storage.get_by_ids(doc_ids)
# Both queries succeed
```

### Format Detection

Clients can detect doc_id format:

```python
import re

def is_uuid_doc_id(doc_id: str) -> bool:
    """Check if doc_id is UUID-based format."""
    uuid_pattern = r'^doc-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
    return bool(re.match(uuid_pattern, doc_id, re.IGNORECASE))

def is_hash_doc_id(doc_id: str) -> bool:
    """Check if doc_id is hash-based format."""
    hash_pattern = r'^doc-[0-9a-f]{32}$'
    return bool(re.match(hash_pattern, doc_id, re.IGNORECASE))

# Usage
doc_id = "doc-550e8400-e29b-41d4-a716-446655440000"
if is_uuid_doc_id(doc_id):
    print("UUID-based doc_id (new format)")
elif is_hash_doc_id(doc_id):
    print("Hash-based doc_id (legacy format)")
```

## Deprecated Endpoints

### `/track_status/{track_id}` - REMOVED

**Status**: Removed in version X.X.X

**Reason**: track_id system eliminated in favor of individual document tracking

**Migration**:

**Before**:
```python
# Old approach using track_id
response = await client.post("/upload", files={"file": file_data})
track_id = response.json()["track_id"]

# Poll track status
track_status = await client.get(f"/track_status/{track_id}")
documents = track_status.json()["documents"]
```

**After**:
```python
# New approach using doc_id
response = await client.post("/upload", files={"file": file_data})
doc_id = response.json()["doc_id"]  # Available immediately!

# Poll document status
doc_status = await client.get(f"/documents/{doc_id}")
```

**Batch Upload Migration**:

**Before**:
```python
# Old batch approach
response = await client.post("/scan", json={"workspace_id": "my-workspace"})
track_id = response.json()["track_id"]

# Poll track status for all documents
track_status = await client.get(f"/track_status/{track_id}")
for doc in track_status.json()["documents"]:
    print(f"Document {doc['doc_id']}: {doc['status']}")
```

**After**:
```python
# New batch approach
response = await client.post("/scan", json={"workspace_id": "my-workspace"})
doc_ids = response.json()["doc_ids"]  # Array of doc_ids

# Poll each document individually
for doc_id in doc_ids:
    doc_status = await client.get(f"/documents/{doc_id}")
    print(f"Document {doc_id}: {doc_status.json()['status']}")
```

## Deprecated Fields

### `track_id` Field - REMOVED

**Status**: Removed from all API responses in version X.X.X

**Affected Endpoints**:
- `POST /upload` - No longer returns `track_id`
- `POST /scan` - No longer returns `track_id`
- `POST /insert_text` - No longer returns `track_id`
- `POST /insert_texts` - No longer returns `track_id`
- `GET /documents/{doc_id}` - No longer includes `track_id`

**Migration**:

Update response parsing to use `doc_id` or `doc_ids`:

```python
# Before
response = await upload_file(file)
track_id = response["track_id"]  # ❌ Field removed

# After
response = await upload_file(file)
doc_id = response["doc_id"]  # ✅ Use doc_id instead
```

```python
# Before (batch)
response = await scan_files(workspace_id)
track_id = response["track_id"]  # ❌ Field removed

# After (batch)
response = await scan_files(workspace_id)
doc_ids = response["doc_ids"]  # ✅ Use doc_ids array instead
```

### Deprecation Warnings

If your client code references `track_id`, you'll see errors:

```python
# This will raise KeyError
response = await upload_file(file)
track_id = response["track_id"]  # KeyError: 'track_id'

# Fix: Use doc_id instead
doc_id = response["doc_id"]
```

## Database Schema Compatibility

### content_hash Column

**Added**: Version X.X.X

**Type**: `VARCHAR(32)`, nullable

**Purpose**: Store MD5 hash of document content for duplicate detection

**Backward Compatibility**:
- ✅ Nullable to support existing documents
- ✅ Populated during migration for old documents
- ✅ Automatically populated for new documents after extraction

**Migration Behavior**:
```sql
-- Old documents (before migration)
SELECT doc_id, content_hash FROM LIGHTRAG_DOC_STATUS 
WHERE doc_id = 'doc-a1b2c3d4...';
-- Returns: doc_id='doc-a1b2c3d4...', content_hash=NULL

-- After migration
SELECT doc_id, content_hash FROM LIGHTRAG_DOC_STATUS 
WHERE doc_id = 'doc-550e8400...';
-- Returns: doc_id='doc-550e8400...', content_hash='a1b2c3d4...'
```

### track_id Column - REMOVED

**Status**: Removed during migration (optional, can be kept for backward compatibility)

**Migration Options**:

**Option 1: Remove column** (recommended for clean schema):
```sql
ALTER TABLE LIGHTRAG_DOC_STATUS DROP COLUMN IF EXISTS track_id;
```

**Option 2: Keep column** (for backward compatibility):
```sql
-- Keep column but stop populating it
-- Existing values preserved, new documents have NULL
```

**Recommendation**: Remove column unless you have external systems that depend on it.

## Client Migration Strategies

### Strategy 1: Immediate Migration (Recommended)

**Timeline**: Update client code before deploying new server version

**Steps**:
1. Update client code to use `doc_id` instead of `track_id`
2. Update batch upload handling to use `doc_ids` array
3. Remove references to `/track_status/{track_id}` endpoint
4. Test with staging environment
5. Deploy server update
6. Deploy client update

**Pros**:
- Clean cutover
- No compatibility layer needed
- Simpler codebase

**Cons**:
- Requires coordinated deployment
- Downtime during update

### Strategy 2: Gradual Migration

**Timeline**: Support both systems during transition

**Steps**:
1. Deploy server update (track_id removed)
2. Update client code to handle missing `track_id` gracefully
3. Gradually migrate client features to use `doc_id`
4. Monitor for errors and edge cases
5. Complete migration when all features updated

**Implementation**:
```python
# Graceful fallback during transition
response = await upload_file(file)

# Try new format first, fall back to old format
doc_id = response.get("doc_id")
if not doc_id:
    # Old server version, use track_id
    track_id = response.get("track_id")
    # Wait for extraction and get doc_id
    doc_id = await wait_for_doc_id(track_id)

# Continue with doc_id
await poll_document_status(doc_id)
```

**Pros**:
- No downtime
- Gradual rollout
- Easy rollback

**Cons**:
- More complex client code
- Longer transition period
- Compatibility layer maintenance

### Strategy 3: Feature Flag Migration

**Timeline**: Use feature flags to control migration

**Steps**:
1. Add feature flag for UUID system
2. Deploy server with both systems (if feasible)
3. Enable UUID system for subset of users
4. Monitor and validate
5. Gradually increase rollout
6. Remove old system when complete

**Implementation**:
```python
# Feature flag check
if feature_flags.is_enabled("uuid_doc_id"):
    # Use new UUID system
    doc_id = response["doc_id"]
    await poll_document_status(doc_id)
else:
    # Use old track_id system
    track_id = response["track_id"]
    await poll_track_status(track_id)
```

**Pros**:
- Controlled rollout
- Easy rollback
- A/B testing possible

**Cons**:
- Most complex implementation
- Requires feature flag infrastructure
- Longer development time

## Testing Backward Compatibility

### Test Cases

#### Test 1: Query Old Hash-Based doc_id

```python
async def test_query_old_doc_id():
    """Test querying document with hash-based doc_id."""
    old_doc_id = "doc-a1b2c3d4e5f6789012345678901234567"
    
    status = await storage.get_by_id(old_doc_id)
    
    assert status is not None
    assert status["doc_id"] == old_doc_id
    assert status["content_hash"] is not None
```

#### Test 2: Query New UUID-Based doc_id

```python
async def test_query_new_doc_id():
    """Test querying document with UUID-based doc_id."""
    new_doc_id = "doc-550e8400-e29b-41d4-a716-446655440000"
    
    status = await storage.get_by_id(new_doc_id)
    
    assert status is not None
    assert status["doc_id"] == new_doc_id
    assert status["content_hash"] is not None
```

#### Test 3: Mixed Format Batch Query

```python
async def test_mixed_format_batch_query():
    """Test querying multiple documents with mixed formats."""
    doc_ids = [
        "doc-a1b2c3d4e5f6789012345678901234567",  # Hash-based
        "doc-550e8400-e29b-41d4-a716-446655440000"  # UUID-based
    ]
    
    statuses = await storage.get_by_ids(doc_ids)
    
    assert len(statuses) == 2
    assert all(s["content_hash"] is not None for s in statuses)
```

#### Test 4: track_id Not in Response

```python
async def test_track_id_removed():
    """Test that track_id is not in upload response."""
    response = await upload_file(test_file)
    
    assert "doc_id" in response
    assert "track_id" not in response
```

#### Test 5: content_hash Nullable

```python
async def test_content_hash_nullable():
    """Test that content_hash is nullable during extraction."""
    response = await upload_file(test_file)
    doc_id = response["doc_id"]
    
    # Immediately after upload
    status = await storage.get_by_id(doc_id)
    assert status["status"] == "EXTRACTING"
    assert status["content_hash"] is None
    
    # After extraction
    await wait_for_extraction(doc_id)
    status = await storage.get_by_id(doc_id)
    assert status["status"] in ["EXTRACTED", "PROCESSED"]
    assert status["content_hash"] is not None
```

## Monitoring and Logging

### Metrics to Track

1. **Hash-based doc_id queries**: Count queries using old format
2. **UUID-based doc_id queries**: Count queries using new format
3. **Migration progress**: Percentage of documents migrated
4. **Deprecation warnings**: Count of deprecated feature usage

### Logging Examples

```python
# Log hash-based doc_id usage
if is_hash_doc_id(doc_id):
    logger.warning(
        f"Hash-based doc_id queried: {doc_id}. "
        "This format is deprecated and will be removed in version X.X+6."
    )

# Log migration progress
migrated_count = await count_uuid_doc_ids()
total_count = await count_all_doc_ids()
migration_percentage = (migrated_count / total_count) * 100

logger.info(
    f"Migration progress: {migration_percentage:.2f}% "
    f"({migrated_count}/{total_count} documents)"
)
```

## Rollback Procedures

### Rollback Scenario 1: Server Rollback

If issues are discovered after deployment:

1. **Stop new uploads**: Prevent new UUID-based documents
2. **Rollback server**: Deploy previous version
3. **Verify functionality**: Test with old system
4. **Investigate issues**: Debug problems
5. **Fix and redeploy**: Address issues and retry

**Note**: track_id system was removed, so full rollback requires restoring track_id functionality or using a compatibility layer.

### Rollback Scenario 2: Database Rollback

If database migration causes issues:

1. **Stop all services**: Prevent data corruption
2. **Restore backup**: Restore pre-migration database
3. **Verify data integrity**: Check document counts and content
4. **Rollback server**: Deploy previous version
5. **Resume operations**: Restart services

**Migration Script Rollback**:
```python
async def rollback_migration(storage):
    """Rollback database migration."""
    # Restore old doc_ids from content_hash
    docs = await storage.get_all_docs()
    
    for doc in docs:
        if is_uuid_doc_id(doc["doc_id"]) and doc["content_hash"]:
            # Restore hash-based doc_id
            old_doc_id = f"doc-{doc['content_hash']}"
            await storage.update_doc_id(doc["doc_id"], old_doc_id)
    
    # Remove content_hash column
    await storage.execute_sql(
        "ALTER TABLE LIGHTRAG_DOC_STATUS DROP COLUMN content_hash"
    )
    
    # Restore track_id column
    await storage.execute_sql(
        "ALTER TABLE LIGHTRAG_DOC_STATUS ADD COLUMN track_id VARCHAR(255)"
    )
```

## Support and Resources

### Getting Help

- **Documentation**: See [API Documentation](./APIDocumentation.md)
- **Migration Guide**: See [Migration Runbook](./MigrationRunbook.md)
- **Issues**: Report problems on GitHub issue tracker
- **Community**: Ask questions in community forums

### Common Issues

#### Issue 1: "track_id not found in response"

**Cause**: Client code expects track_id field that was removed

**Solution**: Update client code to use doc_id instead

```python
# Before
track_id = response["track_id"]  # KeyError

# After
doc_id = response["doc_id"]
```

#### Issue 2: "Cannot query document during extraction"

**Cause**: Expecting old behavior where doc_id not available until extraction

**Solution**: doc_id is now available immediately, no need to wait

```python
# Before (old system)
# Had to wait for extraction to get doc_id

# After (new system)
response = await upload_file(file)
doc_id = response["doc_id"]  # Available immediately!
status = await get_doc_status(doc_id)  # Works even during extraction
```

#### Issue 3: "content_hash is null"

**Cause**: Querying document before extraction completes

**Solution**: This is expected behavior, content_hash populated after extraction

```python
# Check status first
status = await get_doc_status(doc_id)

if status["status"] == "EXTRACTING":
    # content_hash not yet available
    assert status["content_hash"] is None
elif status["status"] in ["EXTRACTED", "PROCESSED"]:
    # content_hash should be available
    assert status["content_hash"] is not None
```

## Version Compatibility Matrix

| Client Version | Server Version | Compatibility | Notes |
|---------------|----------------|---------------|-------|
| Old (track_id) | Old (track_id) | ✅ Full | Legacy system |
| Old (track_id) | New (UUID) | ❌ Broken | track_id removed from server |
| New (UUID) | Old (track_id) | ⚠️ Partial | Can't use immediate doc_id |
| New (UUID) | New (UUID) | ✅ Full | Recommended configuration |

**Recommendation**: Update client and server together for best compatibility.
