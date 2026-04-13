#!/usr/bin/env python3
"""
Migration script to update document statuses from legacy to new statuses.

Legacy → New mapping:
- PENDING → EXTRACTED (full text extraction done, waiting for chunking)
- PROCESSING → CHUNKING (currently chunking and extracting entities)
- PREPROCESSED → CHUNKED (chunking done, waiting for multimodal processing)
- PROCESSED → PROCESSED (no change)
- FAILED → FAILED (no change)

Usage:
    python scripts/migrate_document_statuses.py [--dry-run] [--workspace WORKSPACE_ID]
"""

import asyncio
import argparse
import sys
import datetime
from datetime import timezone
from pathlib import Path

# Add parent directory to path to import lightrag
sys.path.insert(0, str(Path(__file__).parent.parent))

from lightrag.base import DocStatus
from lightrag.api.config import initialize_config
from lightrag.api.lightrag_factory import LLMConfigCache, build_rag_instance


# Status migration mapping
STATUS_MIGRATION = {
    "pending": "extracted",
    "processing": "chunking",
    "preprocessed": "chunked",
    # No change for these
    "processed": "processed",
    "failed": "failed",
}


async def _get_docs_raw_from_db(rag, workspace_id):
    """Query database directly to get raw document status values, bypassing DocProcessingStatus.__post_init__ auto-migration.
    
    Supports multiple storage backends: PostgreSQL, MongoDB, Redis, JSON, OpenSearch.
    
    Args:
        rag: LightRAG instance with initialized doc_status storage
        workspace_id: Workspace ID to query (empty string for default workspace)
    
    Returns:
        list: List of tuples (doc_id, status, file_path) with raw status strings from DB
        
    Raises:
        NotImplementedError: If storage type is not supported
    """
    # Check storage type and query accordingly
    storage_type = type(rag.doc_status).__name__
    
    if "PGDocStatusStorage" in storage_type or "Postgres" in storage_type:
        # PostgreSQL
        sql = "SELECT id, status, file_path FROM LIGHTRAG_DOC_STATUS WHERE workspace=$1"
        result = await rag.doc_status.db.query(sql, [workspace_id], True)
        return [(row["id"], row["status"], row.get("file_path")) for row in result]
    
    elif "MongoDocStatusStorage" in storage_type or "Mongo" in storage_type:
        # MongoDB
        collection = rag.doc_status.collection
        cursor = collection.find({"workspace": workspace_id}, {"_id": 1, "status": 1, "file_path": 1})
        return [(doc["_id"], doc["status"], doc.get("file_path")) async for doc in cursor]
    
    elif "RedisDocStatusStorage" in storage_type or "Redis" in storage_type:
        # Redis
        pattern = f"{rag.doc_status.namespace}:doc_status:{workspace_id}:*"
        keys = await rag.doc_status.redis.keys(pattern)
        docs = []
        for key in keys:
            data = await rag.doc_status.redis.hgetall(key)
            if data and "status" in data:
                doc_id = key.decode().split(":")[-1]
                status = data["status"].decode() if isinstance(data["status"], bytes) else data["status"]
                file_path = data.get("file_path")
                if isinstance(file_path, bytes):
                    file_path = file_path.decode()
                docs.append((doc_id, status, file_path))
        return docs
    
    elif "JSONDocStatusStorage" in storage_type or "JSON" in storage_type:
        # JSON file storage
        if not hasattr(rag.doc_status, 'data') or workspace_id not in rag.doc_status.data:
            return []
        workspace_data = rag.doc_status.data[workspace_id]
        return [(doc_id, doc_data["status"], doc_data.get("file_path")) 
                for doc_id, doc_data in workspace_data.items()]
    
    elif "OpenSearchDocStatusStorage" in storage_type or "OpenSearch" in storage_type:
        # OpenSearch
        query = {
            "query": {
                "term": {"workspace": workspace_id}
            },
            "_source": ["status", "file_path"],
            "size": 10000
        }
        result = await rag.doc_status.client.search(
            index=rag.doc_status.index_name,
            body=query
        )
        return [(hit["_id"], hit["_source"]["status"], hit["_source"].get("file_path")) 
                for hit in result["hits"]["hits"]]
    
    else:
        raise NotImplementedError(f"Storage type {storage_type} not supported for direct DB access")


async def migrate_workspace_statuses(rag, dry_run=False):
    """Migrate document statuses for a single workspace.
    
    Args:
        rag: LightRAG instance for the workspace
        dry_run: If True, only show what would be changed without making changes
        
    Returns:
        dict: Statistics about the migration
    """
    workspace_id = rag.workspace or ""
    print(f"\n{'='*60}")
    print(f"Migrating workspace: {workspace_id or 'default'}")
    print(f"{'='*60}")
    
    stats = {
        "total": 0,
        "migrated": 0,
        "unchanged": 0,
        "errors": 0,
        "by_status": {}
    }
    
    # Query DB directly to get raw status values (bypass __post_init__ auto-migration)
    try:
        all_docs_raw = await _get_docs_raw_from_db(rag, workspace_id)
    except Exception as e:
        print(f"✗ Error querying database: {e}")
        import traceback
        traceback.print_exc()
        return stats
    
    if not all_docs_raw:
        print("\n✓ No documents found in this workspace")
        return stats
    
    # Separate documents by status
    legacy_docs = {}
    unchanged_count = 0
    
    for doc_id, raw_status, file_path in all_docs_raw:
        stats["total"] += 1
        
        # Check if this is a legacy status that needs migration
        if raw_status in STATUS_MIGRATION and STATUS_MIGRATION[raw_status] != raw_status:
            legacy_docs[doc_id] = (raw_status, file_path)
        else:
            unchanged_count += 1
            if raw_status not in stats["by_status"]:
                stats["by_status"][raw_status] = 0
            stats["by_status"][raw_status] += 1
    
    stats["unchanged"] = unchanged_count
    
    if not legacy_docs:
        print(f"\n✓ No documents need migration (Total: {stats['total']}, Unchanged: {unchanged_count})")
        return stats
    
    print(f"\nTotal documents: {stats['total']}")
    print(f"Documents to migrate: {len(legacy_docs)}")
    print(f"Documents unchanged: {unchanged_count}")
    
    if dry_run:
        print("\n[DRY RUN] Would migrate the following documents:")
    else:
        print("\nMigrating documents...")
    
    # Prepare updates
    updates = {}
    
    for doc_id, (old_status, file_path) in legacy_docs.items():
        new_status = STATUS_MIGRATION[old_status]
        
        updates[doc_id] = {
            "status": new_status,
        }
        
        # Track statistics
        if old_status not in stats["by_status"]:
            stats["by_status"][old_status] = 0
        stats["by_status"][old_status] += 1
        
        # Print migration info
        file_path_display = file_path or "unknown"
        print(f"  {doc_id[:20]}... | {file_path_display[:30]:30} | {old_status:12} → {new_status}")
    
    # Apply updates
    if not dry_run:
        try:
            # For PostgreSQL, we can update directly with SQL
            storage_type = type(rag.doc_status).__name__
            
            if "PGDocStatusStorage" in storage_type or "Postgres" in storage_type:
                # Direct SQL update for PostgreSQL
                for doc_id, (old_status, _) in legacy_docs.items():
                    new_status = STATUS_MIGRATION[old_status]
                    sql = "UPDATE LIGHTRAG_DOC_STATUS SET status=$1, updated_at=$2 WHERE workspace=$3 AND id=$4"
                    await rag.doc_status.db.execute(sql, {
                        "status": new_status,
                        "updated_at": datetime.datetime.now(timezone.utc).replace(tzinfo=None),
                        "workspace": workspace_id,
                        "id": doc_id
                    })
            else:
                # For other storage types, fetch full document and update
                for doc_id in legacy_docs.keys():
                    doc = await rag.doc_status.get(doc_id)
                    if doc:
                        old_status = doc.status.value if hasattr(doc.status, 'value') else str(doc.status)
                        new_status = STATUS_MIGRATION.get(old_status, old_status)
                        if old_status != new_status:
                            await rag.doc_status.upsert({
                                doc_id: {
                                    "content_summary": doc.content_summary,
                                    "content_length": doc.content_length,
                                    "status": new_status,
                                    "file_path": doc.file_path,
                                    "chunks_count": doc.chunks_count,
                                    "chunks_list": doc.chunks_list,
                                    "track_id": doc.track_id,
                                    "metadata": doc.metadata,
                                    "error_msg": doc.error_msg,
                                    "created_at": doc.created_at,
                                    "updated_at": doc.updated_at,
                                }
                            })
            
            stats["migrated"] = len(updates)
            print(f"\n✓ Successfully migrated {len(updates)} documents")
        except Exception as e:
            print(f"\n✗ Error during migration: {e}")
            import traceback
            traceback.print_exc()
            stats["errors"] = len(updates)
            return stats
    else:
        stats["migrated"] = len(updates)
        print(f"\n[DRY RUN] Would migrate {len(updates)} documents")
    
    return stats


async def migrate_all_workspaces(global_args, dry_run=False, target_workspace=None):
    """Migrate document statuses across all workspaces.
    
    Args:
        global_args: Global configuration
        dry_run: If True, only show what would be changed
        target_workspace: If specified, only migrate this workspace
        
    Returns:
        dict: Overall statistics
    """
    print("="*60)
    print("Document Status Migration Tool")
    print("="*60)
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE MIGRATION'}")
    if target_workspace:
        print(f"Target: Workspace '{target_workspace}'")
    else:
        print("Target: All workspaces")
    print("="*60)
    
    overall_stats = {
        "workspaces": 0,
        "total_docs": 0,
        "total_migrated": 0,
        "total_unchanged": 0,
        "total_errors": 0,
        "by_workspace": {}
    }
    
    # Initialize config cache
    config_cache = LLMConfigCache(global_args)
    
    # Get list of workspaces
    if target_workspace:
        workspaces = [target_workspace]
    else:
        # Build a temporary RAG instance to get workspace list
        temp_rag = build_rag_instance("", global_args, config_cache)
        await temp_rag.initialize_storages()
        
        try:
            workspace_stats = await temp_rag.doc_status.get_status_counts_across_workspaces()
            workspaces = list(workspace_stats.keys())
            print(f"\nFound {len(workspaces)} workspaces: {', '.join(workspaces)}")
        except Exception as e:
            print(f"Error getting workspace list: {e}")
            print("Falling back to default workspace only")
            workspaces = [""]
    
    # Migrate each workspace
    for workspace_id in workspaces:
        try:
            # Build RAG instance for this workspace
            rag = build_rag_instance(workspace_id, global_args, config_cache)
            await rag.initialize_storages()
            
            # Migrate this workspace
            stats = await migrate_workspace_statuses(rag, dry_run)
            
            # Update overall statistics
            overall_stats["workspaces"] += 1
            overall_stats["total_docs"] += stats["total"]
            overall_stats["total_migrated"] += stats["migrated"]
            overall_stats["total_unchanged"] += stats["unchanged"]
            overall_stats["total_errors"] += stats["errors"]
            overall_stats["by_workspace"][workspace_id or "default"] = stats
            
        except Exception as e:
            print(f"\n✗ Error migrating workspace '{workspace_id}': {e}")
            overall_stats["total_errors"] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("Migration Summary")
    print("="*60)
    print(f"Workspaces processed: {overall_stats['workspaces']}")
    print(f"Total documents: {overall_stats['total_docs']}")
    print(f"Migrated: {overall_stats['total_migrated']}")
    print(f"Unchanged: {overall_stats['total_unchanged']}")
    print(f"Errors: {overall_stats['total_errors']}")
    
    if overall_stats["by_workspace"]:
        print("\nBy workspace:")
        for ws_id, ws_stats in overall_stats["by_workspace"].items():
            print(f"  {ws_id:30} | Total: {ws_stats['total']:4} | Migrated: {ws_stats['migrated']:4}")
            if ws_stats.get("by_status"):
                for status, count in ws_stats["by_status"].items():
                    new_status = STATUS_MIGRATION.get(status, status)
                    if status != new_status:
                        print(f"    - {status:12} → {new_status:12} : {count:4} docs")
    
    print("="*60)
    
    if dry_run:
        print("\n[DRY RUN] No changes were made. Run without --dry-run to apply changes.")
    else:
        print("\n✓ Migration complete!")
    
    return overall_stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate document statuses from legacy to new statuses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview changes without applying)
  python scripts/migrate_document_statuses.py --dry-run
  
  # Migrate all workspaces
  python scripts/migrate_document_statuses.py
  
  # Migrate specific workspace
  python scripts/migrate_document_statuses.py --workspace my-workspace
  
  # Dry run for specific workspace
  python scripts/migrate_document_statuses.py --dry-run --workspace my-workspace

Status Migration:
  pending      → extracted   (full text extraction done)
  processing   → chunking    (currently chunking)
  preprocessed → chunked     (chunking done)
  processed    → processed   (no change)
  failed       → failed      (no change)
        """
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them"
    )
    
    parser.add_argument(
        "--workspace",
        type=str,
        help="Migrate only this workspace (default: all workspaces)"
    )
    
    args = parser.parse_args()
    
    # Initialize global configuration
    print("Initializing configuration...")
    global_args = initialize_config()
    
    # Run migration
    try:
        asyncio.run(migrate_all_workspaces(
            global_args,
            dry_run=args.dry_run,
            target_workspace=args.workspace
        ))
    except KeyboardInterrupt:
        print("\n\nMigration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
