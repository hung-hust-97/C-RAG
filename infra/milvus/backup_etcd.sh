#!/usr/bin/env bash
# ==============================================================================
# etcd Snapshot Backup Script for Milvus Cluster
# - Runs daily snapshot from etcd cluster (crag-etcd1 / crag-etcd2 / crag-etcd3)
# - Saves snapshot in a date-based directory (YYYY-MM-DD)
# - Compresses with gzip to minimize disk usage (~2MB per snapshot)
# - Automatically deletes backup folders older than RETENTION_DAYS (7 days)
# ==============================================================================

set -euo pipefail

# Configuration
BACKUP_ROOT_DIR="${BACKUP_ROOT_DIR:-/home/qxphuc/C-RAG/data/backups/etcd}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
ETCD_CONTAINERS=("crag-etcd1" "crag-etcd2" "crag-etcd3")
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
DATE_FOLDER="$(date +"%Y-%m-%d")"
TARGET_DIR="${BACKUP_ROOT_DIR}/${DATE_FOLDER}"
LOG_FILE="${BACKUP_ROOT_DIR}/backup.log"

# Ensure backup directory exists
mkdir -p "${TARGET_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"

log() {
    local level="$1"
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${level}] $*" | tee -a "${LOG_FILE}"
}

log "INFO" "=================== Starting etcd snapshot backup ==================="
log "INFO" "Target Directory: ${TARGET_DIR}"
log "INFO" "Retention Policy: ${RETENTION_DAYS} days"

# Find a healthy etcd container
ACTIVE_CONTAINER=""
for container in "${ETCD_CONTAINERS[@]}"; do
    if docker ps --format '{{.Names}}' | grep -Eq "^${container}\$"; then
        if docker exec "${container}" etcdctl --endpoints=http://127.0.0.1:2379 endpoint health >/dev/null 2>&1; then
            ACTIVE_CONTAINER="${container}"
            log "INFO" "Found healthy etcd node: ${container}"
            break
        fi
    fi
done

if [ -z "${ACTIVE_CONTAINER}" ]; then
    log "ERROR" "No healthy etcd container found among: ${ETCD_CONTAINERS[*]}"
    exit 1
fi

TEMP_CONTAINER_PATH="/tmp/etcd_snap_${TIMESTAMP}.db"
LOCAL_RAW_FILE="${TARGET_DIR}/etcd_snapshot_${TIMESTAMP}.db"
LOCAL_GZ_FILE="${LOCAL_RAW_FILE}.gz"

# 1. Take snapshot inside etcd container
log "INFO" "Saving snapshot on container [${ACTIVE_CONTAINER}] to ${TEMP_CONTAINER_PATH}..."
if ! docker exec "${ACTIVE_CONTAINER}" etcdctl --endpoints=http://127.0.0.1:2379 snapshot save "${TEMP_CONTAINER_PATH}"; then
    log "ERROR" "Failed to save etcd snapshot on container ${ACTIVE_CONTAINER}"
    exit 1
fi

# 2. Copy snapshot from container to host
log "INFO" "Copying snapshot from container to host: ${LOCAL_RAW_FILE}..."
docker cp "${ACTIVE_CONTAINER}:${TEMP_CONTAINER_PATH}" "${LOCAL_RAW_FILE}"

# Clean up temp file in container
docker exec "${ACTIVE_CONTAINER}" rm -f "${TEMP_CONTAINER_PATH}" || true

# 3. Verify snapshot file integrity
if [ ! -s "${LOCAL_RAW_FILE}" ]; then
    log "ERROR" "Snapshot file is missing or empty: ${LOCAL_RAW_FILE}"
    rm -f "${LOCAL_RAW_FILE}"
    exit 1
fi

RAW_SIZE="$(du -h "${LOCAL_RAW_FILE}" | cut -f1)"
log "INFO" "Snapshot verified successfully (Raw size: ${RAW_SIZE})"

# 4. Compress snapshot with gzip to save disk space
log "INFO" "Compressing snapshot with gzip..."
gzip -f "${LOCAL_RAW_FILE}"
GZ_SIZE="$(du -h "${LOCAL_GZ_FILE}" | cut -f1)"
log "INFO" "Compressed snapshot: ${LOCAL_GZ_FILE} (Size: ${GZ_SIZE})"

# 5. Clean up old backup folders older than RETENTION_DAYS
log "INFO" "Cleaning up backup folders older than ${RETENTION_DAYS} days in ${BACKUP_ROOT_DIR}..."
# Delete directories older than RETENTION_DAYS
find "${BACKUP_ROOT_DIR}" -mindepth 1 -maxdepth 1 -type d -mtime "+${RETENTION_DAYS}" -exec rm -rf {} + 2>/dev/null || true

# Also delete by date calculation for safety against mtime drift
CUTOFF_DATE="$(date -d "${RETENTION_DAYS} days ago" +"%Y-%m-%d" 2>/dev/null || date -v-"${RETENTION_DAYS}"d +"%Y-%m-%d" 2>/dev/null || echo "")"
if [ -n "${CUTOFF_DATE}" ]; then
    for dir in "${BACKUP_ROOT_DIR}"/20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]; do
        if [ -d "${dir}" ]; then
            DIR_NAME="$(basename "${dir}")"
            if [[ "${DIR_NAME}" < "${CUTOFF_DATE}" ]]; then
                log "INFO" "Removing expired backup folder: ${dir}"
                rm -rf "${dir}"
            fi
        fi
    done
fi

log "INFO" "=================== Backup completed successfully ==================="
