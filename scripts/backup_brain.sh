#!/bin/bash
# Daily backup of brain project critical data
# Backs up: model weights, knowledge DB, configs, training scripts
# Keeps last 7 daily backups (rolling)

set -euo pipefail

BACKUP_DIR="/opt/brain/backups"
DATE=$(date +%Y-%m-%d)
BACKUP_PATH="${BACKUP_DIR}/${DATE}"

mkdir -p "${BACKUP_PATH}"

echo "[$(date)] Starting brain backup to ${BACKUP_PATH}"

# 1. Model weights (the most critical — these take hours to train)
echo "  Backing up model weights..."
for dir in v4_mlp v5_mlp v3f_anneal world_model self_model text_grounding causal_graph concepts v4_mlp_online; do
    src="/opt/brain/outputs/cortex/${dir}"
    if [ -d "$src" ]; then
        mkdir -p "${BACKUP_PATH}/models/${dir}"
        cp -a "$src"/* "${BACKUP_PATH}/models/${dir}/" 2>/dev/null || true
    fi
done

# 2. Knowledge database
echo "  Backing up knowledge.db..."
mkdir -p "${BACKUP_PATH}/db"
# Use sqlite3 backup for consistency (avoids copying while writes are happening)
if command -v sqlite3 &>/dev/null; then
    sqlite3 /opt/brain/outputs/cortex/knowledge.db ".backup '${BACKUP_PATH}/db/knowledge.db'"
else
    cp /opt/brain/outputs/cortex/knowledge.db "${BACKUP_PATH}/db/"
fi

# 3. Training scripts
echo "  Backing up scripts..."
mkdir -p "${BACKUP_PATH}/scripts"
cp /opt/brain/scripts/*.py "${BACKUP_PATH}/scripts/" 2>/dev/null || true
cp /opt/brain/scripts/*.sh "${BACKUP_PATH}/scripts/" 2>/dev/null || true

# 4. Embedding cache metadata (not the full embeddings — too large)
echo "  Backing up embedding cache metadata..."
mkdir -p "${BACKUP_PATH}/embed_meta"
for f in whitening.safetensors labels.safetensors; do
    src="/opt/brain/data/vggsound/.embed_cache/${f}"
    [ -f "$src" ] && cp "$src" "${BACKUP_PATH}/embed_meta/"
done

# 5. Templates
echo "  Backing up templates..."
cp -a /opt/brain/templates "${BACKUP_PATH}/"

# 6. Rust source (just the brain crates, not target/)
echo "  Backing up Rust source..."
mkdir -p "${BACKUP_PATH}/rust"
for crate in brain-core brain-experiment brain-db brain-server; do
    cp -a "/opt/brain/rust/crates/${crate}/src" "${BACKUP_PATH}/rust/${crate}-src" 2>/dev/null || true
done
cp /opt/brain/rust/Cargo.toml "${BACKUP_PATH}/rust/" 2>/dev/null || true

# 7. Compute backup size
SIZE=$(du -sh "${BACKUP_PATH}" | cut -f1)
echo "  Backup size: ${SIZE}"

# 8. Clean old backups (keep last 7)
echo "  Cleaning old backups..."
ls -dt "${BACKUP_DIR}"/20* 2>/dev/null | tail -n +8 | while read old; do
    echo "    Removing old backup: ${old}"
    rm -rf "$old"
done

echo "[$(date)] Backup complete: ${BACKUP_PATH} (${SIZE})"
