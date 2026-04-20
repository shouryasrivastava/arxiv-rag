#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# GCP deployment script for the ArXiv Research Assistant
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated: https://cloud.google.com/sdk
#   2. A GCP project with billing enabled
#   3. Run: gcloud auth login && gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   chmod +x scripts/gcp-deploy.sh
#   ./scripts/gcp-deploy.sh [PROJECT_ID] [ZONE]
#
# Defaults:
#   ZONE = us-east1-b   (adjust to your nearest region)
#   MACHINE = n2-standard-4  (4 vCPU, 16 GB RAM — enough for llama3.2:3b on CPU)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ID="${1:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${2:-us-east1-b}"
INSTANCE_NAME="arxiv-rag-demo"
MACHINE_TYPE="n2-standard-4"
DISK_SIZE="50GB"
REPO_URL="https://github.com/shouryasrivastava/arxiv-rag.git"

echo "Deploying to GCP project: ${PROJECT_ID}, zone: ${ZONE}"

# ── 1. Enable required APIs ───────────────────────────────────────────────────
gcloud services enable compute.googleapis.com --project="${PROJECT_ID}"

# ── 2. Create VM ──────────────────────────────────────────────────────────────
gcloud compute instances create "${INSTANCE_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size="${DISK_SIZE}" \
  --boot-disk-type=pd-ssd \
  --tags=http-server,https-server,arxiv-rag \
  --metadata=startup-script='#!/bin/bash
    set -e
    # Install Docker
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    # Install Docker Compose plugin
    apt-get install -y docker-compose-plugin
    # Clone repo
    git clone '"${REPO_URL}"' /opt/arxiv-rag
    cd /opt/arxiv-rag
    # Start services
    docker compose up -d --build
    echo "Startup complete." >> /var/log/arxiv-rag-startup.log
  '

# ── 3. Open firewall for Streamlit (port 8501) ────────────────────────────────
gcloud compute firewall-rules create allow-arxiv-rag \
  --project="${PROJECT_ID}" \
  --allow=tcp:8501 \
  --target-tags=arxiv-rag \
  --description="Allow ArXiv RAG Streamlit UI" 2>/dev/null || \
  echo "Firewall rule already exists."

# ── 4. Get external IP ────────────────────────────────────────────────────────
sleep 10
EXTERNAL_IP=$(gcloud compute instances describe "${INSTANCE_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  VM created: ${INSTANCE_NAME}"
echo "  External IP: ${EXTERNAL_IP}"
echo ""
echo "  The startup script is installing Docker and cloning"
echo "  the repo. First boot takes ~10 minutes."
echo ""
echo "  Once ready, open:"
echo "  http://${EXTERNAL_IP}:8501"
echo ""
echo "  Monitor startup:"
echo "  gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} \\"
echo "    --command='tail -f /var/log/syslog | grep arxiv'"
echo "═══════════════════════════════════════════════════════"
