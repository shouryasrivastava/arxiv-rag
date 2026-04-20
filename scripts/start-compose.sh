#!/usr/bin/env bash
# Wrapper so the preview server can launch docker compose with the correct socket path.
export DOCKER_HOST="unix:///Users/shouryasrivastava/.docker/run/docker.sock"
cd "$(dirname "$0")/.." || exit 1
/Applications/Docker.app/Contents/Resources/bin/docker compose up --build
