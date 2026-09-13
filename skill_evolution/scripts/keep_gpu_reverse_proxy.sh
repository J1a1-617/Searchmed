#!/usr/bin/env bash
set -u

REMOTE_HOST="${GPU_REMOTE_HOST:-visitor@10.31.112.24}"
REMOTE_PORT="${GPU_PROXY_REMOTE_PORT:-17897}"
LOCAL_PROXY="${LOCAL_PROXY_ENDPOINT:-127.0.0.1:7897}"

while true; do
  sshpass -e ssh -N \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=2 \
    -o ConnectTimeout=15 \
    -o PreferredAuthentications=password \
    -o PubkeyAuthentication=no \
    -o StrictHostKeyChecking=no \
    -R "${REMOTE_PORT}:${LOCAL_PROXY}" \
    "${REMOTE_HOST}"
  sleep 5
done
