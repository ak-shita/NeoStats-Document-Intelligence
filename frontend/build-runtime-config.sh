#!/usr/bin/env sh
set -eu

# Render Static Sites expose configured build-time environment variables here.
# API_BASE_URL is public (not a credential) and must be the Railway backend URL.
if [ -n "${API_BASE_URL:-}" ]; then
  case "$API_BASE_URL" in
    https://*|http://*) ;;
    *) echo "API_BASE_URL must start with http:// or https://" >&2; exit 1 ;;
  esac
  escaped_url=$(printf '%s' "$API_BASE_URL" | sed 's/\\/\\\\/g; s/"/\\"/g; s:/*$::')
  printf 'window.NEOSTATS_API_BASE = "%s";\n' "$escaped_url" > js/runtime-config.js
fi
