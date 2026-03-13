#!/usr/bin/env bash
# Build Jekyll docs into public/docs/ so Vite dev server serves them at /MER-Factory/docs/
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GUIDE_DIR="$SCRIPT_DIR/guide"
OUTPUT_DIR="$SCRIPT_DIR/public/docs"

# Use rbenv Ruby if available (system Ruby may be too old)
if command -v rbenv &> /dev/null; then
  eval "$(rbenv init -)"
fi

echo "Building Jekyll docs..."
cd "$GUIDE_DIR"

if ! command -v bundle &> /dev/null; then
  echo "Error: Ruby Bundler is not installed. Install Ruby >= 3.0 and run 'gem install bundler' first."
  exit 1
fi

bundle install --quiet
# Fix SSL cert path for rbenv Ruby; set repo for github-metadata plugin
export SSL_CERT_FILE="/etc/ssl/cert.pem"
PAGES_REPO_NWO="Lum1104/MER-Factory" \
  bundle exec jekyll build --baseurl "/MER-Factory/docs" --destination "$OUTPUT_DIR"

echo "Jekyll docs built to $OUTPUT_DIR"
echo "Run 'npm run dev' to preview the full site."
