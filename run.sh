# Запуск через uv:
# uv run ai-fashion-trends ingest --source rss

set -euo pipefail
cd "$(dirname "$0")"

exec uv run ai-fashion-trends "$@"
