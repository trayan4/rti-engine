#!/bin/sh
set -e

# The dataset is generated, not shipped in the image, so it does not
# exist until a container actually starts. Idempotent: skip if a
# previous replica (or a previous start of this one) already built it.
if [ ! -f /app/data/generated/workforce.parquet ]; then
  make data
fi

exec "$@"
