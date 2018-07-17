#!/bin/bash
. kamiak_config.sh

# Note both have trailing slashes
from="$remotessh:$remotedir"
to="$localdir"

# YOLO backup files and weights, SLURM output files
rsync -Pahuv --include="*/" \
    --include="logs/" --include="logs/*" \
    --include="models/" --include="models/*" \
    --exclude="*" "$from" "$to"
