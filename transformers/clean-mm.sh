#!/usr/bin/bash

# first backup
./backup.sh

# then cleanup
rm -f batch-mm-*.log
rm -f probs_mm.py
rm -f sums_mm.pkl
rm -f models/second-to-last-mm.model
rm -f models/third-col1-70mm.model
rm -f models/third-col2-70mm.model
rm -f models/third-col3-70mm.model
rm -f models/third-col4-70mm.model
rm -f models/third-col5-70mm.model
rm -f models/tray-mm.model
