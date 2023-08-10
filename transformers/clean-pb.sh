#!/usr/bin/bash

# first backup
#./backup.sh

# then cleanup
rm -f batch-pb-*.log
rm -f probs_pb.py
rm -f sums_pb.pkl
rm -f models/second-to-last-pb.model
rm -f models/third-col1-70pb.model
rm -f models/third-col2-70pb.model
rm -f models/third-col3-70pb.model
rm -f models/third-col4-70pb.model
rm -f models/third-col5-70pb.model
rm -f models/tray-pb.model
rm -f models/etray-pb.model
rm -f models/etray-*-pb.model
