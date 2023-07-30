#!/usr/bin/bash
#
# First train
#
#./third-col2-70mm.py --col 1
#./third-col2-70mm.py --col 2
#./third-col2-70mm.py --col 3
./third-col2-70mm.py --col 4
./third-col2-70mm.py --col 5
#
# Then test
#
./third-col2-70mm.py --col 1 --test --skip '[0,69,68,67,66]'
./third-col2-70mm.py --col 2 --test --skip '[0,69,68,67]'
./third-col2-70mm.py --col 3 --test --skip '[0,69,68]'
./third-col2-70mm.py --col 4 --test --skip '[0,69]'
./third-col2-70mm.py --col 5 --test --skip '[0]'
