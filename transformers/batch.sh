#!/usr/bin/bash
#
# First train
#
#./third-col2-70mm.py --col 1 --game mm
#./third-col2-70mm.py --col 2 --game mm
#./third-col2-70mm.py --col 3 --game mm
#./third-col2-70mm.py --col 4 --game mm
#./third-col2-70mm.py --col 5 --game mm
# pb
#./second_to_last.py --game mm
#
# Then test
#
./third-col2-70mm.py --col 1 --game mm --test --skip '[0,69,68,67,66]'
./third-col2-70mm.py --col 2 --game mm --test --skip '[0,69,68,67]'
./third-col2-70mm.py --col 3 --game mm --test --skip '[0,69,68]'
./third-col2-70mm.py --col 4 --game mm --test --skip '[0,69]'
./third-col2-70mm.py --col 5 --game mm --test --skip '[0]'
# pb
./second_to_last.py --game mm --test
