#!/usr/bin/bash

game="mm"

#
# First train
#
for i in 1 2 3 4; do
    for j in 2 3 4 5; do
	echo "Processing: $i $j"
	./third-col2-70mm.py --col $j --game $game
    done
done    
# pb
for i in 1 2 3 4; do
    ./second_to_last.py --game $game
done
# tray
for i in 1 2 3 4 5 6 7 8; do
    ./tray.py --game $game
done

#
# Then test
#
./third-col2-70mm.py --col 1 --game $game --test --skip '[0,69,68,67,66]'
./third-col2-70mm.py --col 2 --game $game --test --skip '[0,69,68,67]'
./third-col2-70mm.py --col 3 --game $game --test --skip '[0,69,68]'
./third-col2-70mm.py --col 4 --game $game --test --skip '[0,69]'
./third-col2-70mm.py --col 5 --game $game --test --skip '[0]'
# pb
./second_to_last.py --game $game --test
# tray
./tray.py --game $game --test
