#!/usr/bin/bash

game='pb'

echo "---"
echo "etray-110-mm"
echo "---"
./etray-n.py --game $game --test --depth 110

echo "---"
echo "PowerBall"
echo "---"
./second_to_last.py --game $game --test

exit 0

echo "---"
echo "Tray"
echo "---"
./tray.py --game $game --test
echo "---"
echo "Column 1"
echo "---"
./third-col2-70mm.py --col 1 --game $game --test
echo "---"
echo "Column 2"
echo "---"
./third-col2-70mm.py --col 2 --game $game --test
echo "---"
echo "Column 3"
echo "---"
./third-col2-70mm.py --col 3 --game $game --test
echo "---"
echo "Column 4"
echo "---"
./third-col2-70mm.py --col 4 --game $game --test
echo "---"
echo "Column 5"
echo "---"
./third-col2-70mm.py --col 5 --game $game --test
