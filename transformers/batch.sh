#!/usr/bin/bash

# housekeeping - make sure directory 'models' exists
# Get the directory name from the user

directory_name="models"
# Check if the directory exists
if [[ ! -d "$directory_name" ]]; then
  # The directory doesn't exist, so create it
  echo "Creating directory '$directory_name'..."
  mkdir "$directory_name"
else
  # The directory already exists, so do nothing
  echo "Directory '$directory_name' already exists."
fi

# Check if the first argument is present
if [ -z "$1" ]; then
  echo "Error: The first argument must be either pb or mm."
  exit 1
fi
# Check if the first argument is either 'pb' or 'mm'
if [[ "$1" != "pb" && "$1" != "mm" ]]; then
  echo "Error: The first argument must be either 'pb' or 'mm'."
  exit 1
fi
game=$1
echo "Processing game $game"

#
# First train our various networks
#

# tray (must be done first)
for i in 1 2 3 4 5 6 7 8; do
    ./tray.py --game $game
done

# powerball
for i in 1 2 3 4; do
    ./second_to_last.py --game $game
done

# build each column
for i in 1 2 3 4; do
    for j in 1 2 3 4 5; do
	echo "Processing: $i $j"
	./third-col2-70mm.py --col $j --game $game
    done
done    

#
# Then test each column
#
#./third-col2-70mm.py --col 1 --game $game --test --skip '[0,69,68,67,66]'
#./third-col2-70mm.py --col 2 --game $game --test --skip '[0,69,68,67]'
#./third-col2-70mm.py --col 3 --game $game --test --skip '[0,69,68]'
#./third-col2-70mm.py --col 4 --game $game --test --skip '[0,69]'
#./third-col2-70mm.py --col 5 --game $game --test --skip '[0]'

#
# generate reports
#
# Get the current date and time
current_time=$(date "+%Y%m%d-%H%M%S")
# create timestamped logs
./batch-$game.sh > batch-$game-$current_time.log
python3 sums.py --game $game >> batch-$game-$current_time.log
