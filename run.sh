#!/bin/bash

echo "Starting the MAB simulation automated runner..."

# Run the python script with the --no_show flag so the plot closes instantly
# You can easily change parameters here without touching main.py!
python main.py -k 100000 -t 3500000 --policy Thompson-Sampling --no_show
python main.py -k 100000 -t 3500000 --policy Epsilon-Greedy --no_show
python main.py -k 100000 -t 3500000 --policy UCB --no_show

echo "Run finished! The terminal is now free."