@echo off
@REM Run the python script with the --no_show flag so the plot closes instantly

@REM python main.py -k 100000 -t 3500000 --policy Thompson-Sampling --no_show
@REM python main.py -k 100000 -t 3500000 --policy Epsilon-Greedy --no_show
@REM python main.py -k 100000 -t 3500000 --policy UCB --no_show

python main.py -k 100 -t 1000000 --policy Thompson-Sampling --no_show
python main.py -k 100 -t 1000000 --policy Epsilon-Greedy --no_show
python main.py -k 100 -t 1000000 --policy UCB --no_show

echo Run finished! Plot has been saved.