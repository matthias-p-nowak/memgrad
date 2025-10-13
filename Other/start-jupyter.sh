#!/bin/bash

if [ -z "$TMUX" ] 
then
  echo "starting tmux session"
  exec tmux new-session -s Jupyter $0
fi
echo "continuing"
source ~/prj/.myenv/bin/activate
echo $VIRTUAL_ENV
sleep 5
jupyter notebook --ip=0.0.0.0 --port=2345 --no-browser
echo "finally..."
sleep 10
echo "exiting"