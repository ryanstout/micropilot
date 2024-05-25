#!/bin/bash

# IP=10.200.31.143
IP=192.168.1.212

rsync -av -e ssh --exclude=__pycache__ --exclude=data --exclude=checkpoints --exclude=wandb --exclude=lightning_logs * admin@$IP:/home/admin/micropilot2/