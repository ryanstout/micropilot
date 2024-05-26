#!/bin/bash

IP=raspberrypi

rsync -av -e ssh --exclude=__pycache__ --exclude=data --exclude=checkpoints --exclude=wandb --exclude=lightning_logs * admin@$IP:/home/admin/micropilot2/