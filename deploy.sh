#!/bin/bash

rsync -av -e ssh --exclude=__pycache__ --exclude=data --exclude=checkpoints --exclude=wandb --exclude=lightning_logs * admin@192.168.1.212:/home/admin/micropilot2/