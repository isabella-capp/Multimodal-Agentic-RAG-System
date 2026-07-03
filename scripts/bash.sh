#!/bin/bash
srun -Q --immediate=10 -w ailb-login-03 --partition=all_serial --account=cvcs2026 --gres=gpu:1 --time 60:00 --pty bash
