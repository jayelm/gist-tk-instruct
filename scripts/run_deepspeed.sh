#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=cache/

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port $port src/run_s2s.py $1
