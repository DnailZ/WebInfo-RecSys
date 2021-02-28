#!/bin/bash
docker run --gpus all -e NVIDIA_VISIBLE_DEVICES=all --rm -it -v $(pwd):$(pwd) -w $(pwd) nvcr.io/nvidia/pytorch:20.06-py3 $@
