#!/usr/bin/env bash

TF_CONFIG='{
    "cluster": {
        "chief": ["127.0.0.1:22220"],
        "worker": ["127.0.0.1:22221", "127.0.0.1:22222", "127.0.0.1:22223"],
        "ps": ["127.0.0.1:22224"]
    },
    "task": {"type": "chief", "index": 0}
}' CUDA_VISIBLE_DEVICES=0 nohup unbuffer python cnn_mnist_main.py > ./log_0 2>&1 &

TF_CONFIG='{
    "cluster": {
        "chief": ["127.0.0.1:22220"],
        "worker": ["127.0.0.1:22221", "127.0.0.1:22222", "127.0.0.1:22223"],
        "ps": ["127.0.0.1:22224"]
    },
    "task": {"type": "worker", "index": 0}
}' CUDA_VISIBLE_DEVICES=1 nohup unbuffer python cnn_mnist_main.py > ./log_1 2>&1 &

TF_CONFIG='{
    "cluster": {
        "chief": ["127.0.0.1:22220"],
        "worker": ["127.0.0.1:22221", "127.0.0.1:22222", "127.0.0.1:22223"],
        "ps": ["127.0.0.1:22224"]
    },
    "task": {"type": "worker", "index": 1}
}' CUDA_VISIBLE_DEVICES=2 nohup unbuffer python cnn_mnist_main.py > ./log_2 2>&1 &

TF_CONFIG='{
    "cluster": {
        "chief": ["127.0.0.1:22220"],
        "worker": ["127.0.0.1:22221", "127.0.0.1:22222", "127.0.0.1:22223"],
        "ps": ["127.0.0.1:22224"]
    },
    "task": {"type": "worker", "index": 2}
}' CUDA_VISIBLE_DEVICES=3 nohup unbuffer python cnn_mnist_main.py > ./log_3 2>&1 &

TF_CONFIG='{
    "cluster": {
        "chief": ["127.0.0.1:22220"],
        "worker": ["127.0.0.1:22221", "127.0.0.1:22222", "127.0.0.1:22223"],
        "ps": ["127.0.0.1:22224"]
    },
    "task": {"type": "ps", "index": 0}
}' CUDA_VISIBLE_DEVICES= nohup unbuffer python cnn_mnist_main.py > ./log_4 2>&1 &

TF_CONFIG='{
    "cluster": {
        "chief": ["127.0.0.1:22220"],
        "worker": ["127.0.0.1:22221", "127.0.0.1:22222", "127.0.0.1:22223"],
        "ps": ["127.0.0.1:22224"]
    },
    "task": {"type": "evaluator", "index": 0}
}' CUDA_VISIBLE_DEVICES=5 nohup unbuffer python cnn_mnist_main.py > ./log_5 2>&1 &