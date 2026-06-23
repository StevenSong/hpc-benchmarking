# 2026-06-23 UPDATE

For testing the new ConnectX 7 on the new `nvl03/4` nodes:
* git clone this repo to both nodes
* run the following on `nvl03` to set it up as the head node (this starts a docker container, leave this running):
    ```
    bash run_cluster.sh vllm/vllm-openai:v0.10.0 \
    10.32.15.23 \
    --head \
    /opt/gpudata/models \
    -e VLLM_HOST_IP=10.32.15.23 \
    -e GLOO_SOCKET_IFNAME=ens4013np0 \
    --ulimit nofile=1048576:1048576 \
    --privileged \
    -e NCCL_IB_HCA=mlx5 \
    -v /dev/shm:/dev/shm \
    -e NCCL_DEBUG=INFO \
    -e NCCL_SOCKET_IFNAME=ens4013np0 \
    -e NCCL_IB_GID_INDEX=3
    ```
* run the following on `nvl04` to set it up as a worker node (this starts a docker container, leave this running):
    ```
    bash run_cluster.sh vllm/vllm-openai:v0.10.0 \
    10.32.15.23 \
    --worker \
    /opt/gpudata/models \
    -e VLLM_HOST_IP=10.32.15.24 \
    -e GLOO_SOCKET_IFNAME=ens4013np0 \
    --ulimit nofile=1048576:1048576 \
    -v /dev/shm:/dev/shm \
    --privileged \
    -e NCCL_IB_HCA=mlx5 \
    -e NCCL_DEBUG=INFO \
    -e NCCL_SOCKET_IFNAME=ens4013np0 \
    -e NCCL_IB_GID_INDEX=3
    ```
* run the following on `nvl03` to start the `vllm` server (this starts a new shell in the head node container and starts vllm, leave this running):
    ```
    docker exec -it $(docker ps --format '{{.Names}}' | head -1) /bin/bash
    # now inside container
    ray status # should have 2 active nodes
    vllm serve /models/meta-llama/Llama-3.1-8B-Instruct -tp 8 # tensor-parallel 8 is most demanding of fast interconnect
    # THE NCCL TRACE ON STARTUP MIGHT BE INTERESTING, I DIDNT QUITE UNDERSTAND IT
    ```
* run the following on `nvl03` to run the benchmark (this starts yet another shell in the head node container and runs the benchmark):
    ```
    docker exec -it $(docker ps --format '{{.Names}}' | head -1) /bin/bash
    # now inside container
    vllm bench serve --backend openai --dataset-name random --model /models/meta-llama/Llama-3.1-8B-Instruct --seed 42
    ```
* the primary metric we want to compare is median TPOT: time per output token. I've copied below the current results of running the above; **NOTE: these results do not seem to reflect proper usage of the ConnectX7 card**:
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Benchmark duration (s):                  476.87
    Total input tokens:                      1021977
    Total generated tokens:                  122419
    Request throughput (req/s):              2.10
    Output token throughput (tok/s):         256.71
    Total Token throughput (tok/s):          2399.79
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          213969.93
    Median TTFT (ms):                        211585.74
    P99 TTFT (ms):                           429554.37
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1969.50
    Median TPOT (ms):                        2033.20
    P99 TPOT (ms):                           3260.55
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1921.21
    Median ITL (ms):                         3094.62
    P99 ITL (ms):                            4164.73
    ==================================================
    ```

# HPC GPU benchmarking

Benchmarking 2x nodes each with 4x H200 NVL 141 GB. With and without ConnectX-7 400 Gbps NIC (w/ GPUDirect RDMA).

# Single-node
create conda env

## Single-node 2GPU benchmark w/ NVLink
1. start vllm server - leave running
    1. variant with no tensor parallel, pipeline parallel 2
    1. variant with tensor parallel 2, no pipeline parallel
    ```bash
    CUDA_VISIBLE_DEVICES=0,1 vllm serve /opt/gpudata/models/meta-llama/Llama-3.1-8B-Instruct -tp X -pp Y
    ```
1. run benchmark:
    ```bash
    vllm bench serve --backend openai --dataset-name random --model /opt/gpudata/models/meta-llama/Llama-3.1-8B-Instruct --seed 42
    ```

## Single-node 2GPU benchmark w/o NVLink
1. computing `~/.cache/vllm/gpu_p2p_access_cache_for_1,2.json` causes process to hang on h200 nodes (haven't debugged why), but works on a100 node, so just copy from a100 node...:
    ```
    {
        "0->0": true,
        "0->1": true,
        "1->0": true,
        "1->1": true
    }
    ```
1. start vllm server - leave running
    1. variant with no tensor parallel, pipeline parallel 2
    1. variant with tensor parallel 2, no pipeline parallel
    ```bash
    CUDA_VISIBLE_DEVICES=1,2 vllm serve /opt/gpudata/models/meta-llama/Llama-3.1-8B-Instruct -tp X -pp Y
    ```
1. run benchmark:
    ```bash
    vllm bench serve --backend openai --dataset-name random --model /opt/gpudata/models/meta-llama/Llama-3.1-8B-Instruct --seed 42
    ```

## Single-node 4GPU benchmark
1. start vllm server - leave running
    1. variant with no tensor parallel, pipeline parallel 4
    1. variant with tensor parallel 2, pipeline parallel 2
    1. variant with tensor parallel 4, no pipeline parallel
    ```bash
    vllm serve /opt/gpudata/models/meta-llama/Llama-3.1-70B-Instruct -tp X -pp Y
    ```
1. run benchmark:
    ```bash
    vllm bench serve --backend openai --dataset-name random --model /opt/gpudata/models/meta-llama/Llama-3.1-70B-Instruct --seed 42
    ```

# Multi-node
## Double-node 8 GPU benchmark
1. require docker (for vllm multi-node deployment via ray)
    1. use version `vllm/vllm-openai:v0.10.0`
1. download models to local storage on each node
1. start ray cluster on head and worker nodes (use modified run_cluster script, will launch docker container) - leave running
    1. add additional GOO env var for [bugfix](https://github.com/vllm-project/vllm/issues/6775), on our nodes, happens to be `ens80f0np0`
    ```
    bash run_cluster.sh \
    vllm/vllm-openai \
    <HEAD_IP> \
    --<head|worker> \
    LOCAL_MODEL_DIR \
    -e VLLM_HOST_IP=<LOCAL_IP> \
    -e GLOO_SOCKET_IFNAME=ens80f0np0
    ```
1. connect to docker container on head node and start vllm server - leave running
    1. variant with no tensor parallel, pipeline parallel 8 (baseline?) (since not using tensor parallel, just omit the `-tp` flag)
    1. variant with tensor parallel 2, pipeline parallel 4 (use 2-way nvlinks)
    1. variant with tensor parallel 4, pipeline parallel 2 (intrAnode tensor parallel)
    1. variant with tensor parallel 8, no pipeline parallel (intErnode tensor parallel) (since not using pipeline parallel, just omit the `-pp` flag)
    ```bash
    vllm serve /models/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 -tp X -pp Y
    ```
1. connect to docker container on head node and run benchmark:
    ```bash
    vllm bench serve --backend openai --dataset-name random --model /models/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --seed 42
    ```
