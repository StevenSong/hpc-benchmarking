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
    1. variant with no tensor parallel, pipeline parallel 8 (baseline?)
    1. variant with tensor parallel 2, pipeline parallel 4 (use 2-way nvlinks)
    1. variant with tensor parallel 4, pipeline parallel 2 (intrAnode tensor parallel)
    1. variant with tensor parallel 8, no pipeline parallel (intErnode tensor parallel)
    ```bash
    vllm serve /models/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 -tp X -pp Y
    ```
1. connect to docker container on head node and run benchmark:
    ```bash
    vllm bench serve --backend openai --dataset-name random --model /models/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --seed 42
    ```
