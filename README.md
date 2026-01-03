# R²Adapter: A Plug-in Routing and Rewriting Adapter for Efficient Hybrid RAG

This repository contains the code for the paper
"R²Adapter: A Plug-in Routing and Rewriting Adapter for Efficient Hybrid Retrieval-Augmented Generation".
R²Adapter is a lightweight, model-agnostic plug-in that dynamically routes user queries between vanilla (passage-based) RAG and graph-based RAG, and selectively rewrites uncertain graph-routed queries.

---

## ✨ Highlights

- 🧠 Light-weight router trained to decide whether a query needs graph-based reasoning.
- ✍️ Optional LLM-based rewriter that refines low-confidence graph-routed queries into structured triplets.
- 🔌 Plug-and-play with vanilla and graph-based RAG systems with no modification to underlying systems required (Employments with vanilla RAG + GraphRAG/HippoRAG 2 are provided in this repository).

## 🚀 Quick Start

### 1) Environment 🔧

Follow GraphRAG and HippoRAG setup instructions before running the experiments:

```bash
# GraphRAG setup (follow GraphRAG build instructions)
cd graphrag
# build based on their guidelines: https://github.com/microsoft/graphrag
# If using NV-Embed-v2, note to place the model files we provided into:
# graphrag/graphrag/register_nvembed_model.py
# graphrag/graphrag/nvembed_embedding_model.py 

# Return to project root
cd ..

# Ensure HippoRAG dependencies are installed
cd HippoRAG
pip3 install -r requirements.txt

```

### 2) Train the Router 🎯

Train a lightweight DeBERTa-based router that predicts suitability for passage vs. graph-based RAG:

```bash
 python train_router.py \
     --base_model_path path/to/deberta-v3-base \
     --train_path router_training_dataset/router_preference_data_train.json \
     --dev_path router_training_dataset/router_preference_data_val.json \
     --test_path router_training_dataset/router_preference_data_test.json \
     --save_dir path/to/router_output_dir \
     --batch_size 64 \
     --epochs 3 \
     --lr 1e-5 \
     --eval_steps 50 \
     --seed 215 \
     --threshold_metric f1_recall_balance
```

**Parameter Description:**

- `--base_model_path`: Path to base encoder (example: `path/to/deberta-v3-base`).
- `--train_path`: Path to training JSON.
- `--dev_path`: Path to validation JSON.
- `--test_path`: Path to test JSON.
- `--save_dir`: Directory where checkpoints and thresholds will be saved.
- `--batch_size`: Training batch size.
- `--epochs`: Number of epochs.
- `--lr`: Learning rate.
- `--eval_steps`: Evaluation frequency in steps.
- `--seed`: Random seed for reproducibility.
- `--threshold_metric`: Metric for picking decision threshold.

### 3) Testing🧪

Before running integration experiments, start a vLLM server for generation:

```bash
 export CUDA_VISIBLE_DEVICES=0,1,2,3
 export VLLM_WORKER_MULTIPROC_METHOD=spawn
 vllm serve path/to/Llama-3.3-70B-Instruct \
     --max_model_len 32768 \
     --tensor-parallel-size 4 \
     --served-model-name Llama-3.3-70B-Instruct \
     --gpu-memory-utilization 0.95 \
     --port 8000
```

**Parameter Description:**

- `path/to/Llama-3.3-70B-Instruct`: Directory or HF cache path for the model.
- `--max_model_len`: Maximum sequence length.
- `--tensor-parallel-size`: Tensor-parallelism across GPUs.
- `--served-model-name`: Label for the served model.
- `--gpu-memory-utilization`: Fraction of GPU memory to target.
- `--port`: HTTP port for model API.

GraphRAG integration:

```bash
 export CUDA_VISIBLE_DEVICES=6,7
 python adapter_integrated_graphrag.py \
     --router_model_path path/to/router/best_router.pt \
     --threshold_file path/to/router/best_threshold.json \
     --low_probability_threshold 0.6 \
     --dataset hotpotqa \
     --save_dir graphrag/outputs \
     --log_file graphrag/outputs/hotpotqa/router_rewriter_0.6.log \
     --config_file graphrag/config.yaml
```

**Parameter Description:**

- `--router_model_path`: Router checkpoint (output file of the router training process).
- `--threshold_file`: JSON with selected threshold (output file of the router training process).
- `--low_probability_threshold`: Confidence cutoff to trigger rewriter.
- `--dataset`: Dataset identifier (`hotpotqa/2wikimultihopqa/musique`).
- `--save_dir`: Directory to store outputs.
- `--log_file`: Path to write run logs.
- `--config_file`: Path to GraphRAG config YAML (Our config file is provided in `graphrag/config.yaml`).

 HippoRAG 2 integration:

```bash
 python adapter_integrated_hipporag.py \
     --router_model_path path/to/router/best_router.pt \
     --threshold_file path/to/router/best_threshold.json \
     --low_probability_threshold 0.6 \
     --dataset hotpotqa \
     --save_dir HippoRAG/outputs \
     --log_file HippoRAG/outputs/hotpotqa/router_rewriter_0.6.log \
     --llm_base_url http://localhost:8000/v1 \
     --llm_name Llama-3.3-70B-Instruct \
     --embedding_name path/to/NV-Embed-v2
```

**Parameter Description:**

- `--router_model_path`: Router checkpoint.
- `--threshold_file`: Threshold JSON.
- `--low_probability_threshold`: Confidence cutoff to trigger rewriter.
- `--dataset`: Dataset identifier (`hotpotqa/2wikimultihopqa/musique`).
- `--save_dir`: Directory to store outputs.
- `--log_file`: Path to write run logs.
- `--llm_base_url`: Base URL for the vLLM service.
- `--llm_name`: Served model name.
- `--embedding_name`: Embedding model path.

## 📁 Directory structure

```text
 .
 ├─ 📂 graphrag/                      # GraphRAG code, configs and helpers
 │  ├─ config.yaml                    # GraphRAG experiment configuration
 │  ├─ ...                            # other GraphRAG files and assets
 │  └─ graphrag/
 │     ├─ register_nvembed_model.py   # helper to register NV-Embed-v2 with GraphRAG
 │     ├─ nvembed_embedding_model.py  # NV-Embed-v2 embedding model
 │     └─ ...                         # GraphRAG internal modules
 ├─ 📂 HippoRAG/                      # HippoRAG code
 │  └─ ...  
 ├─ 📂 router_training_dataset/       # JSON datasets used for router training (train/dev/test)
 │  ├─ router_preference_data_test.json
 │  ├─ router_preference_data_train.json
 │  └─ router_preference_data_val.json
 ├─ adapter_integrated_graphrag.py    # Main integration script to run R²Adapter with GraphRAG (vanilla RAG + GraphRAG)
 ├─ adapter_integrated_hipporag.py    # Main integration script to run R²Adapter with HippoRAG 2 (vanilla RAG + HippoRAG 2)
 ├─ 📜 README.md  
 ├─ rewriter.py                        # LLM-based rewriter
 └─ train_router.py                    # Script to train the router
```

## 🙏 Acknowledgment

We would like to thank the [GraphRAG](https://github.com/microsoft/graphrag) and [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) teams for their open-source work.
