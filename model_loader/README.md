# Candle-vLLM Model Loader Script

This bash script provides an interactive menu system for loading different LLM models using candle-vllm with the correct command-line parameters.

## Features

- **Automatic Platform Detection**: Detects macOS (Metal) vs Linux (CUDA) automatically
- **Hugging Face Token Management**: Automatically detects and uses HF tokens from environment or prompts user
- **Pre-configured Models**: Two categories of test models ready to use:
  - Bittensor SN19 Test Models
  - InsightsLM Test Models
- **Custom Model Support**: Allows manual input of any Hugging Face model
- **Interactive Configuration**: Prompts for port, memory, data type, and quantization options
- **Command Generation**: Shows the exact command before execution for transparency

## Usage

### Prerequisites

1. Make sure candle-vllm is built (run from repository root):
   ```bash
   # For macOS
   cargo build --release --features metal
   
   # For Linux with CUDA
   cargo build --release --features cuda,nccl
   ```

2. **Hugging Face Token (Recommended)**: Set up authentication for downloading models:
   ```bash
   # Option 1: Set environment variable
   export HF_TOKEN="your_token_here"
   
   # Option 2: Use Hugging Face CLI (if you have transformers installed)
   pip install huggingface_hub
   huggingface-cli login
   
   # Option 3: The script will prompt you for a token if needed
   ```
   
   Get your token at: https://huggingface.co/settings/tokens

### Running the Script

From the repository root:
```bash
cd model_loader
./model_loader.sh
```

Or directly:
```bash
model_loader/model_loader.sh
```

### Pre-configured Models

#### Bittensor SN19 Test Models:
- `Qwen/QwQ-32B` (uncompressed)
- `OpenGVLab/InternVL3-14B` (uncompressed) 
- `casperhansen/mistral-nemo-instruct-2407-awq` (AWQ)
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` (uncompressed)
- `unsloth/Llama-3.2-3B-Instruct` (uncompressed)

#### InsightsLM Test Models:
- `Qwen/Qwen3-8B-GGUF` (GGUF)
- `google/gemma-3-27b-it` (uncompressed)
- `openai/gpt-oss-20b` (uncompressed)

### Testing the API

After starting a model server, you can test it using the provided test script:

```bash
./test_curl_script.sh [port] [model_name]
```

Example:
```bash
./test_curl_script.sh 2000 "Qwen/QwQ-32B"
```

### Configuration Options

The script will prompt for:

- **Port**: Server port (default: 2000)
- **Memory**: KV Cache memory in MB (default: 4096) 
- **Data Type**: Model data type (default: bf16)
- **Quantization**: In-situ quantization options (q4k, q5k, q6k, etc.)

### Generated Commands

The script generates commands in the format:
```bash
cargo run --release --features [metal/cuda,nccl] -- --p [port] --mem [memory] --dtype [dtype] [--hf-token ENV_VAR_NAME | --hf-token-path /path/to/token] --m "[model-id]" [additional options]
```

### Example Generated Commands

For uncompressed models:
```bash
cargo run --release --features metal -- --p 2000 --mem 4096 --dtype bf16 --hf-token HUGGING_FACE_HUB_TOKEN --m "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
```

For GGUF models:
```bash
cargo run --release --features metal -- --p 2000 --mem 4096 --dtype bf16 --hf-token-path /Users/username/.cache/huggingface/token --m "Qwen/Qwen3-8B-GGUF" --f "qwen3-8b-gguf-q4_k_m.gguf"
```

**Note**: The `--hf-token` parameter expects the **name of an environment variable**, not the token value directly.

## Files

- `model_loader.sh`: Main interactive model loader script
- `test_curl_script.sh`: API testing script for making requests
- `README.md`: This documentation file

## Platform Support

- **macOS**: Uses `--features metal` 
- **Linux with NVIDIA GPUs**: Uses `--features cuda,nccl`
- **CPU-only**: Uses default features (fallback)

## Troubleshooting

1. **Permission denied**: Make sure scripts are executable
   ```bash
   chmod +x model_loader.sh test_curl_script.sh
   ```

2. **Build not found**: Build candle-vllm first
   ```bash
   cargo build --release --features [metal/cuda,nccl]
   ```

3. **Hugging Face token issues**: 
   - Get a token at https://huggingface.co/settings/tokens
   - Set `HF_TOKEN` environment variable or use `huggingface-cli login`
   - The script will prompt for a token if none is found

4. **Model download issues**: Ensure you have sufficient disk space and internet connection for Hugging Face model downloads

5. **CUDA issues on Linux**: Ensure CUDA toolkit is properly installed and in PATH
   ```bash
   export PATH=$PATH:/usr/local/cuda/bin/
   ```

6. **Private/Gated models**: Some models require approval on Hugging Face and a valid token with appropriate permissions
