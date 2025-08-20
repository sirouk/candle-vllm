# Usage Example: Model Loader with HF Token

## Quick Start

1. **Set your Hugging Face Token (recommended):**
   ```bash
   export HF_TOKEN="your_token_here"
   ```

2. **Run the script:**
   ```bash
   cd model_loader
   ./model_loader.sh
   ```

## Example Session

### With HF Token Set

```bash
$ export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
$ cd model_loader
$ ./model_loader.sh

======================================
    Candle-vLLM Model Loader
======================================

Detected platform features: metal

Checking Hugging Face token status...
✓ HF_TOKEN environment variable found

Available Model Categories:
1. Bittensor SN19 Test Models
2. InsightsLM Test Models  
3. Custom Model (manual input)
4. Exit

Select an option (1-4): 1

Bittensor SN19 Test Models:
1. Qwen/QwQ-32B (uncompressed)
2. OpenGVLab/InternVL3-14B (uncompressed)
3. casperhansen/mistral-nemo-instruct-2407-awq (awq)
4. deepseek-ai/DeepSeek-R1-0528-Qwen3-8B (uncompressed)
5. unsloth/Llama-3.2-3B-Instruct (uncompressed)

Select a model (1-5): 4

Selected Model: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
Model Type: uncompressed

Port (default: 2000): 
Memory for KV Cache MB (default: 4096): 
Data type (default: bf16): 
Add in-situ quantization? (q4k/q5k/q6k/none) [none]: 

Found HF_TOKEN environment variable

Generated command:
cargo run --release --features metal -- --p 2000 --mem 4096 --dtype bf16 --hf-token HF_TOKEN --m "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" 

Run this command? (y/n): y
Starting model server...
```

### Without HF Token

```bash
$ cd model_loader  
$ ./model_loader.sh

======================================
    Candle-vLLM Model Loader
======================================

Detected platform features: metal

Checking Hugging Face token status...
⚠ No Hugging Face token found
  You can set one by:
  1. Setting HF_TOKEN environment variable
  2. Running 'huggingface-cli login' (if you have transformers installed)
  3. The script will prompt you for a token when needed
  Get a token at: https://huggingface.co/settings/tokens

Available Model Categories:
1. Bittensor SN19 Test Models
2. InsightsLM Test Models
3. Custom Model (manual input)
4. Exit

Select an option (1-4): 1
...
[After selecting model and options]

No Hugging Face token found in environment variables or default location
You may need a token to download models from Hugging Face

Options:
1. Set HF_TOKEN environment variable: export HF_TOKEN='your_token'
2. Use huggingface-cli login (if you have it installed)
3. Continue without token (may fail for private models)
Enter your choice (1-3) or press Enter to continue without token: 1
Enter your Hugging Face token: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_TOKEN environment variable set for this session

Generated command:
cargo run --release --features metal -- --p 2000 --mem 4096 --dtype bf16 --hf-token HF_TOKEN --m "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" 

Run this command? (y/n): y
```

## Testing the Server

Once your model is loaded and running, test it with:

```bash
./test_curl_script.sh 2000 "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
```

This will send a test request to the running model server.

## Token Sources (Priority Order)

1. `HF_TOKEN` environment variable
2. `HUGGING_FACE_HUB_TOKEN` environment variable  
3. `~/.cache/huggingface/token` file (from `huggingface-cli login`)
4. Manual input when prompted

## Important Notes

- **CLI Parameter Behavior**: The `--hf-token` parameter expects the **name of an environment variable**, not the token value directly
- **Private/Gated Models**: Some models require special access permissions on Hugging Face
- **Token Permissions**: Make sure your token has the appropriate read permissions
- **Security**: Tokens are handled securely and not logged in plain text

## Recent Fix

**Fixed Issue**: Previously, the script was passing token values directly to `--hf-token`, but candle-vllm expects this parameter to contain the name of an environment variable. The script now correctly passes environment variable names (like `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`) instead of the actual token values.
