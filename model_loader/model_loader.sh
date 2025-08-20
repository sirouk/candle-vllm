#!/bin/bash

# Candle-vLLM Model Loader Script
# This script helps load different LLM models using the correct candle-vllm commands

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_PORT=2000
DEFAULT_MEM=4096
DEFAULT_DTYPE="bf16"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}    Candle-vLLM Model Loader${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Function to check if Rust is installed and install if needed
check_rust_installation() {
    if command -v rustc &> /dev/null && command -v cargo &> /dev/null; then
        echo -e "${GREEN}✓ Rust is already installed${NC}"
        echo -e "${GREEN}  rustc version: $(rustc --version)${NC}"
        echo -e "${GREEN}  cargo version: $(cargo --version)${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Rust is not installed or not in PATH${NC}"
        echo -e "${YELLOW}Installing Rust using rustup...${NC}"
        
        if curl https://sh.rustup.rs -sSf | sh -s -- -y; then
            # Source the cargo environment
            source "$HOME/.cargo/env"
            
            # Verify installation
            if command -v rustc &> /dev/null && command -v cargo &> /dev/null; then
                echo -e "${GREEN}✓ Rust installed successfully!${NC}"
                echo -e "${GREEN}  rustc version: $(rustc --version)${NC}"
                echo -e "${GREEN}  cargo version: $(cargo --version)${NC}"
                return 0
            else
                echo -e "${RED}✗ Rust installation failed or not in PATH${NC}"
                echo -e "${RED}Please restart your shell or run: source \"\$HOME/.cargo/env\"${NC}"
                return 1
            fi
        else
            echo -e "${RED}✗ Failed to install Rust${NC}"
            echo -e "${RED}Please install Rust manually from https://rustup.rs/${NC}"
            return 1
        fi
    fi
}

# Function to check directory location and adjust if needed
check_directory_location() {
    # Check if we're in the project root (contains Cargo.toml and model_loader/ directory)
    if [[ -f "Cargo.toml" && -d "model_loader" && -f "model_loader/model_loader.sh" ]]; then
        echo -e "${GREEN}✓ Running from project root directory${NC}"
        PROJECT_ROOT="$(pwd)"
        MODEL_LOADER_DIR="$(pwd)/model_loader"
        BINARY_PATH="./target/release/candle-vllm"
        CARGO_CMD_PREFIX=""
        return 0
    # Check if we're in the model_loader directory (original behavior)
    elif [[ -f "model_loader.sh" && -f "../Cargo.toml" ]]; then
        echo -e "${GREEN}✓ Running from model_loader directory${NC}"
        PROJECT_ROOT="$(cd .. && pwd)"
        MODEL_LOADER_DIR="$(pwd)"
        BINARY_PATH="../target/release/candle-vllm"
        CARGO_CMD_PREFIX="cd .. && "
        return 0
    else
        echo -e "${RED}✗ Please run this script from either:${NC}"
        echo -e "${RED}  1. The project root directory (contains Cargo.toml)${NC}"
        echo -e "${RED}  2. The model_loader directory${NC}"
        echo -e "${RED}Current directory: $(pwd)${NC}"
        return 1
    fi
}

# Function to detect platform and set build features
detect_platform() {
    local os=$(uname -s)
    case "$os" in
        Darwin*)
            echo "metal"
            ;;
        Linux*)
            if command -v nvidia-smi &> /dev/null; then
                echo "cuda,nccl"
            else
                echo "cpu"
            fi
            ;;
        *)
            echo "cpu"
            ;;
    esac
}

# Function to check if candle-vllm is built and build it if needed
check_build() {
    if [[ ! -f "$BINARY_PATH" ]]; then
        echo -e "${YELLOW}Warning: candle-vllm binary not found at $BINARY_PATH${NC}"
        echo -e "${BLUE}Building candle-vllm with features: $FEATURES${NC}"
        echo ""
        
        read -p "Build candle-vllm now? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${GREEN}Building candle-vllm...${NC}"
            echo -e "${YELLOW}This may take several minutes...${NC}"
            echo ""
            
            # Run the build command
            local build_cmd="${CARGO_CMD_PREFIX}cargo build --release --features $FEATURES"
            echo -e "${BLUE}Running: $build_cmd${NC}"
            
            if eval "$build_cmd"; then
                echo ""
                echo -e "${GREEN}✓ Build completed successfully!${NC}"
                
                # Verify the binary was created
                if [[ -f "$BINARY_PATH" ]]; then
                    echo -e "${GREEN}✓ Binary found at: $BINARY_PATH${NC}"
                else
                    echo -e "${RED}✗ Build completed but binary not found at expected location${NC}"
                    return 1
                fi
            else
                echo ""
                echo -e "${RED}✗ Build failed!${NC}"
                echo -e "${RED}Please check the error messages above and try again${NC}"
                return 1
            fi
        else
            echo -e "${YELLOW}Build skipped. You may need to build manually later.${NC}"
            echo -e "${YELLOW}Command: ${CARGO_CMD_PREFIX}cargo build --release --features $FEATURES${NC}"
            return 1
        fi
    else
        echo -e "${GREEN}✓ candle-vllm binary found at: $BINARY_PATH${NC}"
    fi
}

# Check Rust installation
echo ""
echo -e "${BLUE}Checking Rust installation...${NC}"
if ! check_rust_installation; then
    echo -e "${RED}✗ Rust installation check failed. Exiting.${NC}"
    exit 1
fi

# Check directory location
echo ""
echo -e "${BLUE}Checking directory location...${NC}"
if ! check_directory_location; then
    echo -e "${RED}✗ Directory check failed. Exiting.${NC}"
    exit 1
fi

# Detect platform and set features
FEATURES=$(detect_platform)
echo -e "${GREEN}Detected platform features: $FEATURES${NC}"

# Check if candle-vllm is built and build if needed
echo ""
echo -e "${BLUE}Checking candle-vllm build status...${NC}"
if ! check_build; then
    echo -e "${RED}✗ Build check failed. Exiting.${NC}"
    exit 1
fi

# Check HF token status
echo ""
echo -e "${BLUE}Checking Hugging Face token status...${NC}"
if [[ -n "$HF_TOKEN" ]]; then
    echo -e "${GREEN}✓ HF_TOKEN environment variable found${NC}"
elif [[ -n "$HUGGING_FACE_HUB_TOKEN" ]]; then
    echo -e "${GREEN}✓ HUGGING_FACE_HUB_TOKEN environment variable found${NC}"
elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
    echo -e "${GREEN}✓ Hugging Face token file found at ~/.cache/huggingface/token${NC}"
else
    echo -e "${YELLOW}⚠ No Hugging Face token found${NC}"
    echo -e "${YELLOW}  You can set one by:${NC}"
    echo -e "${YELLOW}  1. Setting HF_TOKEN environment variable${NC}"
    echo -e "${YELLOW}  2. Running 'huggingface-cli login' (if you have transformers installed)${NC}"
    echo -e "${YELLOW}  3. The script will prompt you for a token when needed${NC}"
    echo -e "${YELLOW}  Get a token at: https://huggingface.co/settings/tokens${NC}"
fi

# Model categories and definitions (model_id:type format)
BITTENSOR_MODELS=(
    "Qwen/QwQ-32B:uncompressed"
    "OpenGVLab/InternVL3-14B:uncompressed"
    "casperhansen/mistral-nemo-instruct-2407-awq:awq"
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B:uncompressed"
    "unsloth/Llama-3.2-3B-Instruct:uncompressed"
)

INSIGHTSLM_MODELS=(
    "Qwen/Qwen3-8B-GGUF:gguf:qwen3-8b-gguf-q4_k_m.gguf"
    "google/gemma-3-27b-it:uncompressed"
    "openai/gpt-oss-20b:uncompressed"
)

# Function to show model menu
show_menu() {
    echo -e "${BLUE}Available Model Categories:${NC}"
    echo "1. Bittensor SN19 Test Models"
    echo "2. InsightsLM Test Models"
    echo "3. Custom Model (manual input)"
    echo "4. Exit"
    echo ""
}

# Function to show Bittensor models
show_bittensor_models() {
    echo -e "${BLUE}Bittensor SN19 Test Models:${NC}"
    local i=1
    for model_entry in "${BITTENSOR_MODELS[@]}"; do
        local model_id="${model_entry%%:*}"
        local model_type="${model_entry#*:}"
        echo "$i. $model_id ($model_type)"
        ((i++))
    done
    echo ""
}

# Function to show InsightsLM models
show_insightslm_models() {
    echo -e "${BLUE}InsightsLM Test Models:${NC}"
    local i=1
    for model_entry in "${INSIGHTSLM_MODELS[@]}"; do
        local model_id="${model_entry%%:*}"
        local model_type="${model_entry#*:}"
        echo "$i. $model_id ($model_type)"
        ((i++))
    done
    echo ""
}

# Function to check and get Hugging Face token
get_hf_token() {
    local hf_token_param=""
    
    # Check environment variables - pass the ENV VAR NAME, not the token value
    if [[ -n "$HF_TOKEN" ]]; then
        echo -e "${GREEN}Found HF_TOKEN environment variable${NC}" >&2
        hf_token_param="--hf-token HF_TOKEN"
    elif [[ -n "$HUGGING_FACE_HUB_TOKEN" ]]; then
        echo -e "${GREEN}Found HUGGING_FACE_HUB_TOKEN environment variable${NC}" >&2
        hf_token_param="--hf-token HUGGING_FACE_HUB_TOKEN"
    elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
        echo -e "${GREEN}Found Hugging Face token file at ~/.cache/huggingface/token${NC}" >&2
        hf_token_param="--hf-token-path $HOME/.cache/huggingface/token"
    else
        echo -e "${YELLOW}No Hugging Face token found in environment variables or default location${NC}" >&2
        echo -e "${YELLOW}You may need a token to download models from Hugging Face${NC}" >&2
        
        echo -e "${YELLOW}Options:${NC}" >&2
        echo -e "${YELLOW}1. Set HF_TOKEN environment variable: export HF_TOKEN='your_token'${NC}" >&2
        echo -e "${YELLOW}2. Use huggingface-cli login (if you have it installed)${NC}" >&2
        echo -e "${YELLOW}3. Continue without token (may fail for private models)${NC}" >&2
        read -p "Enter your choice (1-3) or press Enter to continue without token: " choice
        
        case "$choice" in
            1)
                read -p "Enter your Hugging Face token: " user_token
                if [[ -n "$user_token" ]]; then
                    export HF_TOKEN="$user_token"
                    echo -e "${GREEN}HF_TOKEN environment variable set for this session${NC}" >&2
                    hf_token_param="--hf-token HF_TOKEN"
                fi
                ;;
            2)
                echo -e "${YELLOW}Please run 'huggingface-cli login' in another terminal, then restart this script${NC}" >&2
                ;;
            *)
                echo -e "${YELLOW}Proceeding without token (may fail for private/gated models)${NC}" >&2
                ;;
        esac
    fi
    
    # Return the appropriate parameter
    echo "$hf_token_param"
}

# Function to get model command
get_model_command() {
    local model_id="$1"
    local model_type="$2"
    local port="${3:-$DEFAULT_PORT}"
    local mem="${4:-$DEFAULT_MEM}"
    local dtype="${5:-$DEFAULT_DTYPE}"
    
    local cmd="${CARGO_CMD_PREFIX}cargo run --release --features $FEATURES --"
    cmd="$cmd --p $port --mem $mem --dtype $dtype"
    
    # Add HF token if available
    local hf_token_param=$(get_hf_token)
    if [[ -n "$hf_token_param" ]]; then
        cmd="$cmd $hf_token_param"
    fi
    
    case "$model_type" in
        "uncompressed")
            cmd="$cmd --m \"$model_id\""
            ;;
        "awq")
            echo -e "${YELLOW}Note: AWQ models may require conversion to Marlin format for optimal performance${NC}"
            cmd="$cmd --m \"$model_id\""
            ;;
        "gguf:"*)
            local gguf_file="${model_type#gguf:}"
            cmd="$cmd --m \"$model_id\" --f \"$gguf_file\""
            ;;
        *)
            cmd="$cmd --m \"$model_id\""
            ;;
    esac
    
    echo "$cmd"
}

# Function to run selected model
run_model() {
    local model_id="$1"
    local model_type="$2"
    
    echo ""
    echo -e "${GREEN}Selected Model: $model_id${NC}"
    echo -e "${GREEN}Model Type: $model_type${NC}"
    echo ""
    
    # Get configuration options
    read -p "Port (default: $DEFAULT_PORT): " port
    port=${port:-$DEFAULT_PORT}
    
    read -p "Memory for KV Cache MB (default: $DEFAULT_MEM): " mem
    mem=${mem:-$DEFAULT_MEM}
    
    read -p "Data type (default: $DEFAULT_DTYPE): " dtype
    dtype=${dtype:-$DEFAULT_DTYPE}
    
    # Additional options for specific model types
    local additional_opts=""
    case "$model_type" in
        "awq")
            echo -e "${YELLOW}AWQ model detected. You may need to convert to Marlin format first.${NC}"
            read -p "Add in-situ quantization? (q4k/q5k/none) [none]: " isq
            if [[ -n "$isq" && "$isq" != "none" ]]; then
                additional_opts="$additional_opts --isq $isq"
            fi
            ;;
        "uncompressed")
            read -p "Add in-situ quantization? (q4k/q5k/q6k/none) [none]: " isq
            if [[ -n "$isq" && "$isq" != "none" ]]; then
                additional_opts="$additional_opts --isq $isq"
            fi
            ;;
    esac
    
    # Generate and display command
    local cmd=$(get_model_command "$model_id" "$model_type" "$port" "$mem" "$dtype")
    cmd="$cmd $additional_opts"
    
    echo ""
    echo -e "${BLUE}Generated command:${NC}"
    echo -e "${GREEN}$cmd${NC}"
    echo ""
    
    # Ask for confirmation
    read -p "Run this command? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Starting model server...${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
        echo ""
        
        # Execute the command
        eval "$cmd"
    else
        echo -e "${YELLOW}Command execution cancelled.${NC}"
    fi
}

# Function to select from Bittensor models
select_bittensor_model() {
    show_bittensor_models
    read -p "Select a model (1-${#BITTENSOR_MODELS[@]}): " choice
    
    if [[ $choice -ge 1 && $choice -le ${#BITTENSOR_MODELS[@]} ]]; then
        local selected_entry="${BITTENSOR_MODELS[$((choice-1))]}"
        local model_id="${selected_entry%%:*}"
        local model_type="${selected_entry#*:}"
        run_model "$model_id" "$model_type"
    else
        echo -e "${RED}Invalid selection!${NC}"
    fi
}

# Function to select from InsightsLM models
select_insightslm_model() {
    show_insightslm_models
    read -p "Select a model (1-${#INSIGHTSLM_MODELS[@]}): " choice
    
    if [[ $choice -ge 1 && $choice -le ${#INSIGHTSLM_MODELS[@]} ]]; then
        local selected_entry="${INSIGHTSLM_MODELS[$((choice-1))]}"
        local model_id="${selected_entry%%:*}"
        local model_type="${selected_entry#*:}"
        run_model "$model_id" "$model_type"
    else
        echo -e "${RED}Invalid selection!${NC}"
    fi
}

# Function for custom model input
custom_model() {
    echo -e "${BLUE}Custom Model Configuration${NC}"
    read -p "Enter Hugging Face model ID (e.g., microsoft/DialoGPT-medium): " model_id
    
    if [[ -z "$model_id" ]]; then
        echo -e "${RED}Model ID cannot be empty!${NC}"
        return
    fi
    
    echo ""
    echo "Model Types:"
    echo "1. Uncompressed (default)"
    echo "2. GGUF (specify filename)"
    echo "3. AWQ"
    read -p "Select model type (1-3): " type_choice
    
    local model_type="uncompressed"
    case "$type_choice" in
        2)
            read -p "Enter GGUF filename: " gguf_file
            if [[ -n "$gguf_file" ]]; then
                model_type="gguf:$gguf_file"
            fi
            ;;
        3)
            model_type="awq"
            ;;
    esac
    
    run_model "$model_id" "$model_type"
}

# Main menu loop
while true; do
    show_menu
    read -p "Select an option (1-4): " choice
    
    case $choice in
        1)
            select_bittensor_model
            ;;
        2)
            select_insightslm_model
            ;;
        3)
            custom_model
            ;;
        4)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option! Please select 1-4.${NC}"
            echo ""
            ;;
    esac
    
    echo ""
    echo -e "${BLUE}======================================${NC}"
    echo ""
done
