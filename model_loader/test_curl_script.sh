#!/bin/bash

# Test script for making API requests to candle-vllm server
# Usage: ./test_curl_script.sh [port] [model_name]

PORT=${1:-2000}
MODEL=${2:-"default"}

echo "Testing candle-vllm API on port $PORT with model '$MODEL'"
echo "============================================================"

# Test the API with a simple request
curl -v -X POST "http://0.0.0.0:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\", 
        \"messages\": [
            {\"role\": \"user\", \"content\": \"Tell me a story about a whale in a tropical paradise.\"}
        ], 
        \"max_tokens\": 500
    }"
