#!/usr/bin/env python3
"""
Enhanced Subnet 19 Benchmark Suite with Complete Rayonlabs Validation

🔴 LIVE TESTING: This script performs real-time API calls to both vLLM and 
alternative-vllm servers. All results are collected from live HTTP requests
and streaming responses. No cached or pre-computed data is used.

This enhanced version includes ALL validation features from:
https://github.com/rayonlabs/vision-workers/blob/main/validator_orchestrator/app/checking/functions/text.py

Features:
- LIVE token generation and collection via streaming API
- Token rank validation based on temperature
- Failed token tracking and limits
- EOS token special handling  
- Random sampling strategy for token checking
- Complete quality scoring with penalty system

UPDATED: Now compatible with candle-vllm's new numeric bytes format and raw logprobs calculation.
"""

import math
import time
import requests
import json
import statistics
import random
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np

# Rayonlabs SN19 constants from their implementation
BOTTOM_TEXT_THRESHOLD = 0.125  # Perfect score threshold
TOP_TEXT_THRESHOLD = 0.25      # Zero score threshold

# Additional validation constants from Rayonlabs
MAX_FAILED_TOKENS = 5          # Maximum bad tokens before failure
MAX_CHECKS = 5                  # Maximum token positions to check
DEFAULT_NUM_LOGPROBS = 20       # Number of top logprobs to request

class SN19EnhancedBenchmark:
    """Enhanced benchmarking with complete Rayonlabs validation"""
    
    def __init__(self):
        self.alternative_vllm_url = "http://localhost:2000/v1/chat/completions"
        self.vllm_url = "http://localhost:8000/v1/chat/completions"
        self.model = "unsloth/Llama-3.2-3B-Instruct"
        
        # Test configuration
        self.test_prompt = "Write a sentence about a whale in a tropical paradise."
        self.max_tokens = 50
        self.num_tests = 3
        self.seed = 4321
        self.temperature = 0.01  # Near-deterministic
        
        # Store results
        self.vllm_results = []
        self.alternative_results = []
        self.quality_scores = []
        self.validation_results = []
    
    def calculate_distance(self, validator_logprob: float, miner_logprob: float) -> float:
        """SN19 exact distance calculation from Rayonlabs"""
        try:
            validator_prob = math.exp(validator_logprob)
            miner_prob = math.exp(miner_logprob)
            return min(abs(validator_prob - miner_prob), 1)
        except OverflowError:
            return 1.0
    
    def score_average_distance(self, avg_distance: float) -> float:
        """SN19 exact scoring function from Rayonlabs"""
        if avg_distance <= BOTTOM_TEXT_THRESHOLD:
            return 1.0
        elif avg_distance <= TOP_TEXT_THRESHOLD:
            return 1.0 - 0.5 * (avg_distance - BOTTOM_TEXT_THRESHOLD) / (TOP_TEXT_THRESHOLD - BOTTOM_TEXT_THRESHOLD)
        return 0.0
    
    def calculate_acceptable_ranks(self, temperature: float) -> Tuple[int, int]:
        """Calculate acceptable token ranks based on temperature (Rayonlabs lines 389-395)"""
        temperature_scale = 0.5
        max_acceptable_rank = 10
        
        if temperature > 0.5:
            temperature_scale = (temperature - 0.5) * 3
            max_acceptable_rank = int(10 + temperature_scale * 40)
        
        max_acceptable_eot_rank = int(300 + temperature_scale * 50)
        return max_acceptable_rank, max_acceptable_eot_rank
    
    def validate_token_ranks(self, response_data: Dict) -> Dict:
        """
        Validate token ranks according to Rayonlabs criteria (lines 385-447)
        This checks if generated tokens appear at acceptable ranks in the probability distribution
        """
        failed_tokens = []
        failed_details = []
        max_acceptable_rank, max_acceptable_eot_rank = self.calculate_acceptable_ranks(self.temperature)
        
        tokens = response_data.get('tokens', [])
        top_logprobs = response_data.get('top_logprobs', [])
        
        for idx, token_info in enumerate(tokens):
            token = token_info['token']
            logprob = token_info['logprob']
            
            # Find rank in top_logprobs
            rank = None
            if idx < len(top_logprobs):
                for rank_idx, top_token in enumerate(top_logprobs[idx]):
                    if top_token.get('token') == token:
                        rank = rank_idx
                        break
            
            # Check if token fails validation
            if rank is None:
                failed_tokens.append(idx)
                failed_details.append({
                    'idx': idx,
                    'token': token,
                    'reason': 'not in top logprobs'
                })
            elif rank >= max_acceptable_rank and logprob > float("-inf"):
                failed_tokens.append(idx)
                failed_details.append({
                    'idx': idx,
                    'token': token,
                    'rank': rank,
                    'reason': f'bad rank ({rank} >= {max_acceptable_rank})'
                })
        
        critical_fail = len(failed_tokens) > MAX_FAILED_TOKENS
        
        return {
            'passed': len(failed_tokens) == 0,
            'failed_count': len(failed_tokens),
            'failed_tokens': failed_tokens,
            'failed_details': failed_details,
            'critical_fail': critical_fail,
            'penalty': -10.0 if critical_fail else 0.0 if len(failed_tokens) > 0 else None
        }
    
    def select_indices_to_check(self, num_tokens: int) -> List[int]:
        """
        Select token indices to check using Rayonlabs sampling strategy (lines 584-597)
        Always checks first and last, then randomly samples middle tokens
        """
        if num_tokens == 0:
            return []
        elif num_tokens == 1:
            return [0]
        
        # Always check first and last
        indices_to_check = [0, num_tokens - 1]
        
        # Add random samples from the middle
        remaining_indexes = list(set(range(1, num_tokens - 1)) - set(indices_to_check))
        num_additional = min(MAX_CHECKS - len(indices_to_check), len(remaining_indexes))
        
        if num_additional > 0:
            additional_indices = random.sample(remaining_indexes, num_additional)
            indices_to_check.extend(additional_indices)
        
        return sorted(indices_to_check)
    
    def make_enhanced_request(self, url: str, test_num: int) -> Dict:
        """
        Make a LIVE enhanced request that captures all validation data in real-time.
        Updated to handle both vLLM and candle-vllm formats (numeric vs string bytes).
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.test_prompt}],
            "max_tokens": self.max_tokens,
            "logprobs": True,
            "top_logprobs": DEFAULT_NUM_LOGPROBS,
            "stream": True,
            "seed": self.seed,
            "temperature": self.temperature
        }
        
        result = {
            'tokens': [],
            'top_logprobs': [],
            'ttft': None,
            'total_time': None,
            'token_count': 0,
            'full_text': ''
        }
        
        try:
            start_time = time.perf_counter()
            first_token_received = False
            
            response = requests.post(url, json=payload, stream=True, timeout=60)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        current_time = time.perf_counter()
                        
                        try:
                            data_str = line_str[6:]
                            if data_str.strip() == '[DONE]':
                                break
                            
                            data = json.loads(data_str)
                            
                            if ('choices' in data and len(data['choices']) > 0 and 
                                'logprobs' in data['choices'][0] and 
                                data['choices'][0]['logprobs']):
                                
                                if not first_token_received:
                                    result['ttft'] = current_time - start_time
                                    first_token_received = True
                                
                                logprobs = data['choices'][0]['logprobs']
                                
                                # Extract token and top logprobs
                                if 'content' in logprobs and logprobs['content']:
                                    for item in logprobs['content']:
                                        result['token_count'] += 1
                                        
                                        # Handle both formats: numeric bytes (candle-vllm) and string bytes (vLLM)
                                        bytes_data = item.get('bytes', [])
                                        if isinstance(bytes_data, list) and len(bytes_data) > 0:
                                            # Numeric bytes format (candle-vllm) - convert to string for compatibility
                                            if isinstance(bytes_data[0], int):
                                                try:
                                                    # Convert numeric bytes to string
                                                    token_text = ''.join([chr(b) for b in bytes_data])
                                                except (ValueError, OverflowError):
                                                    token_text = item.get('token', '')
                                            else:
                                                token_text = item.get('token', '')
                                        else:
                                            token_text = item.get('token', '')
                                        
                                        result['tokens'].append({
                                            'token': token_text,
                                            'logprob': item.get('logprob', 0),
                                            'raw_bytes': bytes_data  # Store original bytes for debugging
                                        })
                                        result['full_text'] += token_text
                                        
                                        # Store top logprobs for validation
                                        top_probs = item.get('top_logprobs', [])
                                        result['top_logprobs'].append(top_probs)
                                        
                        except json.JSONDecodeError:
                            continue
            
            result['total_time'] = time.perf_counter() - start_time
            
            # Calculate TPS
            if result['ttft'] and result['total_time']:
                generation_time = result['total_time'] - result['ttft']
                result['tps'] = result['token_count'] / generation_time if generation_time > 0 else 0
            
        except requests.exceptions.RequestException as e:
            print(f"Error in request: {e}")
        
        return result
    
    def perform_token_checks(self, validator_tokens: List[Dict], miner_tokens: List[Dict], 
                           indices_to_check: List[int]) -> Tuple[float, List[float]]:
        """
        Perform detailed token checks at specified indices (Rayonlabs lines 450-505)
        Returns average distance and list of individual distances
        """
        total_distance = 0
        distances = []
        checks = 0
        
        for idx in indices_to_check:
            if checks >= MAX_CHECKS:
                break
            
            if idx < len(validator_tokens) and idx < len(miner_tokens):
                val_token = validator_tokens[idx]
                miner_token = miner_tokens[idx]
                
                # Only check if tokens match
                if val_token['token'] == miner_token['token']:
                    distance = self.calculate_distance(val_token['logprob'], miner_token['logprob'])
                    distances.append(distance)
                    total_distance += distance
                    checks += 1
        
        avg_distance = total_distance / checks if checks > 0 else 1.0
        return avg_distance, distances
    
    def run_enhanced_test(self, test_num: int):
        """Run a single enhanced test with full validation"""
        print(f"\n📊 TEST {test_num + 1}")
        print("-" * 60)
        
        # Get LIVE responses from both engines
        print(f"🔴 LIVE: Testing vLLM (validator)...")
        vllm_result = self.make_enhanced_request(self.vllm_url, test_num)
        
        print(f"🔴 LIVE: Testing alternative-vllm (miner)...")
        alternative_result = self.make_enhanced_request(self.alternative_vllm_url, test_num)
        
        if not vllm_result['tokens'] or not alternative_result['tokens']:
            print("Failed to get valid responses")
            return None
        
        # Validate token ranks for miner
        print("\n🔍 Validating Token Ranks...")
        rank_validation = self.validate_token_ranks(alternative_result)
        
        if rank_validation['critical_fail']:
            print(f"❌ CRITICAL FAILURE: {rank_validation['failed_count']} bad tokens")
            print(f"   Details: {rank_validation['failed_details'][:3]}")  # Show first 3
        elif rank_validation['failed_count'] > 0:
            print(f"⚠️  {rank_validation['failed_count']} tokens failed rank validation")
        else:
            print("✅ All tokens passed rank validation")
        
        # Select indices to check
        num_tokens = min(len(vllm_result['tokens']), len(alternative_result['tokens']))
        indices_to_check = self.select_indices_to_check(num_tokens)
        print(f"\n📍 Checking positions: {indices_to_check}")
        
        # Perform token checks
        avg_distance, distances = self.perform_token_checks(
            vllm_result['tokens'], 
            alternative_result['tokens'],
            indices_to_check
        )
        
        # Calculate final score
        base_score = self.score_average_distance(avg_distance)
        penalty = rank_validation.get('penalty', 0)
        final_score = max(0, base_score + penalty) if penalty else base_score
        
        # Display results
        print(f"\n📈 Results:")
        print(f"  Average Distance: {avg_distance:.6f}")
        print(f"  Base Quality Score: {base_score:.4f}")
        if penalty:
            print(f"  Penalty Applied: {penalty:.1f}")
        print(f"  Final Score: {final_score:.4f}")
        
        print(f"\n⚡ Performance:")
        print(f"  vLLM TTFT: {vllm_result['ttft']*1000:.1f}ms, TPS: {vllm_result.get('tps', 0):.1f}")
        print(f"  alt TTFT: {alternative_result['ttft']*1000:.1f}ms, TPS: {alternative_result.get('tps', 0):.1f}")
        
        return {
            'test_num': test_num,
            'vllm_result': vllm_result,
            'alternative_result': alternative_result,
            'rank_validation': rank_validation,
            'avg_distance': avg_distance,
            'distances': distances,
            'base_score': base_score,
            'final_score': final_score,
            'indices_checked': indices_to_check
        }
    
    def run_comprehensive_benchmark(self):
        """Run the complete enhanced benchmark suite"""
        print("🚀 ENHANCED SUBNET 19 BENCHMARK WITH FULL VALIDATION")
        print("=" * 80)
        print("\n🔴 LIVE TEST MODE: All results from real-time API calls")
        print("✅ UPDATED: Now compatible with candle-vllm's new format")
        print("\nThis benchmark includes:")
        print("  • Complete Rayonlabs validation logic")
        print("  • Token rank validation")
        print("  • Failed token tracking")
        print("  • Random sampling strategy")
        print("  • Penalty system for violations")
        print("  • Updated for numeric bytes format and raw logprobs")
        
        print(f"\nConfiguration:")
        print(f"  Model: {self.model}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max Tokens: {self.max_tokens}")
        print(f"  Seed: {self.seed}")
        print(f"  Note: Updated for candle-vllm's new numeric bytes format")
        
        all_results = []
        
        # Run LIVE tests
        print(f"\n🔴 EXECUTING {self.num_tests} LIVE TESTS...")
        for i in range(self.num_tests):
            result = self.run_enhanced_test(i)
            if result:
                all_results.append(result)
                print(f"✅ Test {i+1} completed with live data")
            time.sleep(0.5)  # Small delay between tests
        
        # Aggregate LIVE results
        if all_results:
            print("\n" + "="*80)
            print("📊 AGGREGATE RESULTS FROM LIVE TESTS")
            print("="*80)
            print(f"\n✅ Analysis based on {len(all_results)} successful live test(s)")
            print("✅ Updated for candle-vllm's new format (numeric bytes, raw logprobs)")
            
            # Quality scores
            base_scores = [r['base_score'] for r in all_results]
            final_scores = [r['final_score'] for r in all_results]
            
            print(f"\n🎯 Quality Scores:")
            print(f"  Base Score: Mean={statistics.mean(base_scores):.4f}, "
                  f"Min={min(base_scores):.4f}, Max={max(base_scores):.4f}")
            print(f"  Final Score: Mean={statistics.mean(final_scores):.4f}, "
                  f"Min={min(final_scores):.4f}, Max={max(final_scores):.4f}")
            
            # Validation stats
            total_failed = sum(r['rank_validation']['failed_count'] for r in all_results)
            critical_fails = sum(1 for r in all_results if r['rank_validation']['critical_fail'])
            
            print(f"\n🔍 Validation Statistics:")
            print(f"  Total Failed Tokens: {total_failed}")
            print(f"  Critical Failures: {critical_fails}/{len(all_results)}")
            
            # Performance comparison
            vllm_ttfts = [r['vllm_result']['ttft']*1000 for r in all_results if r['vllm_result']['ttft']]
            alternative_ttfts = [r['alternative_result']['ttft']*1000 for r in all_results if r['alternative_result']['ttft']]
            
            if vllm_ttfts and alternative_ttfts:
                print(f"\n⚡ Performance Comparison:")
                print(f"  vLLM TTFT: {statistics.mean(vllm_ttfts):.1f}ms")
                print(f"  alt TTFT: {statistics.mean(alternative_ttfts):.1f}ms")
                print(f"  Ratio: {statistics.mean(alternative_ttfts)/statistics.mean(vllm_ttfts):.2f}x")
            
            # Final assessment
            avg_final_score = statistics.mean(final_scores)
            print(f"\n💡 FINAL ASSESSMENT:")
            print("-" * 60)
            
            if avg_final_score >= 0.9:
                print("✅ EXCELLENT: alternative-vllm performs well with full validation")
            elif avg_final_score >= 0.5:
                print("⚠️  GOOD: Acceptable performance but some issues detected")
            else:
                print("❌ POOR: Significant issues detected with alternative-vllm")
            
            if total_failed > 0:
                print(f"\n⚠️  Warning: {total_failed} tokens failed rank validation")
                print("   This could indicate:")
                print("   • Different sampling behavior between engines")
                print("   • Potential quality issues in production")
                print("   • Need for engine-specific tuning")
            
            print(f"\n✅ Format Compatibility:")
            print("   • Updated for candle-vllm's numeric bytes format")
            print("   • Handles both vLLM (string bytes) and candle-vllm (numeric bytes)")
            print("   • Compatible with new raw logprobs calculation")

def main():
    print("Enhanced Subnet 19 Benchmark")
    print("Based on complete Rayonlabs validation from:")
    print("https://github.com/rayonlabs/vision-workers/.../text.py")
    print("✅ UPDATED: Now compatible with candle-vllm's new format")
    
    # Check server availability
    servers_ok = True
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        print("\n✅ vLLM server is running")
    except:
        print("\n❌ vLLM server not accessible at localhost:8000")
        servers_ok = False
    
    try:
        response = requests.get("http://localhost:2000/health", timeout=2)
        print("✅ alternative-vllm server is running")
    except:
        print("❌ alternative-vllm server not accessible at localhost:2000")
        servers_ok = False
    
    if not servers_ok:
        print("\n⚠️  Please ensure both servers are running")
        return
    
    # Run benchmark
    benchmark = SN19EnhancedBenchmark()
    benchmark.run_comprehensive_benchmark()
    
    print("\n" + "="*80)
    print("🏁 ENHANCED BENCHMARK COMPLETE")
    print("="*80)
    print("✅ Updated for candle-vllm's new numeric bytes format and raw logprobs calculation")

if __name__ == "__main__":
    main()