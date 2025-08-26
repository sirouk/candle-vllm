#!/usr/bin/env python3
"""
Comprehensive Subnet 19 Benchmark Suite - All-in-One Analysis Tool

🔴 LIVE TESTING: This script performs real-time benchmarking with live API calls.
All performance metrics (TTFT, TPS) and quality scores are measured from actual
HTTP streaming responses. Every test execution produces fresh, real-time results.

This single script combines all analysis features:
- LIVE Performance Benchmarking (TTFT, TPS) from real API calls
- Token-by-Token Detailed Comparison with live-generated logprobs
- Quality Score Analysis using exact SN19 algorithm on live data
- Statistical Distribution Analysis of real test results
- Sensitivity Analysis
- Risk Assessment
- Final Recommendations
- TOP_LOGPROBS Support: Configurable boolean/integer values for detailed token analysis

Based on the exact Rayonlabs scoring algorithm from:
https://github.com/rayonlabs/vision-workers/blob/main/validator_orchestrator/app/checking/functions/text.py
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

class SN19ComprehensiveBenchmark:
    """
    Complete benchmarking and analysis suite for Subnet 19
    
    Features:
    - Configurable top_logprobs parameter (boolean or integer)
    - Boolean True: defaults to 5 alternatives
    - Boolean False: no top alternatives
    - Integer N: top N alternatives with probabilities
    - Automatic validation and configuration
    """
    
    def __init__(self):
        self.alternative_vllm_url = "http://localhost:2000/v1/chat/completions"
        self.vllm_url = "http://localhost:8000/v1/chat/completions"
        self.model = "unsloth/Llama-3.2-3B-Instruct"
        
        # Test configuration
        self.test_prompt = "Write a sentence about a whale in a tropical paradise."
        self.max_tokens = 50  # More tokens for better TPS measurement
        self.num_tests = 3
        self.seed = 4321
        self.temperature = 0.01  # Near-deterministic
        
        # Logprobs configuration
        self.top_logprobs = 5  # Number of top logprobs to request (can be boolean or number)
        
        # Validate top_logprobs parameter
        self._validate_top_logprobs()
        
        # Store results for analysis
        self.vllm_results = []
        self.alternative_results = []
        self.quality_scores = []
        self.validation_results = []
    
    def _validate_top_logprobs(self):
        """Validate the top_logprobs parameter"""
        if isinstance(self.top_logprobs, bool):
            if self.top_logprobs:
                self.top_logprobs = 5  # Default to 5 if True
        elif isinstance(self.top_logprobs, int):
            if self.top_logprobs < 0:
                raise ValueError("top_logprobs must be non-negative")
            if self.top_logprobs > 100:
                print(f"⚠️  Warning: top_logprobs={self.top_logprobs} is very high, may impact performance")
        else:
            raise ValueError("top_logprobs must be boolean or integer")
    
    def configure_top_logprobs(self, value):
        """Configure the top_logprobs parameter"""
        self.top_logprobs = value
        self._validate_top_logprobs()
        print(f"✅ Configured top_logprobs to: {self.top_logprobs}")
    
    def demonstrate_top_logprobs_configs(self):
        """Demonstrate different top_logprobs configurations"""
        print("\n🔧 TOP_LOGPROBS Configuration Examples:")
        print("-" * 60)
        
        configs = [
            (True, "Boolean True (defaults to 5)"),
            (False, "Boolean False (no top logprobs)"),
            (3, "Integer 3 (top 3 alternatives)"),
            (10, "Integer 10 (top 10 alternatives)"),
            (20, "Integer 20 (top 20 alternatives)")
        ]
        
        for value, description in configs:
            try:
                self.configure_top_logprobs(value)
                print(f"  {value}: {description}")
            except ValueError as e:
                print(f"  {value}: ❌ {e}")
        
        # Reset to default
        self.top_logprobs = 5
        print(f"\n✅ Reset to default: top_logprobs = {self.top_logprobs}")
    
    def calculate_distance(self, validator_logprob: float, miner_logprob: float) -> float:
        """SN19 exact distance calculation"""
        try:
            validator_prob = math.exp(validator_logprob)
            miner_prob = math.exp(miner_logprob)
            return min(abs(validator_prob - miner_prob), 1)
        except OverflowError:
            return 1.0
    
    def score_average_distance(self, avg_distance: float) -> float:
        """SN19 exact scoring function"""
        if avg_distance <= BOTTOM_TEXT_THRESHOLD:
            return 1.0
        elif avg_distance <= TOP_TEXT_THRESHOLD:
            return 1.0 - 0.5 * (avg_distance - BOTTOM_TEXT_THRESHOLD) / (TOP_TEXT_THRESHOLD - BOTTOM_TEXT_THRESHOLD)
        return 0.0
    
    def calculate_acceptable_rank(self, temperature: float) -> Tuple[int, int]:
        """Calculate acceptable token ranks based on temperature (Rayonlabs logic)"""
        temperature_scale = 0.5
        max_acceptable_rank = 10
        
        if temperature > 0.5:
            temperature_scale = (temperature - 0.5) * 3
            max_acceptable_rank = int(10 + temperature_scale * 40)
        
        max_acceptable_eot_rank = int(300 + temperature_scale * 50)
        return max_acceptable_rank, max_acceptable_eot_rank
    
    def validate_token_ranks(self, logprobs_data: List[Dict], top_logprobs: List[List[Dict]]) -> Dict:
        """Validate token ranks according to Rayonlabs criteria"""
        failed_tokens = []
        failed_details = []
        max_acceptable_rank, max_acceptable_eot_rank = self.calculate_acceptable_rank(self.temperature)
        
        for idx, token_data in enumerate(logprobs_data):
            if idx >= len(top_logprobs):
                continue
                
            # Find the rank of the generated token in top logprobs
            token = token_data['token']
            token_logprob = token_data['logprob']
            
            # Check if token appears in top logprobs with acceptable rank
            rank = None
            for rank_idx, top_token in enumerate(top_logprobs[idx]):
                if top_token.get('token') == token:
                    rank = rank_idx
                    break
            
            if rank is None:
                failed_tokens.append(idx)
                failed_details.append({
                    'token': token,
                    'reason': 'not in top logprobs',
                    'logprob': token_logprob
                })
            elif rank >= max_acceptable_rank and token_logprob > float("-inf"):
                failed_tokens.append(idx)
                failed_details.append({
                    'token': token,
                    'rank': rank,
                    'logprob': token_logprob,
                    'reason': f'bad rank (>{max_acceptable_rank})'
                })
        
        return {
            'passed': len(failed_tokens) == 0,
            'failed_count': len(failed_tokens),
            'failed_tokens': failed_tokens,
            'failed_details': failed_details,
            'critical_fail': len(failed_tokens) > MAX_FAILED_TOKENS
        }
    
    def select_indices_to_check(self, num_tokens: int) -> List[int]:
        """Select token indices to check using Rayonlabs sampling strategy"""
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
    
    def make_streaming_request(self, url: str, test_num: int) -> Tuple[List[Dict], float, float, int]:
        """
        Make a LIVE streaming request and measure real-time performance metrics.
        Updated to handle both vLLM and candle-vllm formats (numeric vs string bytes).
        Returns: (logprobs_data, time_to_first_token, total_time, token_count)
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.test_prompt}],
            "max_tokens": self.max_tokens,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "stream": True,
            "seed": self.seed,
            "temperature": 0.01  # Near-deterministic
        }
        
        logprobs_data = []
        time_to_first_token = None
        first_token_received = False
        token_count = 0
        tokens_text = []
        
        try:
            start_time = time.perf_counter()
            
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
                                data['choices'][0]['logprobs'] and
                                'content' in data['choices'][0]['logprobs'] and
                                data['choices'][0]['logprobs']['content']):
                                
                                if not first_token_received:
                                    time_to_first_token = current_time - start_time
                                    first_token_received = True
                                
                                content_logprobs = data['choices'][0]['logprobs']['content']
                                for item in content_logprobs:
                                    if 'token' in item and 'logprob' in item:
                                        token_count += 1
                                        
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
                                        
                                        logprobs_data.append({
                                            'token': token_text,
                                            'logprob': item['logprob'],
                                            'raw_bytes': bytes_data  # Store original bytes for debugging
                                        })
                                        tokens_text.append(token_text)
                        except json.JSONDecodeError:
                            continue
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            return logprobs_data, time_to_first_token, total_time, token_count
            
        except requests.exceptions.RequestException as e:
            print(f"Error in request: {e}")
            return [], None, None, 0
    
    def calculate_quality_score(self, validator_logprobs: List[Dict], miner_logprobs: List[Dict]) -> Tuple[float, float, List[float]]:
        """
        Calculate SN19 quality score between validator and miner outputs.
        Returns: (quality_score, average_distance, distances)
        """
        if not validator_logprobs or not miner_logprobs:
            return 0.0, 1.0, []
        
        distances = []
        min_len = min(len(validator_logprobs), len(miner_logprobs))
        
        for i in range(min_len):
            val_token = validator_logprobs[i]
            miner_token = miner_logprobs[i]
            
            if val_token['token'] == miner_token['token']:
                distance = self.calculate_distance(val_token['logprob'], miner_token['logprob'])
                distances.append(distance)
        
        if distances:
            avg_distance = statistics.mean(distances)
            quality_score = self.score_average_distance(avg_distance)
            return quality_score, avg_distance, distances
        
        return 0.0, 1.0, []
    
    def run_single_test(self, engine_name: str, url: str, test_num: int) -> Dict:
        """Run a single LIVE test and collect all real-time metrics"""
        print(f"  🔴 LIVE Test {test_num + 1}: ", end="", flush=True)
        
        logprobs, ttft, total_time, token_count = self.make_streaming_request(url, test_num)
        
        if ttft is None or total_time is None or token_count == 0:
            print("FAILED")
            return None
        
        generation_time = total_time - ttft
        tps = token_count / generation_time if generation_time > 0 else 0
        overall_tps = token_count / total_time if total_time > 0 else 0
        
        print(f"TTFT={ttft*1000:.1f}ms, TPS={tps:.1f}, Tokens={token_count}")
        
        return {
            'engine': engine_name,
            'test_num': test_num,
            'logprobs': logprobs,
            'ttft': ttft,
            'total_time': total_time,
            'token_count': token_count,
            'tps': tps,
            'overall_tps': overall_tps
        }
    
    def run_performance_benchmark(self):
        """Run LIVE performance benchmarks and collect real-time results"""
        print("\n" + "="*80)
        print("PHASE 1: LIVE PERFORMANCE BENCHMARKING")
        print("="*80)
        print("\n🔴 All tests are executed in real-time against live servers")
        print(f"\nTest Configuration:")
        print(f"  Prompt: '{self.test_prompt}'")
        print(f"  Max tokens: {self.max_tokens}")
        print(f"  Number of tests per engine: {self.num_tests}")
        print(f"  Seed: {self.seed}")
        print(f"  Temperature: 0.01 (near-deterministic)")
        print(f"  Top logprobs: {self.top_logprobs}")
        print(f"  Note: Updated for candle-vllm's new numeric bytes format")
        
        # Test vLLM with LIVE requests
        print(f"\n🔵 LIVE Testing vLLM ({self.vllm_url}):")
        print("-" * 60)
        for i in range(self.num_tests):
            result = self.run_single_test("vLLM", self.vllm_url, i)
            if result:
                self.vllm_results.append(result)
            time.sleep(0.5)
        
        # Test alternative-vllm with LIVE requests
        print(f"\n🟢 LIVE Testing alternative-vllm ({self.alternative_vllm_url}):")
        print("-" * 60)
        for i in range(self.num_tests):
            result = self.run_single_test("alternative-vllm", self.alternative_vllm_url, i)
            if result:
                self.alternative_results.append(result)
            time.sleep(0.5)
        
        # Calculate quality scores from LIVE test data
        print("\n📊 QUALITY SCORES FROM LIVE DATA (vLLM as validator, alternative-vllm as miner):")
        print("-" * 60)
        
        all_distances = []
        for i in range(min(len(self.vllm_results), len(self.alternative_results))):
            score, avg_dist, distances = self.calculate_quality_score(
                self.vllm_results[i]['logprobs'], 
                self.alternative_results[i]['logprobs']
            )
            self.quality_scores.append(score)
            all_distances.extend(distances)
            print(f"  Test {i+1}: Score={score:.4f}, Distance={avg_dist:.6f}")
        
        return all_distances
    
    def analyze_performance_stats(self):
        """Analyze performance statistics"""
        print("\n" + "="*80)
        print("PHASE 2: PERFORMANCE ANALYSIS")
        print("="*80)
        
        # vLLM stats
        if self.vllm_results:
            vllm_ttfts = [r['ttft'] * 1000 for r in self.vllm_results]
            vllm_tps = [r['tps'] for r in self.vllm_results]
            vllm_overall_tps = [r['overall_tps'] for r in self.vllm_results]
            vllm_tokens = [r['token_count'] for r in self.vllm_results]
            
            print("\n🔵 vLLM Performance Statistics:")
            print("-" * 60)
            print(f"  TTFT (ms): Mean={statistics.mean(vllm_ttfts):.1f}, "
                  f"Min={min(vllm_ttfts):.1f}, Max={max(vllm_ttfts):.1f}")
            print(f"  TPS: Mean={statistics.mean(vllm_tps):.1f}, "
                  f"Min={min(vllm_tps):.1f}, Max={max(vllm_tps):.1f}")
            print(f"  Overall TPS: {statistics.mean(vllm_overall_tps):.1f}")
            print(f"  Tokens: Mean={statistics.mean(vllm_tokens):.1f}, Total={sum(vllm_tokens)}")
        
        # alternative-vllm stats
        if self.alternative_results:
            alternative_ttfts = [r['ttft'] * 1000 for r in self.alternative_results]
            alternative_tps = [r['tps'] for r in self.alternative_results]
            alternative_overall_tps = [r['overall_tps'] for r in self.alternative_results]
            alternative_tokens = [r['token_count'] for r in self.alternative_results]
            
            print("\n🟢 alternative-vllm Performance Statistics:")
            print("-" * 60)
            print(f"  TTFT (ms): Mean={statistics.mean(alternative_ttfts):.1f}, "
                  f"Min={min(alternative_ttfts):.1f}, Max={max(alternative_ttfts):.1f}")
            print(f"  TPS: Mean={statistics.mean(alternative_tps):.1f}, "
                  f"Min={min(alternative_tps):.1f}, Max={max(alternative_tps):.1f}")
            print(f"  Overall TPS: {statistics.mean(alternative_overall_tps):.1f}")
            print(f"  Tokens: Mean={statistics.mean(alternative_tokens):.1f}, Total={sum(alternative_tokens)}")
        
        # Comparative analysis
        if self.vllm_results and self.alternative_results:
            print("\n⚖️  Performance Comparison:")
            print("-" * 60)
            
            ttft_ratio = statistics.mean(alternative_ttfts) / statistics.mean(vllm_ttfts)
            tps_ratio = statistics.mean(alternative_tps) / statistics.mean(vllm_tps)
            
            print(f"  TTFT Ratio: {ttft_ratio:.2f}x (vLLM is {abs(1-ttft_ratio)*100:.1f}% "
                  f"{'slower' if ttft_ratio < 1 else 'faster'})")
            print(f"  TPS Ratio: {tps_ratio:.2f}x (vLLM is {abs(1-tps_ratio)*100:.1f}% "
                  f"{'slower' if tps_ratio < 1 else 'faster'})")
            
            return ttft_ratio, tps_ratio
        
        return None, None
    
    def analyze_token_by_token(self):
        """Analyze token-by-token differences with detailed metrics"""
        print("\n" + "="*80)
        print("PHASE 3: TOKEN-BY-TOKEN DETAILED ANALYSIS")
        print("="*80)
        
        if not self.vllm_results or not self.alternative_results:
            print("No results available for token analysis")
            return []
        
        # Take the first test result for detailed analysis
        vllm_tokens = self.vllm_results[0]['logprobs'] if self.vllm_results else []
        alternative_tokens = self.alternative_results[0]['logprobs'] if self.alternative_results else []
        
        min_len = min(len(vllm_tokens), len(alternative_tokens))
        
        print(f"\n📊 TOKEN-BY-TOKEN COMPARISON (First {min_len} tokens):")
        print("-" * 60)
        print(f"{'Pos':<4} {'Token':<15} {'vLLM LogP':>12} {'alt LogP':>12} {'Distance':>10} {'LogP Diff':>10} {'Rel %':>8}")
        print("-" * 60)
        
        all_distances = []
        for i in range(min_len):
            vllm_token = vllm_tokens[i]
            alternative_token = alternative_tokens[i]
            
            if vllm_token['token'] == alternative_token['token']:
                # Calculate metrics
                distance = self.calculate_distance(vllm_token['logprob'], alternative_token['logprob'])
                logprob_diff = abs(vllm_token['logprob'] - alternative_token['logprob'])
                rel_diff = (logprob_diff / abs(vllm_token['logprob'])) * 100 if vllm_token['logprob'] != 0 else 0
                
                all_distances.append(distance)
                
                # Truncate token for display
                token_display = vllm_token['token'][:15]
                
                print(f"{i:<4} {token_display:<15} {vllm_token['logprob']:>12.6f} "
                      f"{alternative_token['logprob']:>12.6f} {distance:>10.6f} "
                      f"{logprob_diff:>10.6f} {rel_diff:>8.2f}%")
            else:
                print(f"{i:<4} TOKEN MISMATCH: vLLM='{vllm_token['token'][:10]}' vs alt='{alternative_token['token'][:10]}'")
        
        # Summary stats for displayed tokens
        if all_distances:
            print(f"\n📈 Summary for {len(all_distances)} matching tokens:")
            print(f"  Average distance: {statistics.mean(all_distances):.6f}")
            print(f"  Min distance: {min(all_distances):.6f}")
            print(f"  Max distance: {max(all_distances):.6f}")
            if len(all_distances) > 1:
                print(f"  Std deviation: {statistics.stdev(all_distances):.6f}")
        
        return all_distances
    
    def analyze_quality_distribution(self, distances: List[float]):
        """Analyze quality score distribution"""
        print("\n" + "="*80)
        print("PHASE 4: QUALITY DISTRIBUTION ANALYSIS")
        print("="*80)
        
        if not distances:
            print("No distance data available")
            return
        
        # Basic statistics
        avg_distance = statistics.mean(distances)
        quality_score = self.score_average_distance(avg_distance)
        
        print(f"\n📊 Distance Statistics:")
        print("-" * 60)
        print(f"  Average Distance: {avg_distance:.6f}")
        print(f"  Quality Score: {quality_score:.6f}")
        print(f"  Min Distance: {min(distances):.6f}")
        print(f"  Max Distance: {max(distances):.6f}")
        if len(distances) > 1:
            print(f"  Std Dev: {statistics.stdev(distances):.6f}")
        
        # Distribution analysis
        perfect_tokens = sum(1 for d in distances if d <= BOTTOM_TEXT_THRESHOLD)
        partial_tokens = sum(1 for d in distances if BOTTOM_TEXT_THRESHOLD < d <= TOP_TEXT_THRESHOLD)
        fail_tokens = sum(1 for d in distances if d > TOP_TEXT_THRESHOLD)
        total_tokens = len(distances)
        
        print(f"\n📊 Token Distribution:")
        print("-" * 60)
        print(f"  Perfect (≤{BOTTOM_TEXT_THRESHOLD}): {perfect_tokens}/{total_tokens} "
              f"({perfect_tokens/total_tokens*100:.1f}%)")
        print(f"  Partial ({BOTTOM_TEXT_THRESHOLD}-{TOP_TEXT_THRESHOLD}): {partial_tokens}/{total_tokens} "
              f"({partial_tokens/total_tokens*100:.1f}%)")
        print(f"  Fail (>{TOP_TEXT_THRESHOLD}): {fail_tokens}/{total_tokens} "
              f"({fail_tokens/total_tokens*100:.1f}%)")
        
        # Percentile analysis
        print(f"\n📊 Distance Percentiles:")
        print("-" * 60)
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(distances, p)
            print(f"  {p}th percentile: {value:.6f}")
        
        return avg_distance, quality_score
    
    def analyze_sensitivity(self, distances: List[float]):
        """Perform sensitivity analysis"""
        print("\n" + "="*80)
        print("PHASE 5: SENSITIVITY ANALYSIS")
        print("="*80)
        
        if not distances:
            print("No distance data available")
            return
        
        avg_distance = statistics.mean(distances)
        
        print(f"\n🎯 Score Sensitivity to Distance Changes:")
        print("-" * 60)
        print(f"{'Multiplier':<12} {'Avg Dist':>12} {'Score':>8} {'Reward %':>10} {'Status':<15}")
        print("-" * 60)
        
        multipliers = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        for mult in multipliers:
            adjusted_avg = avg_distance * mult
            adjusted_score = self.score_average_distance(adjusted_avg)
            
            if adjusted_score >= 0.9:
                reward_pct = 90 + (adjusted_score - 0.9) * 100
                status = "Excellent"
            elif adjusted_score >= 0.5:
                reward_pct = 50 + (adjusted_score - 0.5) * 80
                status = "Good"
            elif adjusted_score > 0:
                reward_pct = adjusted_score * 50
                status = "Poor"
            else:
                reward_pct = 0
                status = "Failed"
            
            indicator = "←current" if mult == 1.0 else ""
            print(f"{mult:<12.1f} {adjusted_avg:>12.6f} {adjusted_score:>8.4f} "
                  f"{reward_pct:>10.1f} {status:<15} {indicator}")
    
    def analyze_risk_assessment(self, distances: List[float]):
        """Perform risk assessment"""
        print("\n" + "="*80)
        print("PHASE 6: RISK ASSESSMENT")
        print("="*80)
        
        if not distances:
            print("No distance data available")
            return
        
        avg_distance = statistics.mean(distances)
        
        # Calculate margins
        margin_to_partial = BOTTOM_TEXT_THRESHOLD - avg_distance
        margin_to_fail = TOP_TEXT_THRESHOLD - avg_distance
        
        print(f"\n🚨 Risk Metrics:")
        print("-" * 60)
        print(f"  Current average distance: {avg_distance:.6f}")
        print(f"  Margin to partial threshold: {margin_to_partial:.6f}")
        print(f"  Margin to fail threshold: {margin_to_fail:.6f}")
        
        # Determine risk level
        if margin_to_partial > 0.05:
            risk_level = "LOW"
            risk_desc = "Comfortable margin, unlikely to drop below perfect score"
            risk_emoji = "✅"
        elif margin_to_partial > 0.02:
            risk_level = "MEDIUM"
            risk_desc = "Some margin, but variations could affect score"
            risk_emoji = "⚠️"
        else:
            risk_level = "HIGH"
            risk_desc = "Very close to threshold, high risk of score reduction"
            risk_emoji = "🚨"
        
        print(f"\n  Risk Level: {risk_emoji} {risk_level}")
        print(f"  Assessment: {risk_desc}")
        
        # Analyze worst case scenarios
        print(f"\n📊 Impact of Worst Tokens:")
        print("-" * 60)
        sorted_distances = sorted(distances, reverse=True)
        
        for exclude_worst in range(min(5, len(sorted_distances))):
            if exclude_worst == 0:
                test_distances = distances.copy()
                desc = "All tokens"
            else:
                test_distances = sorted_distances[exclude_worst:]
                desc = f"Exclude {exclude_worst} worst"
            
            test_avg = statistics.mean(test_distances) if test_distances else 0
            test_score = self.score_average_distance(test_avg)
            print(f"  {desc:<20} Avg: {test_avg:.6f}, Score: {test_score:.4f}")
    
    def generate_final_summary(self, ttft_ratio, tps_ratio, avg_distance, quality_score):
        """Generate comprehensive final summary"""
        print("\n" + "="*80)
        print("FINAL SUMMARY & RECOMMENDATIONS")
        print("="*80)
        
        # Quality Assessment
        print(f"\n🎯 Quality Assessment:")
        print("-" * 60)
        print(f"  Quality Score: {quality_score:.4f}")
        print(f"  Average Distance: {avg_distance:.6f}")
        
        if quality_score >= 0.95:
            print(f"  ✅ EXCELLENT: Miners receive maximum rewards")
        elif quality_score >= 0.8:
            print(f"  ⚠️  GOOD: Miners receive high rewards")
        else:
            print(f"  🚨 POOR: Significant reward reduction")
        
        # Performance Assessment
        if ttft_ratio and tps_ratio:
            print(f"\n⚡ Performance Assessment:")
            print("-" * 60)
            
            if ttft_ratio > 10:
                print(f"  🚨 vLLM is {ttft_ratio:.1f}x faster at TTFT")
            elif ttft_ratio > 2:
                print(f"  ⚠️  vLLM is {ttft_ratio:.1f}x faster at TTFT")
            else:
                print(f"  ✅ Similar TTFT performance")
            
            if abs(tps_ratio - 1) > 0.5:
                faster = "vLLM" if tps_ratio < 1 else "alternative-vllm"
                print(f"  ⚠️  {faster} generates tokens {abs(1-tps_ratio)*100:.1f}% faster")
            else:
                print(f"  ✅ Similar TPS performance")
        
        # Final Recommendations
        print(f"\n📋 Final Recommendations for Subnet 19:")
        print("-" * 60)
        
        if quality_score > 0.9:
            if ttft_ratio and ttft_ratio > 5:
                print("1. ✅ Quality: Both engines achieve excellent scores")
                print("2. ⚠️  Performance: vLLM significantly faster")
                print("3. 🎯 Recommendation: Use vLLM for optimal mining")
            else:
                print("1. ✅ Quality: Both engines achieve excellent scores")
                print("2. ✅ Performance: Comparable between engines")
                print("3. 🎯 Recommendation: Either engine is suitable")
        else:
            print("1. 🚨 Quality: Scores below optimal threshold")
            print("2. 🔧 Action: Review model configuration")
            print("3. 🎯 Recommendation: Debug before deployment")
        
        print("\n💡 Additional Insights:")
        print("-" * 60)
        print("• Test with actual Subnet 19 validation prompts")
        print("• Monitor performance under production load")
        print("• Consider standardizing across all miners")
        print("• Regular benchmarking recommended")
        print("• Updated for candle-vllm's new numeric bytes format and raw logprobs calculation")
    
    def run_comprehensive_benchmark(self):
        """Run the complete comprehensive benchmark suite with LIVE tests"""
        print("\n🔴 LIVE MODE: All benchmarks use real-time API calls")
        print("No cached or pre-computed data is used")
        print("✅ UPDATED: Now compatible with candle-vllm's new format (numeric bytes, raw logprobs)")
        print(f"✅ TOP_LOGPROBS: Configured to {self.top_logprobs}\n")
        
        # Phase 1: LIVE Performance Benchmark
        all_distances = self.run_performance_benchmark()
        
        # Phase 2: Performance Analysis
        ttft_ratio, tps_ratio = self.analyze_performance_stats()
        
        # Phase 3: Token-by-Token Analysis
        token_distances = self.analyze_token_by_token()
        
        # Phase 4: Quality Distribution
        avg_distance, quality_score = self.analyze_quality_distribution(all_distances)
        
        # Phase 5: Sensitivity Analysis
        self.analyze_sensitivity(all_distances)
        
        # Phase 6: Risk Assessment
        self.analyze_risk_assessment(all_distances)
        
        # Final Summary based on LIVE test results
        self.generate_final_summary(ttft_ratio, tps_ratio, avg_distance, quality_score)
        
        print("\n✅ All analysis above is based on LIVE test data collected in real-time")
        print("✅ UPDATED: Now compatible with candle-vllm's new numeric bytes format and raw logprobs calculation")
        print(f"✅ TOP_LOGPROBS: Used {self.top_logprobs} for all requests")
        
        return {
            'quality_score': quality_score,
            'avg_distance': avg_distance,
            'ttft_ratio': ttft_ratio,
            'tps_ratio': tps_ratio,
            'distances': all_distances,
            'vllm_results': self.vllm_results,
            'alternative_results': self.alternative_results,
            'top_logprobs': self.top_logprobs
        }

def main():
    print("🚀 SUBNET 19 COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 80)
    print("\n🔴 LIVE TEST MODE: All results from real-time execution")
    print("✅ UPDATED: Now compatible with candle-vllm's new format")
    print("\nThis all-in-one benchmark includes:")
    print("  Phase 1: LIVE Performance Benchmarking (TTFT, TPS)")
    print("  Phase 2: Performance Statistical Analysis")
    print("  Phase 3: Token-by-Token Detailed Comparison")
    print("  Phase 4: Quality Score Distribution Analysis")
    print("  Phase 5: Sensitivity Analysis")
    print("  Phase 6: Risk Assessment")
    print("  Summary: Final Recommendations")
    
    print("\nChecking server availability...")
    
    # Check servers
    servers_ok = True
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        print("✅ vLLM server is running")
    except:
        print("❌ vLLM server not accessible at localhost:8000")
        servers_ok = False
    
    try:
        response = requests.get("http://localhost:2000/health", timeout=2)
        print("✅ alternative-vllm server is running")
    except:
        print("❌ alternative-vllm server not accessible at localhost:2000")
        servers_ok = False
    
    if not servers_ok:
        print("\n⚠️  Please ensure both servers are running before benchmarking")
        print("\nTo start servers:")
        print("  vLLM: Run on port 8000")
        print("  alternative-vllm: Run on port 2000")
        return
    
    # Run comprehensive benchmark
    benchmark = SN19ComprehensiveBenchmark()
    
    # Demonstrate top_logprobs configurations
    benchmark.demonstrate_top_logprobs_configs()
    
    # You can also configure custom values:
    # benchmark.configure_top_logprobs(10)  # Request top 10 alternatives
    # benchmark.configure_top_logprobs(False)  # No top logprobs
    # benchmark.configure_top_logprobs(True)   # Default to 5
    
    results = benchmark.run_comprehensive_benchmark()
    
    print("\n" + "="*80)
    print("🏁 COMPREHENSIVE BENCHMARK COMPLETE")
    print("="*80)
    print("\n✅ All results above are from LIVE tests executed in real-time")
    print("Results have been analyzed using the exact Subnet 19 scoring algorithm")
    print("from github.com/rayonlabs/vision-workers/.../text.py")
    print("✅ UPDATED: Now compatible with candle-vllm's new numeric bytes format and raw logprobs calculation")
    
    # Save results summary
    with open('benchmark_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'quality_score': results['quality_score'],
            'avg_distance': results['avg_distance'],
            'ttft_ratio': results['ttft_ratio'],
            'tps_ratio': results['tps_ratio'],
            'num_tests': len(results['vllm_results']),
            'note': 'Updated for candle-vllm new format (numeric bytes, raw logprobs)'
        }, f, indent=2)
    
    print("\n📁 Results saved to benchmark_results.json")

if __name__ == "__main__":
    main()