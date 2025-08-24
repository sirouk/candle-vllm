#!/usr/bin/env python3
"""
Performance Optimizer for candle-vllm
Optimizes TTFT and TPS performance by tuning key parameters
"""

import requests
import json
import time
import statistics
from typing import Dict, List, Tuple

class PerformanceOptimizer:
    def __init__(self, server_url: str = "http://localhost:2000"):
        self.server_url = server_url
        self.test_prompt = "Write a short story about a robot learning to paint."
        self.max_tokens = 50
        
    def test_performance(self, params: Dict) -> Tuple[float, float]:
        """Test performance with given parameters"""
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": self.test_prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.1,
            "stream": True,
            **params
        }
        
        start_time = time.time()
        first_token_time = None
        token_count = 0
        
        try:
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=30
            )
            
            for line in response.iter_lines():
                if line:
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    
                    token_count += 1
                    if token_count >= self.max_tokens:
                        break
            
            total_time = time.time() - start_time
            
            if first_token_time and total_time > 0:
                ttft = first_token_time * 1000  # Convert to ms
                tps = token_count / total_time
                return ttft, tps
                
        except Exception as e:
            print(f"Error testing performance: {e}")
            
        return float('inf'), 0.0
    
    def optimize_parameters(self) -> Dict:
        """Find optimal parameters for best performance"""
        print("🔧 Optimizing candle-vllm performance parameters...")
        
        # Test different configurations with more aggressive settings
        configs = [
            # Ultra-fast TTFT configurations
            {"block_size": 8, "holding_time": 25},
            {"block_size": 16, "holding_time": 25},
            {"block_size": 32, "holding_time": 25},
            
            # Aggressive TPS configurations
            {"max_num_seqs": 128, "holding_time": 25},
            {"max_num_seqs": 256, "holding_time": 25},
            {"max_num_seqs": 512, "holding_time": 25},
            
            # Memory optimization configurations
            {"prefill_chunk_size": 4096, "holding_time": 25},
            {"prefill_chunk_size": 8192, "holding_time": 25},
            {"prefill_chunk_size": 16384, "holding_time": 25},
            
            # Combined optimizations
            {"block_size": 8, "max_num_seqs": 256, "holding_time": 25},
            {"block_size": 16, "max_num_seqs": 512, "holding_time": 25},
            {"prefill_chunk_size": 8192, "max_num_seqs": 256, "holding_time": 25},
        ]
        
        best_config = {}
        best_score = float('inf')
        
        for config in configs:
            print(f"\nTesting config: {config}")
            ttft, tps = self.test_performance(config)
            
            # Score based on TTFT and TPS (lower is better)
            # Weight TTFT more heavily for better responsiveness
            score = (ttft / 1000) * 2.0 + (1.0 / max(tps, 0.1))  # Weighted score favoring TTFT
            
            print(f"  TTFT: {ttft:.1f}ms, TPS: {tps:.1f}, Score: {score:.3f}")
            
            if score < best_score:
                best_score = score
                best_config = config.copy()
        
        print(f"\n✅ Best configuration: {best_config}")
        print(f"   Score: {best_score:.3f}")
        
        return best_config
    
    def run_benchmark(self, config: Dict) -> Dict:
        """Run comprehensive benchmark with optimized config"""
        print(f"\n🚀 Running benchmark with optimized config: {config}")
        
        results = []
        for i in range(5):
            ttft, tps = self.test_performance(config)
            results.append({"ttft": ttft, "tps": tps})
            time.sleep(1)  # Brief pause between tests
        
        # Calculate statistics
        ttfts = [r["ttft"] for r in results]
        tpss = [r["tps"] for r in results]
        
        stats = {
            "config": config,
            "ttft_ms": {
                "mean": statistics.mean(ttfts),
                "min": min(ttfts),
                "max": max(ttfts),
                "std": statistics.stdev(ttfts) if len(ttfts) > 1 else 0
            },
            "tps": {
                "mean": statistics.mean(tpss),
                "min": min(tpss),
                "max": max(tpss),
                "std": statistics.stdev(tpss) if len(tpss) > 1 else 0
            }
        }
        
        print(f"\n📊 Benchmark Results:")
        print(f"  TTFT: {stats['ttft_ms']['mean']:.1f}ms ± {stats['ttft_ms']['std']:.1f}ms")
        print(f"  TPS: {stats['tps']['mean']:.1f} ± {stats['tps']['std']:.1f}")
        
        return stats
    
    def generate_recommendations(self, stats: Dict) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        ttft_mean = stats["ttft_ms"]["mean"]
        tps_mean = stats["tps"]["mean"]
        
        if ttft_mean > 1000:
            recommendations.append("🚨 TTFT is very high (>1s). Consider reducing model size or using quantization.")
        elif ttft_mean > 500:
            recommendations.append("⚠️  TTFT is high (>500ms). Try reducing block_size and holding_time.")
        elif ttft_mean < 100:
            recommendations.append("✅ TTFT is excellent (<100ms). Great performance!")
        
        if tps_mean < 10:
            recommendations.append("🚨 TPS is very low (<10). Check GPU utilization and memory.")
        elif tps_mean < 50:
            recommendations.append("⚠️  TPS could be improved. Consider increasing max_num_seqs.")
        elif tps_mean > 100:
            recommendations.append("✅ TPS is excellent (>100). Great throughput!")
        
        # Specific recommendations based on config
        config = stats["config"]
        if "block_size" in config and config["block_size"] > 32:
            recommendations.append("💡 Consider reducing block_size for faster TTFT.")
        
        if "holding_time" in config and config["holding_time"] > 100:
            recommendations.append("💡 Consider reducing holding_time for faster response.")
        
        if "max_num_seqs" in config and config["max_num_seqs"] < 128:
            recommendations.append("💡 Consider increasing max_num_seqs for better TPS.")
        
        return recommendations

def main():
    optimizer = PerformanceOptimizer()
    
    try:
        # Test server connectivity
        response = requests.get(f"{optimizer.server_url}/v1/health")
        if response.status_code != 200:
            print(f"❌ Server not accessible at {optimizer.server_url}")
            return
        
        print("✅ Server is accessible")
        
        # Optimize parameters
        best_config = optimizer.optimize_parameters()
        
        # Run benchmark with best config
        stats = optimizer.run_benchmark(best_config)
        
        # Generate recommendations
        recommendations = optimizer.generate_recommendations(stats)
        
        print(f"\n💡 Performance Recommendations:")
        for rec in recommendations:
            print(f"  {rec}")
        
        # Save results
        with open("optimization_results.json", "w") as f:
            json.dump({
                "timestamp": time.time(),
                "best_config": best_config,
                "benchmark_stats": stats,
                "recommendations": recommendations
            }, f, indent=2)
        
        print(f"\n💾 Results saved to optimization_results.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
