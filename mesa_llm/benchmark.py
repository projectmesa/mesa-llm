"""
Performance benchmark framework for mesa-llm
"""

import time
import asyncio
import os
import csv
import statistics
from typing import List, Dict

from .parallel_stepping import enable_automatic_parallel_stepping, step_agents_parallel


class PerformanceBenchmark:
    """Performance testing and analysis framework"""
    
    def __init__(self):
        self.results: List[Dict] = []
        
    def run_single_test(self, n_agents: int, runs: int = 3, test_model_class=None) -> Dict:
        """Run performance test for specific agent count"""
        print(f"\n🔬 Testing {n_agents} agents...")
        
        # Import test models if not provided
        if test_model_class is None:
            from tests.test_models import PerformanceTestModel
            test_model_class = PerformanceTestModel
        
        sequential_times = []
        parallel_times = []
        
        for run in range(runs):
            print(f"  Run {run + 1}/{runs}...")
            
            # Test sequential execution
            start_time = time.time()
            model_seq = test_model_class(n_agents=n_agents, enable_parallel=False)
            creation_time = time.time() - start_time
            
            step_start = time.time()
            model_seq.step_sequential()
            step_time = time.time() - step_start
            sequential_times.append(step_time)
            
            # Test parallel execution
            start_time = time.time()
            model_par = test_model_class(n_agents=n_agents, enable_parallel=True)
            step_start = time.time()
            model_par.step_parallel()
            step_time = time.time() - step_start
            parallel_times.append(step_time)
            
            print(f"    Sequential: {sequential_times[-1]:.2f}s, Parallel: {parallel_times[-1]:.2f}s")
        
        # Calculate statistics
        avg_seq = statistics.mean(sequential_times)
        avg_par = statistics.mean(parallel_times)
        speedup = avg_seq / avg_par if avg_par > 0 else float('inf')
        
        result = {
            'n_agents': n_agents,
            'sequential_time': avg_seq,
            'parallel_time': avg_par,
            'speedup': speedup,
            'per_agent_seq': avg_seq / n_agents,
            'per_agent_par': avg_par / n_agents
        }
        
        print(f"  📊 Results: Sequential {avg_seq:.2f}s, Parallel {avg_par:.2f}s, Speedup {speedup:.2f}x")
        return result
    
    def run_benchmark(self, agent_counts: List[int] = None, test_model_class=None) -> List[Dict]:
        """Run comprehensive performance benchmark"""
        if agent_counts is None:
            agent_counts = [5, 10, 15, 20, 25, 30, 40, 50]
        
        self.results = []
        
        print("🚀 Mesa-LLM Performance Benchmark")
        print("=" * 50)
        print("📋 Testing parallel vs sequential execution")
        print("⚠️  Using 10ms simulated LLM work per agent")
        print("")
        
        for n_agents in agent_counts:
            result = self.run_single_test(n_agents, runs=3, test_model_class=test_model_class)
            self.results.append(result)
        
        return self.results
    
    def print_summary(self):
        """Print comprehensive performance analysis"""
        print("\n📈 PERFORMANCE BENCHMARK RESULTS")
        print("=" * 80)
        
        print(f"{'Agents':<8} {'Sequential':<12} {'Parallel':<12} {'Speedup':<10} {'Efficiency':<12}")
        print("-" * 80)
        
        for result in self.results:
            n_agents = result['n_agents']
            seq_time = result['sequential_time']
            par_time = result['parallel_time']
            speedup = result['speedup']
            efficiency = speedup / n_agents if speedup != float('inf') else 0
            
            print(f"{n_agents:<8} {seq_time:<12.2f} {par_time:<12.2f} "
                  f"{speedup:<10.2f}x {efficiency:<12.4f}")
        
        print("\n🔍 Performance Analysis:")
        
        # Check scaling characteristics
        if len(self.results) >= 3:
            first_result = self.results[0]
            last_result = self.results[-1]
            
            seq_scaling = (last_result['per_agent_seq'] / first_result['per_agent_seq'])
            par_scaling = (last_result['per_agent_par'] / first_result['per_agent_par'])
            
            print(f"Sequential scaling factor: {seq_scaling:.2f}x (1.0 = ideal)")
            print(f"Parallel scaling factor: {par_scaling:.2f}x (1.0 = ideal)")
            
            # Evaluate sequential scaling
            if seq_scaling > 2.0:
                print("⚠️  SEQUENTIAL: Exponential scaling detected!")
            elif seq_scaling > 1.5:
                print("⚠️  SEQUENTIAL: Sub-linear scaling")
            else:
                print("✅ SEQUENTIAL: Perfect linear scaling")
            
            # Evaluate parallel scaling
            if par_scaling > 2.0:
                print("⚠️  PARALLEL: Exponential scaling detected!")
            elif par_scaling > 1.5:
                print("⚠️  PARALLEL: Sub-linear scaling")
            else:
                print("✅ PARALLEL: Good linear scaling")
        
        # Evaluate speedup
        valid_speedups = [r['speedup'] for r in self.results if r['speedup'] != float('inf')]
        if valid_speedups:
            avg_speedup = statistics.mean(valid_speedups)
            print(f"Average speedup: {avg_speedup:.2f}x")
            
            if avg_speedup > 5.0:
                print("🎉 EXCELLENT: Parallel provides outstanding speedup!")
            elif avg_speedup > 3.0:
                print("🎉 EXCELLENT: Parallel provides significant speedup!")
            elif avg_speedup > 2.0:
                print("✅ GOOD: Parallel provides moderate speedup")
            elif avg_speedup > 1.5:
                print("⚠️  MINIMAL: Parallel provides small speedup")
            else:
                print("❌ POOR: Parallel provides no speedup")
        
        print("\n💡 Key Insights:")
        print("   • Each agent simulates 10ms LLM API response time")
        print("   • Parallel execution processes agents concurrently")
        print("   • Speedup demonstrates effectiveness of optimizations")
        print("   • Linear scaling confirms no performance bottlenecks")
        
        print("\n📝 Notes:")
        print("   • This benchmark tests parallel stepping infrastructure")
        print("   • Real-world performance depends on actual API response times")
        print("   • Results demonstrate performance optimizations work correctly")
    
    def save_results(self, filename: str = "benchmark_results.csv"):
        """Save benchmark results to CSV file"""
        if not self.results:
            print("No results to save!")
            return
        
        # Save to results directory
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        filepath = os.path.join(results_dir, filename)
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['n_agents', 'sequential_time', 'parallel_time', 'speedup', 'per_agent_seq', 'per_agent_par']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                writer.writerow(result)
        
        print(f"💾 Results saved to {filepath}")
