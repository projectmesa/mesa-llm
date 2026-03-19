#!/usr/bin/env python3
"""
Mesa-LLM Performance Benchmark Test

This script tests and showcases the performance optimizations implemented
in mesa-llm, comparing sequential vs parallel execution with simulated LLM work.

Usage:
    python tests/test_performance_benchmark.py

Results:
    - Shows speedup from parallel execution
    - Demonstrates linear scaling
    - Validates performance optimizations

Performance test models for mesa-llm benchmarks

"""

import asyncio
import time

from mesa import Agent, Model
from mesa.space import MultiGrid

from mesa_llm.benchmark import PerformanceBenchmark
from mesa_llm.parallel_stepping import (
    enable_automatic_parallel_stepping,
    step_agents_parallel,
)


class PerformanceTestAgent(Agent):
    """Mock agent that simulates LLM work for performance testing"""

    def __init__(self, model, agent_id):
        super().__init__(model)
        self.agent_id = agent_id
        self.steps_taken = 0
        self.async_steps_taken = 0

    async def astep(self):
        """Async step that simulates LLM API call"""
        await asyncio.sleep(0.01)  # Simulate 10ms API response time
        self.async_steps_taken += 1

    def step(self):
        """Sync step that simulates LLM API call"""
        time.sleep(0.01)  # Simulate 10ms API response time
        self.steps_taken += 1


class PerformanceTestModel(Model):
    """Model for performance testing with configurable agent counts"""

    def __init__(self, n_agents: int = 10, enable_parallel: bool = True):
        super().__init__()
        self.n_agents = n_agents
        self.grid = MultiGrid(20, 20, torus=False)
        self.custom_agents = []

        # Create agents with simulated LLM work
        for i in range(n_agents):
            agent = PerformanceTestAgent(self, i)
            x = i % 20
            y = (i // 20) % 20
            self.grid.place_agent(agent, (x, y))
            self.custom_agents.append(agent)

        # Enable parallel stepping if requested
        if enable_parallel:
            enable_automatic_parallel_stepping(mode="asyncio", max_concurrent=10)

    def step_sequential(self):
        """Execute agents sequentially"""
        for agent in self.custom_agents:
            agent.step()

    def step_parallel(self):
        """Execute agents in parallel"""
        asyncio.run(step_agents_parallel(self.custom_agents))


def main():
    """Run performance benchmark"""
    print(" Mesa-LLM Performance Optimization Showcase")
    print("=" * 60)
    print("This benchmark demonstrates the performance improvements")
    print("implemented in mesa-llm for large-scale agent simulations.")

    benchmark = PerformanceBenchmark()

    # Run benchmark with increasing agent counts
    agent_counts = [5, 10, 15, 20, 25, 30, 40, 50]

    benchmark.run_benchmark(
        agent_counts=agent_counts, test_model_class=PerformanceTestModel
    )

    # Print comprehensive analysis
    benchmark.print_summary()

    # Save results for documentation
    benchmark.save_results()

    print("\n✅ Performance Benchmark Completed!")
    print("🚀 Mesa-LLM is ready for large-scale agent simulations!")


if __name__ == "__main__":
    main()
