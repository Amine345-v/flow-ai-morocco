"""
Performance benchmarks for FlowLang.
"""
import time
import timeit
import statistics
from pathlib import Path

import pytest

from flowlang.parser import Parser
from flowlang.runtime import Runtime

# Number of repetitions for benchmarks
BENCHMARK_ITERATIONS = 100


def benchmark_flow_execution():
    """Benchmark the execution time of a simple flow."""
    # Setup
    parser = Parser()
    runtime = Runtime()
    
    # Register teams
    runtime.register_team("devs", "DEVELOPMENT")
    runtime.register_team("qa", "QUALITY_ASSURANCE")
    
    # Define a simple flow
    flow_src = """
    flow benchmark_flow(devs, qa) {
        checkpoint "develop" {
            devs: write_code()
            qa: review_code()
        }
        
        checkpoint "test" {
            devs: fix_bugs()
            qa: verify_fixes()
        }
        
        checkpoint "deploy" {
            devs: deploy()
            qa: smoke_test()
        }
    }
    """
    
    # Parse the flow
    flow = parser.parse(flow_src)
    
    # Time the execution
    start_time = time.perf_counter()
    runtime.execute_flow(flow, {"devs": "devs", "qa": "qa"})
    end_time = time.perf_counter()
    
    return end_time - start_time


def benchmark_chain_operations():
    """Benchmark chain touch operations."""
    runtime = Runtime()
    
    # Create a chain with many nodes
    nodes = [f"node_{i}" for i in range(1000)]
    runtime.create_chain("large_chain", nodes)
    
    # Time touching each node
    def touch_nodes():
        for node in nodes:
            runtime.touch_chain("large_chain", node)
    
    duration = timeit.timeit(touch_nodes, number=10) / 10  # Average over 10 runs
    return duration / len(nodes)  # Time per touch


def benchmark_process_operations():
    """Benchmark process operations."""
    runtime = Runtime()
    
    # Setup chain
    runtime.create_chain("deploy_flow", ["build", "test", "deploy"])
    
    # Create process
    runtime.create_process("deployment", "deploy_flow")
    
    # Time marking process status
    def mark_process():
        for i in range(1000):
            runtime.mark_process("deployment", f"status_{i % 5}", f"Test {i}")
    
    duration = timeit.timeit(mark_process, number=10) / 10  # Average over 10 runs
    return duration / 1000  # Time per mark operation


@pytest.mark.benchmark
class TestPerformance:
    """Performance test cases."""
    
    def test_flow_execution_performance(self, benchmark):
        """Test the performance of flow execution."""
        # Run benchmark multiple times and collect results
        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            start_time = time.perf_counter()
            benchmark_flow_execution()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Calculate statistics
        mean = statistics.mean(times)
        stddev = statistics.stdev(times) if len(times) > 1 else 0
        
        print(f"\nFlow Execution Performance (n={BENCHMARK_ITERATIONS}):")
        print(f"  Mean: {mean*1000:.2f}ms")
        print(f"  StdDev: {stddev*1000:.2f}ms")
        print(f"  Min: {min(times)*1000:.2f}ms")
        print(f"  Max: {max(times)*1000:.2f}ms")
        
        # Add assertion to fail if performance degrades significantly
        assert mean < 0.1, f"Flow execution too slow: {mean*1000:.2f}ms"
    
    def test_chain_operations_performance(self, benchmark):
        """Test the performance of chain operations."""
        # Run benchmark multiple times and collect results
        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            times.append(benchmark_chain_operations())
        
        # Convert to microseconds for better readability
        times_us = [t * 1_000_000 for t in times]
        
        # Calculate statistics
        mean = statistics.mean(times_us)
        stddev = statistics.stdev(times_us) if len(times_us) > 1 else 0
        
        print(f"\nChain Touch Performance (n={BENCHMARK_ITERATIONS}):")
        print(f"  Mean: {mean:.2f}μs per touch")
        print(f"  StdDev: {stddev:.2f}μs")
        print(f"  Min: {min(times_us):.2f}μs")
        print(f"  Max: {max(times_us):.2f}μs")
        
        # Add assertion to fail if performance degrades significantly
        assert mean < 1000, f"Chain touch operation too slow: {mean:.2f}μs"
    
    def test_process_operations_performance(self, benchmark):
        """Test the performance of process operations."""
        # Run benchmark multiple times and collect results
        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            times.append(benchmark_process_operations())
        
        # Convert to microseconds for better readability
        times_us = [t * 1_000_000 for t in times]
        
        # Calculate statistics
        mean = statistics.mean(times_us)
        stddev = statistics.stdev(times_us) if len(times_us) > 1 else 0
        
        print(f"\nProcess Mark Performance (n={BENCHMARK_ITERATIONS}):")
        print(f"  Mean: {mean:.2f}μs per mark")
        print(f"  StdDev: {stddev:.2f}μs")
        print(f"  Min: {min(times_us):.2f}μs")
        print(f"  Max: {max(times_us):.2f}μs")
        
        # Add assertion to fail if performance degrades significantly
        assert mean < 1000, f"Process mark operation too slow: {mean:.2f}μs"
