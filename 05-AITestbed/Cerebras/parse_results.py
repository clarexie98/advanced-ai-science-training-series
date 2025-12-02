#!/usr/bin/env python3
"""Parse Cerebras training logs to extract performance metrics"""

import re
import sys
from pathlib import Path

def parse_log_file(log_path):
    """Extract key metrics from log file"""
    results = {
        'batch_size': None,
        'samples_per_sec': [],
        'global_rate': None,
        'total_time': None,
        'total_samples': None,
        'final_loss': None
    }
    
    with open(log_path, 'r') as f:
        content = f.read()
        
        # Extract batch size from filename
        match = re.search(r'log_bs(\d+)\.txt', str(log_path))
        if match:
            results['batch_size'] = int(match.group(1))
        
        # Extract performance metrics from training steps
        rate_pattern = r'Rate=([0-9.]+) samples/sec, GlobalRate=([0-9.]+) samples/sec'
        for match in re.finditer(rate_pattern, content):
            results['samples_per_sec'].append(float(match.group(1)))
            results['global_rate'] = float(match.group(2))
        
        # Extract final training summary
        summary_match = re.search(r'Processed (\d+) training sample\(s\) in ([0-9.]+) seconds', content)
        if summary_match:
            results['total_samples'] = int(summary_match.group(1))
            results['total_time'] = float(summary_match.group(2))
        
        # Extract final loss
        loss_matches = re.findall(r'Step=200, Loss=([0-9.]+)', content)
        if loss_matches:
            results['final_loss'] = float(loss_matches[-1])
    
    return results

def add_baseline_result():
    """Add the already-completed batch_size=1024 result"""
    return {
        'batch_size': 1024,
        'samples_per_sec': [33.05, 33.03, 33.08, 33.06],  # From your actual run
        'global_rate': 33.04,
        'total_time': 7393.91,
        'total_samples': 204800,
        'final_loss': 6.10848
    }

def main():
    homework_dir = Path('homework_batch_comparison')
    
    print("=" * 95)
    print("CEREBRAS LLAMA-7B BATCH SIZE COMPARISON")
    print("=" * 95)
    print()
    
    all_results = []
    
    # Add the baseline (already completed batch_size=1024)
    all_results.append(add_baseline_result())
    print("✓ Added baseline result: batch_size=1024 (from previous run)")
    
    # Check if homework directory exists
    if homework_dir.exists():
        log_files = sorted(homework_dir.glob('log_bs*.txt'))
        
        if log_files:
            print(f"✓ Found {len(log_files)} additional log file(s)")
            for log_file in log_files:
                results = parse_log_file(log_file)
                if results['batch_size']:
                    all_results.append(results)
        else:
            print("⚠ No new log files found yet in homework_batch_comparison/")
    else:
        print("⚠ homework_batch_comparison/ directory not found")
        print("  Only showing baseline result (batch_size=1024)")
    
    print()
    
    # Sort by batch size
    all_results.sort(key=lambda x: x['batch_size'])
    
    # Print table header
    print(f"{'Batch Size':<12} {'Avg Rate':<15} {'Global Rate':<15} {'Total Time':<15} {'Throughput':<15} {'Final Loss':<12}")
    print(f"{'(samples)':<12} {'(samples/sec)':<15} {'(samples/sec)':<15} {'(seconds)':<15} {'(samples/sec)':<15} {'(value)':<12}")
    print("-" * 95)
    
    for r in all_results:
        avg_rate = sum(r['samples_per_sec']) / len(r['samples_per_sec']) if r['samples_per_sec'] else 0
        global_rate = r['global_rate'] if r['global_rate'] else 0
        total_time = r['total_time'] if r['total_time'] else 0
        throughput = r['total_samples'] / r['total_time'] if r['total_time'] else 0
        final_loss = r['final_loss'] if r['final_loss'] else 0
        
        print(f"{r['batch_size']:<12} {avg_rate:<15.2f} {global_rate:<15.2f} {total_time:<15.2f} {throughput:<15.2f} {final_loss:<12.5f}")
    
    print()
    print("=" * 95)
    print("Key Metrics:")
    print("  - Avg Rate: Average samples/sec across training steps")
    print("  - Global Rate: Final global rate reported")
    print("  - Total Time: Total training time in seconds")
    print("  - Throughput: Total samples / Total time")
    print()
    print("Batch sizes completed: " + ", ".join(str(r['batch_size']) for r in all_results))
    print("Batch sizes remaining: " + ", ".join(str(bs) for bs in [256, 512, 2048, 4096] 
                                                   if bs not in [r['batch_size'] for r in all_results]))
    print("=" * 95)

if __name__ == '__main__':
    main()
