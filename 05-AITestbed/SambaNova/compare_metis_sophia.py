"""
SambaNova Homework: Compare Metis vs Sophia Performance
=======================================================

This script compares the performance of GPT-OSS model on:
1. Metis (SambaNova SN40L - Inference-optimized hardware)
2. Sophia (ALCF GPU cluster - General-purpose GPUs)

Dataset: HuggingFace 'squad' dataset (Stanford Question Answering Dataset)
Metrics: Latency, throughput, response quality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from inference_auth_token import get_access_token
import time
import json
from datetime import datetime
from typing import List, Dict, Tuple


def load_huggingface_dataset(num_samples: int = 15) -> List[Dict]:
    """
    Load prompts from HuggingFace SQuAD dataset.
    
    Dataset: SQuAD v1.1 (Stanford Question Answering Dataset)
    - Contains questions based on Wikipedia articles
    - Ideal for testing question-answering capabilities
    
    Parameters
    ----------
    num_samples : int
        Number of samples to load from the dataset
    
    Returns
    -------
    List[Dict]
        List of dictionaries with 'question', 'context', and 'prompt' keys
    """
    try:
        from datasets import load_dataset
        
        print("📚 Loading HuggingFace SQuAD dataset...")
        # Load SQuAD v1 validation set
        dataset = load_dataset("squad", split="validation")
        
        # Select a subset of questions
        samples = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            
            # Format the prompt: include context for better answers
            prompt = f"Context: {item['context']}\n\nQuestion: {item['question']}\n\nAnswer:"
            
            samples.append({
                "id": item["id"],
                "question": item["question"],
                "context": item["context"][:200] + "...",  # Truncate for display
                "prompt": prompt,
                "reference_answer": item["answers"]["text"][0] if item["answers"]["text"] else "N/A"
            })
        
        print(f"✓ Loaded {len(samples)} samples from SQuAD dataset")
        return samples
    
    except ImportError:
        print("⚠️  'datasets' package not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        # Retry after installation
        return load_huggingface_dataset(num_samples)
    except Exception as e:
        print(f"❌ Error loading HuggingFace dataset: {e}")
        print("   Falling back to simple questions...")
        # Fallback to simple prompts if dataset loading fails
        return [
            {
                "id": f"fallback_{i}",
                "question": q,
                "context": "N/A",
                "prompt": q,
                "reference_answer": "N/A"
            }
            for i, q in enumerate([
                "What is quantum computing?",
                "Explain machine learning in simple terms.",
                "What causes climate change?",
                "How does photosynthesis work?",
                "What is DNA?"
            ])
        ]


def get_client(endpoint: str) -> OpenAI:
    """
    Create OpenAI client for specified endpoint.
    
    Parameters
    ----------
    endpoint : str
        Either 'metis' or 'sophia'
    
    Returns
    -------
    OpenAI
        Configured OpenAI client
    """
    access_token = get_access_token()
    
    if endpoint == 'metis':
        base_url = "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1"
    elif endpoint == 'sophia':
        base_url = "https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1"
    else:
        raise ValueError(f"Unknown endpoint: {endpoint}")
    
    return OpenAI(api_key=access_token, base_url=base_url)


def run_single_inference(
    client: OpenAI,
    prompt: str,
    model: str = "gpt-oss-120b-131072"
) -> Tuple[str, float, Dict]:
    """
    Run a single inference and measure latency.
    
    Parameters
    ----------
    client : OpenAI
        OpenAI client configured for endpoint
    prompt : str
        Input prompt
    model : str
        Model name
    
    Returns
    -------
    Tuple[str, float, Dict]
        (response_text, latency_seconds, metadata)
    """
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512
        )
        
        end_time = time.time()
        latency = end_time - start_time
        
        response_text = response.choices[0].message.content
        
        # Extract metadata
        metadata = {
            "model": response.model if hasattr(response, 'model') else model,
            "finish_reason": response.choices[0].finish_reason,
            "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') else None,
            "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') else None,
            "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else None,
        }
        
        return response_text, latency, metadata
    
    except Exception as e:
        error_msg = str(e)
        print(f"Error during inference: {error_msg}")
        import traceback
        traceback.print_exc()
        return None, None, {"error": error_msg}


def benchmark_endpoint(
    endpoint_name: str,
    prompts: List[str],
    model: str,
    num_runs: int = 1
) -> List[Dict]:
    """
    Benchmark a specific endpoint with given prompts.
    
    Parameters
    ----------
    endpoint_name : str
        'metis' or 'sophia'
    prompts : List[str]
        List of prompts to test
    model : str
        Model name
    num_runs : int
        Number of times to repeat each prompt
    
    Returns
    -------
    Dict
        Benchmark results with statistics
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking {endpoint_name.upper()}")
    print(f"{'='*70}")
    
    client = get_client(endpoint_name)
    results = []
    
    for idx, prompt in enumerate(prompts, 1):
        print(f"\n[{idx}/{len(prompts)}] Testing prompt: {prompt[:60]}...")
        
        for run in range(num_runs):
            response_text, latency, metadata = run_single_inference(
                client, prompt, model
            )
            
            if response_text is not None:
                result = {
                    "endpoint": endpoint_name,
                    "prompt_id": idx,
                    "prompt": prompt,
                    "run": run + 1,
                    "response": response_text,
                    "latency_sec": latency,
                    "metadata": metadata
                }
                results.append(result)
                
                print(f"  Run {run+1}: {latency:.3f}s, "
                      f"{metadata.get('total_tokens', 'N/A')} tokens, "
                      f"Response length: {len(response_text)} chars")
            else:
                print(f"  Run {run+1}: FAILED - {metadata.get('error', 'Unknown error')}")
    
    return results


def calculate_statistics(results: List[Dict]) -> Dict:
    """Calculate aggregate statistics from results."""
    if not results:
        return {
            "num_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "latency": {"mean": 0, "min": 0, "max": 0, "median": 0},
            "tokens": {"mean": 0, "total": 0},
            "response_length": {"mean": 0},
            "throughput_tokens_per_sec": 0
        }
    
    latencies = [r['latency_sec'] for r in results if r.get('latency_sec') is not None]
    total_tokens = [r['metadata'].get('total_tokens', 0) for r in results 
                   if r.get('metadata', {}).get('total_tokens')]
    response_lengths = [len(r['response']) for r in results if r.get('response')]
    
    stats = {
        "num_requests": len(results),
        "successful_requests": len([r for r in results if r.get('latency_sec') is not None]),
        "failed_requests": len([r for r in results if r.get('latency_sec') is None]),
        "latency": {
            "mean": sum(latencies) / len(latencies) if latencies else 0,
            "min": min(latencies) if latencies else 0,
            "max": max(latencies) if latencies else 0,
            "median": sorted(latencies)[len(latencies)//2] if latencies else 0,
        },
        "tokens": {
            "mean": sum(total_tokens) / len(total_tokens) if total_tokens else 0,
            "total": sum(total_tokens) if total_tokens else 0,
        },
        "response_length": {
            "mean": sum(response_lengths) / len(response_lengths) if response_lengths else 0,
        },
        "throughput_tokens_per_sec": (
            sum(total_tokens) / sum(latencies) if latencies and sum(latencies) > 0 else 0
        )
    }
    
    return stats


def print_comparison_report(metis_stats: Dict, sophia_stats: Dict):
    """Print a formatted comparison report."""
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON REPORT")
    print("="*70)
    
    print("\n📊 LATENCY COMPARISON (seconds)")
    print(f"{'Metric':<20} {'Metis (SambaNova)':<25} {'Sophia (GPU)':<25}")
    print("-" * 70)
    print(f"{'Mean Latency':<20} {metis_stats['latency']['mean']:>24.3f} {sophia_stats['latency']['mean']:>24.3f}")
    print(f"{'Min Latency':<20} {metis_stats['latency']['min']:>24.3f} {sophia_stats['latency']['min']:>24.3f}")
    print(f"{'Max Latency':<20} {metis_stats['latency']['max']:>24.3f} {sophia_stats['latency']['max']:>24.3f}")
    print(f"{'Median Latency':<20} {metis_stats['latency']['median']:>24.3f} {sophia_stats['latency']['median']:>24.3f}")
    
    print("\n🚀 THROUGHPUT COMPARISON")
    print(f"{'Metric':<20} {'Metis (SambaNova)':<25} {'Sophia (GPU)':<25}")
    print("-" * 70)
    print(f"{'Tokens/sec':<20} {metis_stats['throughput_tokens_per_sec']:>24.2f} {sophia_stats['throughput_tokens_per_sec']:>24.2f}")
    print(f"{'Avg tokens/req':<20} {metis_stats['tokens']['mean']:>24.1f} {sophia_stats['tokens']['mean']:>24.1f}")
    
    print("\n✅ SUCCESS RATE")
    print(f"{'Metric':<20} {'Metis (SambaNova)':<25} {'Sophia (GPU)':<25}")
    print("-" * 70)
    print(f"{'Successful':<20} {metis_stats['successful_requests']:>24} {sophia_stats['successful_requests']:>24}")
    print(f"{'Failed':<20} {metis_stats['failed_requests']:>24} {sophia_stats['failed_requests']:>24}")
    
    # Calculate speedup
    if sophia_stats['latency']['mean'] > 0:
        speedup = sophia_stats['latency']['mean'] / metis_stats['latency']['mean']
        print(f"\n⚡ SPEEDUP: Metis is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than Sophia")
    
    print("\n" + "="*70)


def save_results(metis_results: List[Dict], sophia_results: List[Dict], 
                metis_stats: Dict, sophia_stats: Dict, output_file: str, 
                dataset_info: Dict = None):
    """Save detailed results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_info or {"name": "Unknown"},
        "models": {
            "metis": "gpt-oss-120b-131072",
            "sophia": "openai/gpt-oss-120b"
        },
        "metis": {
            "results": metis_results,
            "statistics": metis_stats
        },
        "sophia": {
            "results": sophia_results,
            "statistics": sophia_stats
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")


def main():
    """Main execution function."""
    print("="*70)
    print("SAMBANOVA HOMEWORK: Metis vs Sophia Performance Comparison")
    print("="*70)
    print("\n📊 Dataset: HuggingFace SQuAD (Stanford Question Answering Dataset)")
    print("   URL: https://huggingface.co/datasets/squad")
    print("   Description: Questions based on Wikipedia articles")
    print("\n" + "="*70)
    
    # Load HuggingFace dataset
    dataset_samples = load_huggingface_dataset(num_samples=10)
    test_prompts = [sample["prompt"] for sample in dataset_samples]
    
    print(f"\n✓ Loaded {len(test_prompts)} prompts from SQuAD dataset")
    
    # Model names - Metis uses different name than Sophia
    metis_model = "gpt-oss-120b-131072"
    sophia_model = "openai/gpt-oss-120b"
    
    print(f"\n🤖 Models:")
    print(f"   Metis:  {metis_model}")
    print(f"   Sophia: {sophia_model}")
    
    # Benchmark both endpoints
    print("\n" + "="*70)
    print("🔬 Starting benchmarks...")
    
    try:
        # Benchmark Metis (SambaNova)
        metis_results = benchmark_endpoint("metis", test_prompts, metis_model, num_runs=1)
        metis_stats = calculate_statistics(metis_results)
        
        # Add small delay between endpoints
        print("\n⏳ Waiting 5 seconds before testing Sophia...")
        time.sleep(5)
        
        # Benchmark Sophia (GPU cluster)
        sophia_results = benchmark_endpoint("sophia", test_prompts, sophia_model, num_runs=1)
        sophia_stats = calculate_statistics(sophia_results)
        
        # Print comparison
        print_comparison_report(metis_stats, sophia_stats)
        
        # Save results
        output_file = f"metis_vs_sophia_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(metis_results, sophia_results, metis_stats, sophia_stats, output_file, dataset_info={
            "name": "SQuAD",
            "version": "1.1",
            "source": "HuggingFace",
            "url": "https://huggingface.co/datasets/squad",
            "num_samples": len(dataset_samples),
            "description": "Stanford Question Answering Dataset - questions based on Wikipedia articles"
        })
        
        print("\n✅ Benchmark completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
