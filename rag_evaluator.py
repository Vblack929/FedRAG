import os
import json
import time
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm
import numpy as np
from dataset_handler import NQDataset
from src.rag_system import RAG


class RAGEvaluator:
    """Class for evaluating RAG systems."""
    
    def __init__(self, rag_system: RAG):
        """Initialize the evaluator.
        
        Args:
            rag_system: RAG system to evaluate
        """
        self.rag_system = rag_system
        
    def evaluate_dataset(self, 
                         eval_data: List[Dict],
                         output_file: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate the RAG system on a dataset.
        
        Args:
            eval_data: List of question-answer pairs
            output_file: Path to save the evaluation results
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = []
        latencies = []
        
        print(f"Evaluating RAG system on {len(eval_data)} questions...")
        for i, item in enumerate(tqdm(eval_data)):
            question = item["question"]
            ground_truth = item["answers"]
            
            # Measure query time
            start_time = time.time()
            answer = self.rag_system.query(question)
            end_time = time.time()
            
            # Calculate latency
            latency = end_time - start_time
            latencies.append(latency)
            
            # Store result
            result = {
                "question": question,
                "ground_truth": ground_truth,
                "predicted": answer,
                "latency": latency
            }
            results.append(result)
        
        # Calculate metrics
        avg_latency = np.mean(latencies)
        metrics = {
            "num_questions": len(eval_data),
            "avg_latency": avg_latency,
            "results": results
        }
        
        # Save results if output file is provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Evaluation results saved to {output_file}")
        
        return metrics
    
    def evaluate_nq_dataset(self, 
                            num_samples: int = 50, 
                            output_file: str = "data/evaluation_results.json") -> Dict[str, Any]:
        """Evaluate the RAG system on the NQ dataset.
        
        Args:
            num_samples: Number of samples to evaluate on
            output_file: Path to save the evaluation results
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Load NQ dataset
        nq_dataset = NQDataset()
        
        # Create evaluation set
        eval_file = nq_dataset.create_evaluation_set(num_samples=num_samples)
        
        # Load evaluation data
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
        
        # Run evaluation
        return self.evaluate_dataset(eval_data, output_file)
    
    def print_evaluation_summary(self, metrics: Dict[str, Any]) -> None:
        """Print a summary of the evaluation results.
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        print("\n===== RAG Evaluation Summary =====")
        print(f"Number of questions: {metrics['num_questions']}")
        print(f"Average latency: {metrics['avg_latency']:.2f} seconds")
        
        # Print a few example results
        print("\nExample results:")
        for i, result in enumerate(metrics['results'][:3]):
            print(f"\nQuestion {i+1}: {result['question']}")
            print(f"Ground truth: {result['ground_truth']}")
            print(f"Predicted: {result['predicted']}")
            print(f"Latency: {result['latency']:.2f} seconds")
        
        print("\n=================================")


def main():
    """Example usage of the RAG evaluator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a RAG system on the NQ dataset")
    parser.add_argument("--model-type", type=str, default="openai", choices=["openai", "llama"], 
                        help="Type of model to use (openai or llama)")
    parser.add_argument("--llama-model-path", type=str, default=None,
                        help="Path to Llama model (required if model-type is llama)")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of samples to evaluate on")
    parser.add_argument("--output-file", type=str, default="data/evaluation_results.json",
                        help="Path to save the evaluation results")
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG system
        if args.model_type == "llama" and not args.llama_model_path:
            raise ValueError("llama_model_path must be provided when using Llama model")
        
        rag = RAG(
            model_type=args.model_type,
            llama_model_path=args.llama_model_path
        )
        
        # Load documents (expects NQ dataset to be in data/processed/nq)
        print("Preparing NQ dataset...")
        nq_dataset = NQDataset()
        questions_path, contexts_path = nq_dataset.convert_to_rag_format()
        
        # Initialize evaluator
        evaluator = RAGEvaluator(rag)
        
        # Run evaluation
        print(f"Evaluating RAG system with {args.model_type} model on {args.num_samples} questions...")
        metrics = evaluator.evaluate_nq_dataset(
            num_samples=args.num_samples,
            output_file=args.output_file
        )
        
        # Print summary
        evaluator.print_evaluation_summary(metrics)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if "OPENAI_API_KEY" in str(e):
            print("Please make sure your OpenAI API key is correctly set in the .env file")
        elif "llama_model_path" in str(e):
            print("Please provide a valid path to your Llama model file")
        else:
            print("Please check your configuration and try again")


if __name__ == "__main__":
    main() 