import os
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import urllib.request


class DatasetHandler:
    """Class to handle loading and processing of various datasets for RAG systems."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the dataset handler.
        
        Args:
            data_dir: Directory to store the datasets
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_file(self, url: str, filepath: str) -> None:
        """Download a file from a URL with a progress bar.
        
        Args:
            url: URL to download from
            filepath: Path to save the downloaded file
        """
        print(f"Downloading {url} to {filepath}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(filepath, 'wb') as f:
            for data in tqdm(response.iter_content(block_size), total=total_size//block_size, unit='KiB', unit_scale=True):
                f.write(data)


class NQDataset(DatasetHandler):
    """Handler for the Natural Questions dataset."""
    
    def __init__(self, data_dir: str = "data/nq"):
        """Initialize the NQ dataset handler.
        
        Args:
            data_dir: Directory to store the NQ dataset
        """
        super().__init__(data_dir)
        self.simplified_url = "https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl"
        self.simplified_filepath = os.path.join(self.data_dir, "nq_open_dev.jsonl")
        
        # Full dataset URLs (these are large)
        self.full_urls = {
            "train": "https://storage.googleapis.com/natural_questions/v1.0/train/nq-train-00.jsonl.gz",
            "dev": "https://storage.googleapis.com/natural_questions/v1.0/dev/nq-dev-00.jsonl.gz"
        }
    
    def download_simplified(self) -> None:
        """Download the simplified NQ-open dataset (smaller and easier to work with)."""
        if not os.path.exists(self.simplified_filepath):
            self.download_file(self.simplified_url, self.simplified_filepath)
            print(f"Downloaded simplified NQ dataset to {self.simplified_filepath}")
        else:
            print(f"Simplified NQ dataset already exists at {self.simplified_filepath}")
    
    def download_full(self, split: str = "dev") -> str:
        """Download the full NQ dataset (warning: very large).
        
        Args:
            split: Dataset split to download ("train" or "dev")
            
        Returns:
            Path to the downloaded file
        """
        if split not in self.full_urls:
            raise ValueError(f"Invalid split: {split}. Must be one of {list(self.full_urls.keys())}")
        
        filepath = os.path.join(self.data_dir, f"nq_{split}.jsonl.gz")
        if not os.path.exists(filepath):
            self.download_file(self.full_urls[split], filepath)
            print(f"Downloaded full NQ {split} dataset to {filepath}")
        else:
            print(f"Full NQ {split} dataset already exists at {filepath}")
            
        return filepath
    
    def load_simplified(self) -> List[Dict]:
        """Load the simplified NQ-open dataset.
        
        Returns:
            List of question-answer pairs
        """
        # Download if not exists
        if not os.path.exists(self.simplified_filepath):
            self.download_simplified()
        
        # Load the data
        data = []
        with open(self.simplified_filepath, 'r') as f:
            for line in f:
                item = json.loads(line)
                data.append({
                    "question": item["question"],
                    "answers": item["answer"]
                })
        
        print(f"Loaded {len(data)} questions from simplified NQ dataset")
        return data
    
    def convert_to_rag_format(self, output_dir: str = "data/processed/nq") -> Tuple[str, str]:
        """Convert the NQ dataset to a format suitable for RAG training/evaluation.
        
        Args:
            output_dir: Directory to save the processed files
            
        Returns:
            Tuple of (questions_path, contexts_path)
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the data
        data = self.load_simplified()
        
        # Create questions file
        questions_path = os.path.join(output_dir, "questions.txt")
        with open(questions_path, 'w') as f:
            for item in data:
                f.write(f"{item['question']}\n")
        
        # Create contexts file with answers
        contexts_path = os.path.join(output_dir, "contexts.txt")
        with open(contexts_path, 'w') as f:
            for item in data:
                # Join multiple answers with newlines
                if isinstance(item['answers'], list):
                    answers_text = "\n".join(item['answers'])
                else:
                    answers_text = str(item['answers'])
                f.write(f"{answers_text}\n\n")
        
        print(f"Converted NQ dataset to RAG format:")
        print(f"- Questions saved to {questions_path}")
        print(f"- Contexts/answers saved to {contexts_path}")
        
        return questions_path, contexts_path
    
    def create_evaluation_set(self, num_samples: int = 100, output_file: str = "data/nq_eval.json") -> str:
        """Create an evaluation set from the NQ dataset.
        
        Args:
            num_samples: Number of samples to include in the evaluation set
            output_file: Path to save the evaluation set
            
        Returns:
            Path to the evaluation set
        """
        # Load the data
        data = self.load_simplified()
        
        # Take a sample
        import random
        if len(data) > num_samples:
            eval_data = random.sample(data, num_samples)
        else:
            eval_data = data
        
        # Create the evaluation set
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        print(f"Created evaluation set with {len(eval_data)} samples at {output_file}")
        return output_file
    
    def create_minimal_dataset(self, num_samples: int = 10, output_dir: str = "data/minimal_nq") -> Dict[str, str]:
        """Create a minimal dataset for quick experimentation.
        
        This creates a very small subset of the NQ dataset with separate files for questions and answers,
        plus a combined file for easy reference, making it ideal for initial RAG experiments.
        
        Args:
            num_samples: Number of samples to include in the minimal dataset
            output_dir: Directory to save the minimal dataset
            
        Returns:
            Dictionary with paths to the created files
        """
        # Load the data
        data = self.load_simplified()
        
        # Take a small sample
        import random
        random.seed(42)  # For reproducibility
        if len(data) > num_samples:
            sample_data = random.sample(data, num_samples)
        else:
            sample_data = data
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create individual files for questions and answers
        questions_path = os.path.join(output_dir, "questions.txt")
        answers_path = os.path.join(output_dir, "answers.txt")
        combined_path = os.path.join(output_dir, "qa_pairs.json")
        
        # Write questions file
        with open(questions_path, 'w') as f:
            for item in sample_data:
                f.write(f"{item['question']}\n")
        
        # Write answers file
        with open(answers_path, 'w') as f:
            for item in sample_data:
                if isinstance(item['answers'], list):
                    f.write(f"{' | '.join(item['answers'])}\n")
                else:
                    f.write(f"{item['answers']}\n")
        
        # Write combined JSON file
        with open(combined_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"\nCreated minimal dataset with {len(sample_data)} samples:")
        print(f"- Questions: {questions_path}")
        print(f"- Answers: {answers_path}")
        print(f"- Combined QA pairs: {combined_path}")
        
        return {
            "questions": questions_path,
            "answers": answers_path,
            "combined": combined_path
        }


def main():
    """Example usage of the dataset handlers."""
    # Initialize the NQ dataset handler
    nq_dataset = NQDataset()
    
    # Create a minimal dataset for quick experimentation
    print("Creating a minimal dataset for quick experimentation...")
    minimal_dataset_paths = nq_dataset.create_minimal_dataset(num_samples=10)
    
    # Download and process the simplified NQ dataset
    print("\nDownloading and processing the simplified NQ dataset...")
    nq_dataset.download_simplified()
    questions_path, contexts_path = nq_dataset.convert_to_rag_format()
    
    # Create an evaluation set
    eval_path = nq_dataset.create_evaluation_set(num_samples=50)
    
    print(f"\nDataset preparation complete!")
    print(f"You can use these files with your RAG system:")
    print(f"- Minimal dataset: {minimal_dataset_paths['combined']}")
    print(f"- Full questions: {questions_path}")
    print(f"- Full contexts: {contexts_path}")
    print(f"- Evaluation set: {eval_path}")


if __name__ == "__main__":
    main() 