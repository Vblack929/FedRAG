import os
import json
import logging
import argparse
from tqdm import tqdm
from poisoner import RAGPoisoner

def parse_args():
    parser = argparse.ArgumentParser(description="Generate adversarial attacks for NQ dataset samples with long answers")
    parser.add_argument("--input_file", type=str, default="data/nq/nq_full_2000.json",
                        help="Path to the input JSON file")
    parser.add_argument("--output_dir", type=str, default="data/poisoned/nq",
                        help="Directory to save the poisoned dataset")
    parser.add_argument("--output_file", type=str, default="nq_long_answer_attacks_300_4o.json",
                        help="Name of the output file")
    parser.add_argument("--adv_per_query", type=int, default=3,
                        help="Number of adversarial texts per query")
    parser.add_argument("--corpus_length", type=int, default=20,
                        help="Approximate word length of each adversarial text")
    return parser.parse_args()

def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the dataset
    logger.info(f"Loading dataset from {args.input_file}")
    with open(args.input_file, 'r') as f:
        dataset = json.load(f)
    
    logger.info(f"Loaded {len(dataset)} samples from {args.input_file}")
    
    # Filter samples with non-empty long answers
    # Set a limit for the number of samples with long answers
    MAX_SAMPLES_WITH_LONG_ANSWERS = 300
    
    # Filter samples with non-empty long answers
    samples_with_long_answers = []
    for sample in dataset:
        if sample.get("long_answer"):
            samples_with_long_answers.append(sample)
            # Break if we've reached the maximum number of samples
            if len(samples_with_long_answers) >= MAX_SAMPLES_WITH_LONG_ANSWERS:
                break
    logger.info(f"Found {len(samples_with_long_answers)} samples with non-empty long answers")
    
    # Format the data for the poisoner
    qa_pairs = []
    for sample in samples_with_long_answers:
        if sample.get("correct_answer"):
            correct_answer = sample["correct_answer"]
        elif sample.get("short_answers") and len(sample["short_answers"]) > 0:
            correct_answer = sample["short_answers"][0]
        else:
            # Extract a short answer from the long answer (first 50 chars)
            correct_answer = sample["long_answer"][:50].strip()
            if len(correct_answer) >= 50:
                correct_answer += "..."
        
        qa_pairs.append({
            "id": sample["id"],
            "question": sample["question"],
            "correct_answer": correct_answer,
            "long_answer": sample["long_answer"]
        })
    
    # Initialize the poisoner
    logger.info("Initializing RAG poisoner")
    poisoner = RAGPoisoner(model_name="gpt-4o")
    
    # Initialize progress bar
    progress_bar = tqdm(total=len(qa_pairs), desc="Generating adversarial attacks")
    
    # Define callback function for progress updates
    def update_progress(current, total):
        progress_bar.update(1)
    
    # Prepare output data structure
    attack_results = {}
    
    # Generate attacks for each sample
    logger.info("Generating adversarial attacks for samples with long answers")
    for sample in qa_pairs:
        try:
            # Generate attack for this sample
            attack_result = poisoner.generate_targeted_attack(
                question=sample["question"],
                correct_answer=sample["correct_answer"],
                adv_per_query=args.adv_per_query,
                corpus_length=args.corpus_length
            )
            
            # Store the results including long_answer
            attack_results[sample["id"]] = {
                "id": sample["id"],
                "question": sample["question"],
                "correct_answer": sample["correct_answer"],
                "long_answer": sample["long_answer"],
                "incorrect_answer": attack_result["incorrect_answer"],
                "poisoned_texts": attack_result["adv_texts"],
                "concatenated_poisoned_text": " ".join(attack_result["adv_texts"])
            }
            
            # Update progress
            update_progress(1, len(qa_pairs))
            
        except Exception as e:
            logger.error(f"Error generating attack for question {sample['id']}: {e}")
    
    # Close the progress bar
    progress_bar.close()
    
    # Save the results
    output_path = os.path.join(args.output_dir, args.output_file)
    with open(output_path, 'w') as f:
        json.dump(list(attack_results.values()), f, indent=2)
    
    logger.info(f"Successfully generated attacks for {len(attack_results)} questions")
    logger.info(f"Results saved to {output_path}")
    
    # Print a sample of the results
    if attack_results:
        sample_id = next(iter(attack_results))
        logger.info(f"\nSample attack for question ID {sample_id}:")
        logger.info(f"Question: {attack_results[sample_id]['question']}")
        logger.info(f"Correct answer: {attack_results[sample_id]['correct_answer']}")
        logger.info(f"Long answer: {attack_results[sample_id]['long_answer'][:100]}...")
        logger.info(f"Incorrect answer: {attack_results[sample_id]['incorrect_answer']}")
        logger.info(f"First poisoned text: {attack_results[sample_id]['poisoned_texts'][0][:100]}...")
        logger.info(f"Concatenated poisoned text length: {len(attack_results[sample_id]['concatenated_poisoned_text'])}")

if __name__ == "__main__":
    main() 