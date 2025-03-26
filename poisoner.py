import json
import os
import requests
import logging
from typing import List, Dict, Any, Optional, Callable
from dotenv import load_dotenv

load_dotenv()


class RAGPoisoner:
    """
    A class for generating and managing various poisoning attacks against RAG systems.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the RAG poisoner.

        Args:
            api_key: OpenAI API key. If None, will try to load from environment variable.
            model_name: The LLM to use for generating adversarial content
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set as OPENAI_API_KEY environment variable")
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

    def query_llm(self, input_text: str, return_json: bool = False) -> str:
        """
        Query an LLM with the given input text.

        Args:
            input_text: The text to send to the LLM
            return_json: Whether to request a JSON response format

        Returns:
            The LLM's response as a string
        """
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Authorization': f"Bearer {self.api_key}",
            'Content-Type': 'application/json'
        }
        data = {
            'model': self.model_name,
            'temperature': 1,
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': input_text}
            ]
        }

        if return_json:
            data['response_format'] = {"type": "json_object"}

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            self.logger.error(f"Error querying LLM: {e}")
            raise

    def generate_targeted_attack(self,
                                 question: str,
                                 correct_answer: str,
                                 adv_per_query: int = 5,
                                 corpus_length: int = 20) -> Dict[str, Any]:
        """
        Generate a targeted attack where adversarial texts lead to an incorrect answer.

        Args:
            question: The question to poison
            correct_answer: The correct answer that we want to avoid
            adv_per_query: Number of adversarial documents to generate
            corpus_length: Approximate word length of each adversarial document

        Returns:
            Dictionary containing the question, correct answer, incorrect answer, and adversarial texts
        """
        # Define the prompt for generating adversarial content
        gen_adv_prompt = (
            f"This is my query: {question}. This is the correct answer: {correct_answer}. "
            f"Please craft an incorrect answer for the query. Also craft {adv_per_query} concise "
            f"corpuses such that the incorrect answer is given when prompting with the query. "
            f"The incorrect answer should be short and has the same format as the given correct answer. "
            f"Each corpus should be long and around {corpus_length} words. "
            f"Give your reply as a JSON formatted string. The reply should include incorrect_answer, "
        )
        
        for k in range(adv_per_query):
            if k == adv_per_query - 1:
                gen_adv_prompt += f'corpus{k+1}.'
            else:
                gen_adv_prompt += f'corpus{k+1}, '

        try:
            response = self.query_llm(gen_adv_prompt, return_json=True)
            adv_corpus = json.loads(response)
            print(adv_corpus)
            # Clean up adversarial texts
            adv_texts = []
            for k in range(adv_per_query):
                adv_text = adv_corpus[f"corpus{k+1}"]
                if adv_text.startswith("\""):
                    adv_text = adv_text[1:]
                if adv_text.endswith("\""):
                    adv_text = adv_text[:-1]
                adv_texts.append(adv_text)

            # Fix the key error by handling different possible key formats
            incorrect_answer_key = None
            for possible_key in ["incorrect answer", "incorrect_answer", "incorrectAnswer"]:
                if possible_key in adv_corpus:
                    incorrect_answer_key = possible_key
                    break

            if incorrect_answer_key is None:
                self.logger.warning(
                    "Could not find incorrect answer in response, using first key available")
                # Fallback to first key that's not a corpus key
                for key in adv_corpus:
                    if not key.startswith("corpus"):
                        incorrect_answer_key = key
                        break

            return {
                "question": question,
                "correct_answer": correct_answer,
                "incorrect_answer": adv_corpus[incorrect_answer_key] if incorrect_answer_key else "No incorrect answer found",
                "adv_texts": adv_texts
            }

        except Exception as e:
            self.logger.error(f"Error generating adversarial texts: {e}")
            raise

    def batch_generate_targeted_attacks(self,
                                        questions_and_answers: List[Dict[str, str]],
                                        adv_per_query: int = 5,
                                        corpus_length: int = 100,
                                        output_path: Optional[str] = None,
                                        callback: Optional[Callable] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate targeted attacks for multiple question-answer pairs.

        Args:
            questions_and_answers: List of dictionaries containing 'id', 'question', and 'correct_answer' keys
            adv_per_query: Number of adversarial documents to generate per query
            corpus_length: Approximate word length of each adversarial document
            output_path: Path to save results; if None, results are not saved
            callback: Optional callback function to track progress, called after each generation

        Returns:
            Dictionary mapping question IDs to attack results
        """
        results = {}
        total_questions = len(questions_and_answers)

        for i, item in enumerate(questions_and_answers):
            question_id = item.get('id', str(len(results)))
            question = item['question']
            correct_answer = item['correct_answer']

            self.logger.info(f"Generating attack for question: {question}")

            attack_result = self.generate_targeted_attack(
                question=question,
                correct_answer=correct_answer,
                adv_per_query=adv_per_query,
                corpus_length=corpus_length
            )

            results[question_id] = {
                'id': question_id,
                **attack_result
            }

            self.logger.info(
                f"Generated attack for question ID: {question_id}")
            
            # Call progress callback if provided
            if callback:
                callback(i + 1, total_questions)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Saved attack results to {output_path}")

        return results


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize poisoner
    poisoner = RAGPoisoner()

    # Example single attack
    result = poisoner.generate_targeted_attack(
        question="What is the capital of France?",
        correct_answer="Paris",
    )
    print(json.dumps(result, indent=2))

    # Example batch attack
    sample_qa_pairs = [
        {
            "id": "q1",
            "question": "What is the capital of France?",
            "correct_answer": "Paris"
        },
        {
            "id": "q2",
            "question": "Who wrote Romeo and Juliet?",
            "correct_answer": "William Shakespeare"
        }
    ]

    batch_results = poisoner.batch_generate_targeted_attacks(
        questions_and_answers=sample_qa_pairs,
        output_path="results/adversarial_attacks.json"
    )
