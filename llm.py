from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()


class GPT:
    """A ChatGPT-specific implementation for generating responses."""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo-1106",  # Latest GPT-3.5 model
        temperature: float = 0.3,  # Lower temperature for more focused responses
        max_tokens: Optional[int] = 1000,  # Reasonable default for responses
        top_p: float = 0.95,  # Nucleus sampling parameter
        frequency_penalty: float = 0.0,  # No frequency penalty by default
        presence_penalty: float = 0.0,  # No presence penalty by default
        streaming: bool = False  # No streaming by default
    ):
        """Initialize the ChatGPT model.
        
        Args:
            model_name: Name of the OpenAI model to use (default: latest GPT-3.5)
            temperature: Temperature for text generation (0.0 to 1.0, lower = more focused)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            frequency_penalty: Penalty for frequency of tokens (-2.0 to 2.0)
            presence_penalty: Penalty for presence of tokens (-2.0 to 2.0)
            streaming: Whether to stream the response
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            streaming=streaming,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Default prompt template optimized for ChatGPT
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful and precise assistant. Your task is to answer questions based on the provided context.
            Follow these rules:
            1. Use only information from the provided context
            2. If the context doesn't contain enough information, say "I cannot answer this question based on the provided context."
            3. Be as concise as possible in your responses
            
            Context: {context}"""),
            ("user", "{question}")
        ])
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def generate_response(
        self,
        question: str,
        context: List[Document],
        max_context_length: int = 4000,  # Increased for GPT-3.5's larger context window
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response based on the question and context.
        
        Args:
            question: The question to answer
            context: List of relevant documents
            max_context_length: Maximum length of context to include
            system_prompt: Optional custom system prompt to override the default
            
        Returns:
            Generated response as a string
        """
        # Combine context documents
        context_text = "\n\n".join(doc.page_content for doc in context)
        
        # Truncate context if too long
        if len(context_text) > max_context_length:
            context_text = context_text[:max_context_length] + "..."
        
        # Generate response
        response = self.chain.invoke({
            "context": context_text,
            "question": question
        })
        
        return response["text"]
    
def main():
    gpt = GPT(model_name="gpt-3.5-turbo")
    print(gpt.generate_response("What is the capital of France?", []))

if __name__ == "__main__":
    main()
