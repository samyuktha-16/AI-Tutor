import os
import json
import random
from typing import List, Dict, Any

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

class RAGQuizGenerator:
    def __init__(self, api_key, vector_db_path="./chroma_db"):
        """Initialize the RAG Quiz Generator with LLM and vector DB."""
        self.api_key = api_key
        self.vector_db_path = vector_db_path
        
        # Initialize the LLM
        self.llm = ChatGroq(api_key=api_key, model_name="llama3-70b-8192")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load vector store if it exists
        if os.path.exists(vector_db_path):
            self.vectorstore = Chroma(
                persist_directory=vector_db_path,
                embedding_function=self.embeddings
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        else:
            self.vectorstore = None
            self.retriever = None
            print("Vector database not found. Please load documents first.")
    
    def load_documents(self, documents: List[Document]):
        """Load documents into the vector store."""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.vector_db_path
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        print(f"Loaded {len(documents)} documents into vector store.")
    
    def generate_quiz(self, topic: str, num_questions: int = 5, difficulty: str = "medium"):
        """Generate a quiz on the specified topic using RAG."""
        if not self.retriever:
            return {"error": "No documents loaded. Please load documents first."}
        
        # Retrieve relevant context for the topic
        contexts = self.retriever.invoke(topic)
        context_text = "\n\n".join([doc.page_content for doc in contexts])
        
        # Create the quiz generation prompt
        quiz_prompt = ChatPromptTemplate.from_template("""
        You are an educational quiz creator. Based on the following context and topic, 
        create a quiz with {num_questions} questions at {difficulty} difficulty level.
        
        The quiz should test understanding of key concepts, applications, and principles
        related to {topic}.
        
        Include a mix of:
        - Multiple-choice questions (with 4 options each)
        - True/False questions
        - Short answer questions
        
        For each question, provide:
        1. The question text
        2. The answer options (for multiple choice)
        3. The correct answer
        4. A brief explanation of why the answer is correct
        
        CONTEXT:
        {context}
        
        TOPIC: {topic}
        
        Format your response as a JSON object with the following structure:
        {{
            "quiz_title": "Quiz title here",
            "topic": "{topic}",
            "difficulty": "{difficulty}",
            "questions": [
                {{
                    "question_type": "multiple_choice",
                    "question_text": "Question text here",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "Correct option here",
                    "explanation": "Explanation here"
                }},
                {{
                    "question_type": "true_false",
                    "question_text": "True/False question here",
                    "correct_answer": true,
                    "explanation": "Explanation here"
                }},
                {{
                    "question_type": "short_answer",
                    "question_text": "Short answer question here",
                    "correct_answer": "Correct answer here",
                    "explanation": "Explanation here"
                }}
            ]
        }}
        
        IMPORTANT: Return ONLY the JSON object, with no additional text.
        """)
        
        # Generate the quiz
        quiz_chain = (
            quiz_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        try:
            result = quiz_chain.invoke({
                "topic": topic,
                "num_questions": num_questions,
                "difficulty": difficulty,
                "context": context_text
            })
            
            # Parse the JSON output
            quiz_data = json.loads(result)
            return quiz_data
        except Exception as e:
            return {"error": f"Failed to generate quiz: {str(e)}"}
    
    def evaluate_response(self, question, user_answer):
        """Evaluate a user's response to a quiz question."""
        evaluation_prompt = ChatPromptTemplate.from_template("""
        You are grading a student's answer to a quiz question.
        
        Question: {question_text}
        Correct Answer: {correct_answer}
        Student's Answer: {user_answer}
        
        Evaluate how correct the student's answer is on a scale of 0-100%, and provide
        feedback explaining why.
        
        Return your response as a JSON object with the following structure:
        {{
            "score": score_as_integer,
            "feedback": "Your feedback here"
        }}
        
        IMPORTANT: Return ONLY the JSON object, with no additional text.
        """)
        
        evaluation_chain = (
            evaluation_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        try:
            result = evaluation_chain.invoke({
                "question_text": question["question_text"],
                "correct_answer": question["correct_answer"],
                "user_answer": user_answer
            })
            
            # Parse the JSON output
            evaluation_data = json.loads(result)
            return evaluation_data
        except Exception as e:
            return {"error": f"Failed to evaluate response: {str(e)}"}
    
    def evaluate_quiz_result(self, total_score, total_questions):
        """Evaluate the overall quiz result to determine pass/fail.
        
        Args:
            total_score: The total score achieved by the user
            total_questions: The total number of questions in the quiz
            
        Returns:
            dict: A dictionary containing the percentage score and pass/fail status
        """
        percentage = (total_score / total_questions) * 100
        
        result = {
            "percentage": round(percentage, 2),
            "status": "PASS" if percentage > 70 else "FAIL"
        }
        
        return result
    
    def run_quiz(self, quiz_data):
        """Run a quiz interactively and evaluate user responses."""
        if "error" in quiz_data:
            print(f"Error: {quiz_data['error']}")
            return
        
        print(f"\n===== {quiz_data['quiz_title']} =====")
        print(f"Topic: {quiz_data['topic']}")
        print(f"Difficulty: {quiz_data['difficulty']}")
        print("\nAnswer the following questions:\n")
        
        total_score = 0
        total_questions = len(quiz_data['questions'])
        question_results = []
        
        for i, question in enumerate(quiz_data['questions']):
            print(f"Question {i+1}: {question['question_text']}")
            
            user_answer = None
            if question['question_type'] == 'multiple_choice':
                for j, option in enumerate(question['options']):
                    print(f"  {chr(97+j)}) {option}")
                
                user_input = input("Your answer (a, b, c, d): ").lower()
                # Convert letter answer to actual option
                try:
                    index = ord(user_input) - ord('a')
                    if 0 <= index < len(question['options']):
                        user_answer = question['options'][index]
                    else:
                        print("Invalid option. Skipping question.")
                        continue
                except:
                    print("Invalid input. Skipping question.")
                    continue
                
            elif question['question_type'] == 'true_false':
                user_input = input("True or False? (T/F): ").upper()
                if user_input == 'T':
                    user_answer = True
                elif user_input == 'F':
                    user_answer = False
                else:
                    print("Invalid input. Skipping question.")
                    continue
                
            elif question['question_type'] == 'short_answer':
                user_answer = input("Your answer: ")
            
            # Evaluate the response
            evaluation = self.evaluate_response(question, user_answer)
            
            if "error" in evaluation:
                print(f"Error evaluating response: {evaluation['error']}")
                question_score = 0
            else:
                question_score = evaluation["score"] / 100
                total_score += question_score
                
                print(f"\nFeedback: {evaluation['feedback']}")
                print(f"Score for this question: {evaluation['score']}%")
            
            print(f"Correct answer: {question['correct_answer']}")
            print(f"Explanation: {question['explanation']}\n")
            
            question_results.append({
                "question": question['question_text'],
                "user_answer": user_answer,
                "correct_answer": question['correct_answer'],
                "score": question_score * 100
            })
        
        # Evaluate overall result
        result = self.evaluate_quiz_result(total_score, total_questions)
        
        print("\n===== Quiz Results =====")
        print(f"Total Score: {result['percentage']}%")
        print(f"Status: {result['status']}")
        
        if result['status'] == "PASS":
            print("Congratulations! You passed the quiz.")
        else:
            print("You did not pass the quiz. Keep studying and try again!")
        
        return {
            "quiz_title": quiz_data['quiz_title'],
            "topic": quiz_data['topic'],
            "difficulty": quiz_data['difficulty'],
            "total_score": result['percentage'],
            "status": result['status'],
            "question_results": question_results
        }
    
    def save_quiz(self, quiz_data, file_path):
        """Save the generated quiz to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(quiz_data, f, indent=2)
        print(f"Quiz saved to {file_path}")
    
    def load_quiz(self, file_path):
        """Load a quiz from a JSON file."""
        with open(file_path, 'r') as f:
            quiz_data = json.load(f)
        return quiz_data
    
    def save_quiz_results(self, results, file_path):
        """Save quiz results to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Quiz results saved to {file_path}")

# Example usage
if __name__ == "__main__":
    # Replace with your actual API key
    API_KEY = "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"
    
    # Initialize the quiz generator
    quiz_gen = RAGQuizGenerator(api_key=API_KEY)
    
    # Example documents (in a real application, you would load these from files)
    sample_docs = [
        Document(
            page_content="""
            Retrieval-Augmented Generation (RAG) is an AI framework that enhances large language
            models with external knowledge. RAG combines information retrieval with text generation
            to produce more accurate, up-to-date, and verifiable responses. The process involves
            retrieving relevant information from a knowledge base and using it as context for
            generating responses.
            """,
            metadata={"source": "rag_overview.txt"}
        ),
        Document(
            page_content="""
            The typical RAG architecture consists of:
            1. A document store where source documents are stored
            2. An embedding model to convert text into vector representations
            3. A vector database for efficient similarity search
            4. A retriever component that finds relevant documents
            5. A generator component (usually an LLM) that produces the final output
            """,
            metadata={"source": "rag_architecture.txt"}
        ),
        Document(
            page_content="""
            Vector databases like Chroma, Pinecone, or Weaviate are essential for RAG systems.
            They index embeddings for efficient similarity search, enabling the system to quickly
            find relevant documents based on semantic meaning rather than just keywords.
            """,
            metadata={"source": "vector_databases.txt"}
        )
    ]
    
    # Load the documents
    quiz_gen.load_documents(sample_docs)
    
    # Choose what you want to do:
    mode = input("Do you want to (g)enerate a new quiz or (l)oad an existing one? (g/l): ").lower()
    
    if mode == 'g':
        # Generate a quiz on RAG
        topic = input("Enter quiz topic: ")
        num_questions = int(input("Enter number of questions: "))
        difficulty = input("Enter difficulty (easy/medium/hard): ")
        
        quiz = quiz_gen.generate_quiz(
            topic=topic,
            num_questions=num_questions,
            difficulty=difficulty
        )
        
        # Save the quiz
        quiz_gen.save_quiz(quiz, "rag_quiz.json")
        
    elif mode == 'l':
        # Load an existing quiz
        file_path = input("Enter quiz file path (default: rag_quiz.json): ") or "rag_quiz.json"
        quiz = quiz_gen.load_quiz(file_path)
    else:
        print("Invalid option. Exiting.")
        exit()
    
    # Ask if the user wants to take the quiz
    take_quiz = input("Do you want to take the quiz now? (y/n): ").lower()
    
    if take_quiz == 'y':
        # Run the quiz
        results = quiz_gen.run_quiz(quiz)
        
        # Ask if the user wants to save the results
        save_results = input("Do you want to save your quiz results? (y/n): ").lower()
        if save_results == 'y':
            results_file = input("Enter results file path (default: quiz_results.json): ") or "quiz_results.json"
            quiz_gen.save_quiz_results(results, results_file)