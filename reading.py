# import os
# import torch
# import chromadb
# import PyPDF2
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# # Set up device
# device = "cpu"
# print(f"Using device: {device}")

# # Load embedding model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Initialize ChromaDB
# chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
# collection = chroma_client.get_or_create_collection(name="document_qa_collection")

# # Function to compute embeddings
# def get_local_embedding(text):
#     return embedding_model.encode(text).tolist()

# # Extract text from PDFs
# def extract_text_from_pdfs(directory_path):
#     print("==== Extracting text from PDFs ====")
#     documents = []
#     for filename in os.listdir(directory_path):
#         if filename.endswith(".pdf"):
#             pdf_path = os.path.join(directory_path, filename)
#             with open(pdf_path, "rb") as file:
#                 reader = PyPDF2.PdfReader(file)
#                 text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
#                 if text.strip():
#                     documents.append({"id": filename, "text": text})
#     return documents

# # Split text into chunks
# def split_text(text, chunk_size=1000, chunk_overlap=50):
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start = end - chunk_overlap
#     return chunks

# # Load PDFs and store chunks in ChromaDB
# # Make sure to use the directory path, not a file path
# directory_path = 'C:\\Users\\S SHARMISTHA\\OneDrive\\Desktop\\Edtech\\machine learning'
# documents = extract_text_from_pdfs(directory_path)

# print(f"Loaded {len(documents)} documents from PDFs.")

# for doc in documents:
#     chunks = split_text(doc["text"])
#     print(f"Splitting {doc['id']} into {len(chunks)} chunks")
#     for i, chunk in enumerate(chunks):
#         chunk_id = f"{doc['id']}_chunk{i+1}"
#         embedding = get_local_embedding(chunk)
#         collection.upsert(ids=[chunk_id], documents=[chunk], embeddings=[embedding])

# print("==== Documents stored in ChromaDB ====")

# # Query documents for the specific topic
# def query_documents(topic, n_results=3):
#     topic_embedding = get_local_embedding(topic)
#     results = collection.query(query_embeddings=[topic_embedding], n_results=n_results)
    
#     print("===== Query Results from ChromaDB =====")
#     print(results)

#     relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]

#     if not relevant_chunks:
#         print("⚠ No relevant chunks found. Check embeddings and stored documents.")

#     return relevant_chunks

# # Load LLM (Mistral 7B)
# model_name = "mixtral-8x7b-32768"
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,  
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
# llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device = -1)

# # Generate learning content
# def generate_learning_content(topic, relevant_chunks):
#     context = "\n\n".join(relevant_chunks)
#     prompt = f"""You are an AI tutor generating content *only for the given topic*. Ensure the content is structured, clear, and includes examples.

# *Topic:* {topic}  
# *Relevant Content:* {context}

# *Format:*
# 1. *Introduction*
# 2. *Key Concepts*
# 3. *Step-by-Step Explanation*
# 4. *Examples & Applications*
# 5. *Common Mistakes & Misconceptions*
# 6. *Summary & Key Takeaways*

# Generate structured learning material for the topic: {topic}."""

#     response = llm(prompt, max_new_tokens=700)
#     return response[0]["generated_text"] if response else "No learning content generated."

# # AI Agent to verify and refine content
# def verify_content(topic, generated_content):
#     verification_prompt = f"""You are an AI verifier. Review the generated content for factual accuracy, clarity, and completeness. Ensure it matches the topic and is free from hallucinations.

# *Topic:* {topic}
# *Generated Content:*
# {generated_content}

# If needed, refine and correct errors. Provide the final reviewed content."""

#     response = llm(verification_prompt, max_new_tokens=700)
#     return response[0]["generated_text"] if response else "Verification failed."

# # User input
# topic_query = "Explain deep learning basics"
# relevant_chunks = query_documents(topic_query)
# learning_content = generate_learning_content(topic_query, relevant_chunks)
# verified_content = verify_content(topic_query, learning_content)

# # Output final verified content
# print("\n===== FINAL VERIFIED CONTENT =====\n")
# print(verified_content)

import os
import chromadb
import PyPDF2
from groq import Groq
from sentence_transformers import SentenceTransformer

# Set up Groq client
groq_client = Groq(api_key="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")  # Replace with your actual Groq API key

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = chroma_client.get_or_create_collection(name="document_qa_collection")

# Function to compute embeddings
def get_local_embedding(text):
    return embedding_model.encode(text).tolist()

# Extract text from PDFs
def extract_text_from_pdfs(directory_path):
    print("==== Extracting text from PDFs ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                if text.strip():
                    documents.append({"id": filename, "text": text})
    return documents

# Split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Load PDFs and store chunks in ChromaDB
# Make sure to use the directory path, not a file path
directory_path = 'C:\\Users\\S SHARMISTHA\\OneDrive\\Desktop\\Edtech\\machine learning'
documents = extract_text_from_pdfs(directory_path)

print(f"Loaded {len(documents)} documents from PDFs.")

for doc in documents:
    chunks = split_text(doc["text"])
    print(f"Splitting {doc['id']} into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc['id']}_chunk{i+1}"
        embedding = get_local_embedding(chunk)
        collection.upsert(ids=[chunk_id], documents=[chunk], embeddings=[embedding])

print("==== Documents stored in ChromaDB ====")

# Query documents for the specific topic
def query_documents(topic, n_results=3):
    topic_embedding = get_local_embedding(topic)
    results = collection.query(query_embeddings=[topic_embedding], n_results=n_results)
    
    print("===== Query Results from ChromaDB =====")
    print(results)

    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]

    if not relevant_chunks:
        print("⚠ No relevant chunks found. Check embeddings and stored documents.")

    return relevant_chunks

# Generate learning content using Groq API
def generate_learning_content(topic, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = f"""You are an AI tutor generating content *only for the given topic*. Ensure the content is structured, clear, and includes examples.

*Topic:* {topic}  
*Relevant Content:* {context}

*Format:*
1. *Introduction*
2. *Key Concepts*
3. *Step-by-Step Explanation*
4. *Examples & Applications*
5. *Common Mistakes & Misconceptions*
6. *Summary & Key Takeaways*

Generate structured learning material for the topic: {topic}."""

    # Use Groq to generate content
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",  # You can choose different models here
        messages=[
            {"role": "system", "content": "You are an AI tutor that creates concise, accurate educational content."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=700
    )
    
    return response.choices[0].message.content if response.choices else "No learning content generated."

# AI Agent to verify and refine content
def verify_content(topic, generated_content):
    verification_prompt = f"""You are an AI verifier. Review the generated content for factual accuracy, clarity, and completeness. Ensure it matches the topic and is free from hallucinations.

*Topic:* {topic}
*Generated Content:*
{generated_content}

If needed, refine and correct errors. Provide the final reviewed content."""

    # Use Groq to verify content
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",  # You can choose different models here
        messages=[
            {"role": "system", "content": "You are an AI reviewer that verifies and improves educational content."},
            {"role": "user", "content": verification_prompt}
        ],
        max_tokens=700
    )
    
    return response.choices[0].message.content if response.choices else "Verification failed."

# User input
topic_query = "Explain deep learning basics"
relevant_chunks = query_documents(topic_query)
learning_content = generate_learning_content(topic_query, relevant_chunks)
verified_content = verify_content(topic_query, learning_content)

# Output final verified content
print("\n===== FINAL VERIFIED CONTENT =====\n")
print(verified_content)