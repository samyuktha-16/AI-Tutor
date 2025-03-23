import os
import torch
import chromadb
import PyPDF2
from groq import Groq
from sentence_transformers import SentenceTransformer


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(name=collection_name)


# Set up Groq client
groq_client = Groq(api_key="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")  # Replace with your actual Groq API key


def get_local_embedding(text):
    return embedding_model.encode(text).tolist()


def extract_text_from_pdfs(directory_path):
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


def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


directory_path = r"C:\Users\S SHARMISTHA\OneDrive\Desktop\Edtech\machine learning"  # Update with your directory
documents = extract_text_from_pdfs(directory_path)

chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc['id']}_chunk{i+1}"
        embedding = get_local_embedding(chunk)
        chunked_documents.append({"id": chunk_id, "text": chunk, "embedding": embedding})
        collection.upsert(ids=[chunk_id], documents=[chunk], embeddings=[embedding])


def query_documents(question, n_results=2):
    question_embedding = get_local_embedding(question)
    results = collection.query(query_embeddings=[question_embedding], n_results=n_results)

    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]

    return relevant_chunks


def generate_kinesthetic_activities(relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = f"""Based on the following educational content, create 6 kinesthetic learning activities that will help users physically engage with the material. Each activity should:
1. Have a clear title
2. Include step-by-step instructions
3. Specify any simple materials needed (items commonly found at home or in a classroom)
4. Explain how the activity reinforces the key concepts
5. Be accessible for different learning environments (classroom, home, etc.)

Educational content to base activities on:
{context}

Format your response as a list of 6 numbered kinesthetic activities, with clear section breaks between each activity."""
    
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are an AI tutor that creates engaging kinesthetic learning activities."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800
    )
    
    return response.choices[0].message.content if response.choices else "No kinesthetic activities generated."


def generate_learning_content(relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = f"""You are an AI tutor who explains concepts in a structured and engaging way. Use the given content to generate the best possible learning material with detailed explanations, real-world examples, and key takeaways.

Context:
{context}

Learning Content:"""
    
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are an AI tutor that creates concise, accurate educational content."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    
    learning_content = response.choices[0].message.content if response.choices else "No learning content generated."
    
    # Generate kinesthetic activities separately
    kinesthetic_activities = generate_kinesthetic_activities(relevant_chunks)
    
    return learning_content, kinesthetic_activities


def verify_content(topic, generated_content):
    verification_prompt = f"""You are an AI verifier. Review the generated content for factual accuracy, clarity, and completeness. Ensure it matches the topic and is free from hallucinations.

*Topic:* {topic}
*Generated Content:*
{generated_content}

If needed, refine and correct errors. Provide the final reviewed content."""

    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are an AI reviewer that verifies and improves educational content."},
            {"role": "user", "content": verification_prompt}
        ],
        max_tokens=700
    )
    
    return response.choices[0].message.content if response.choices else "Verification failed."


# Example usage
def main():
    query = input("Enter a topic to learn about: ")
    relevant_chunks = query_documents(query)
    learning_content, kinesthetic_activities = generate_learning_content(relevant_chunks)
    
    # Verify the learning content
    verified_content = verify_content(query, learning_content)
    
    print("\n=== Generated Learning Content ===")
    print(verified_content)

    print("\n=== Kinesthetic Learning Activities ===")
    print(kinesthetic_activities)
    
    # Save activities to file
    with open("kinesthetic_activities.txt", "w") as f:
        f.write(kinesthetic_activities)
    print("\nKinesthetic activities saved to 'kinesthetic_activities.txt'")


if __name__ == "__main__":
    main()


# import os
# import torch
# import chromadb
# import PyPDF2
# import streamlit as st
# from groq import Groq
# from sentence_transformers import SentenceTransformer
# from tempfile import TemporaryDirectory
# import pandas as pd
# import time

# # Set page config
# st.set_page_config(page_title="Interactive Learning Content Generator", layout="wide")

# # Initialize session state variables
# if 'documents_processed' not in st.session_state:
#     st.session_state.documents_processed = False
# if 'collection' not in st.session_state:
#     st.session_state.collection = None
# if 'embedding_model' not in st.session_state:
#     st.session_state.embedding_model = None
# if 'groq_client' not in st.session_state:
#     st.session_state.groq_client = None
# if 'temp_dir' not in st.session_state:
#     st.session_state.temp_dir = TemporaryDirectory()

# # App title and description
# st.title("Interactive Learning Content Generator")
# st.markdown("""
# This application helps you generate educational content and kinesthetic learning activities based on your PDF documents.
# Upload your PDF files, ask questions, and get personalized learning materials!
# """)

# # Sidebar for configuration
# with st.sidebar:
#     st.header("Configuration")
    
#     # API key input
#     groq_api_key = st.text_input("Enter your Groq API Key:", type="password", 
#                                   value="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")
    
#     # Model selection
#     embedding_model_name = st.selectbox(
#         "Select Embedding Model:",
#         ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
#         index=0
#     )
    
#     llm_model = st.selectbox(
#         "Select LLM Model:",
#         ["mixtral-8x7b-32768", "llama3-70b-8192", "llama3-8b-8192"],
#         index=0
#     )
    
#     # Chunking parameters
#     chunk_size = st.slider("Chunk Size:", min_value=500, max_value=2000, value=1000, step=100)
#     chunk_overlap = st.slider("Chunk Overlap:", min_value=0, max_value=200, value=20, step=10)
    
#     # Number of results to retrieve
#     n_results = st.slider("Number of chunks to retrieve:", min_value=1, max_value=10, value=2)

# # Initialize models
# def initialize_models():
#     with st.spinner("Initializing models..."):
#         # Check CUDA availability
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         st.session_state.device = device
#         st.info(f"Using device: {device}")
        
#         # Initialize embedding model
#         st.session_state.embedding_model = SentenceTransformer(embedding_model_name)
        
#         # Initialize Groq client
#         st.session_state.groq_client = Groq(api_key=groq_api_key)
        
#         # Initialize ChromaDB
#         chroma_client = chromadb.PersistentClient(path=os.path.join(st.session_state.temp_dir.name, "chroma_db"))
#         collection_name = "document_qa_collection"
#         st.session_state.collection = chroma_client.get_or_create_collection(name=collection_name)
        
#         st.success("Models initialized successfully!")

# # Function to get embeddings
# def get_local_embedding(text):
#     return st.session_state.embedding_model.encode(text).tolist()

# # Function to extract text from PDFs
# def extract_text_from_pdfs(uploaded_files):
#     documents = []
    
#     for uploaded_file in uploaded_files:
#         with st.spinner(f"Processing {uploaded_file.name}..."):
#             # Save uploaded file to temp directory
#             temp_file_path = os.path.join(st.session_state.temp_dir.name, uploaded_file.name)
#             with open(temp_file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
            
#             # Extract text from PDF
#             with open(temp_file_path, "rb") as file:
#                 reader = PyPDF2.PdfReader(file)
#                 text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
#                 if text.strip():
#                     documents.append({"id": uploaded_file.name, "text": text})
    
#     return documents

# # Function to split text into chunks
# def split_text(text, chunk_size, chunk_overlap):
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = min(start + chunk_size, len(text))
#         chunks.append(text[start:end])
#         start = end - chunk_overlap
#     return chunks

# # Function to query documents
# def query_documents(question, n_results):
#     question_embedding = get_local_embedding(question)
#     results = st.session_state.collection.query(query_embeddings=[question_embedding], n_results=n_results)
    
#     relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
#     chunk_ids = [id for sublist in results["ids"] for id in sublist]
    
#     return relevant_chunks, chunk_ids

# # Function to generate kinesthetic activities
# def generate_kinesthetic_activities(relevant_chunks, topic):
#     context = "\n\n".join(relevant_chunks)
#     prompt = f"""Based on the following educational content about "{topic}", create 6 kinesthetic learning activities that will help users physically engage with the material. Each activity should:
# 1. Have a clear title
# 2. Include step-by-step instructions
# 3. Specify any simple materials needed (items commonly found at home or in a classroom)
# 4. Explain how the activity reinforces the key concepts
# 5. Be accessible for different learning environments (classroom, home, etc.)

# Educational content to base activities on:
# {context}

# Format your response as a list of 6 numbered kinesthetic activities, with clear section breaks between each activity."""
    
#     response = st.session_state.groq_client.chat.completions.create(
#         model=llm_model,
#         messages=[
#             {"role": "system", "content": "You are an AI tutor that creates engaging kinesthetic learning activities."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=800
#     )
    
#     return response.choices[0].message.content if response.choices else "No kinesthetic activities generated."

# # Function to generate learning content
# def generate_learning_content(relevant_chunks, topic):
#     context = "\n\n".join(relevant_chunks)
#     prompt = f"""You are an AI tutor who explains concepts in a structured and engaging way. Use the given content to generate the best possible learning material about "{topic}" with detailed explanations, real-world examples, and key takeaways.

# Context:
# {context}

# Learning Content:"""
    
#     response = st.session_state.groq_client.chat.completions.create(
#         model=llm_model,
#         messages=[
#             {"role": "system", "content": "You are an AI tutor that creates concise, accurate educational content."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=500
#     )
    
#     return response.choices[0].message.content if response.choices else "No learning content generated."

# # Function to verify content
# def verify_content(topic, generated_content):
#     verification_prompt = f"""You are an AI verifier. Review the generated content for factual accuracy, clarity, and completeness. Ensure it matches the topic "{topic}" and is free from hallucinations.

# *Topic:* {topic}
# *Generated Content:*
# {generated_content}

# If needed, refine and correct errors. Provide the final reviewed content."""

#     response = st.session_state.groq_client.chat.completions.create(
#         model=llm_model,
#         messages=[
#             {"role": "system", "content": "You are an AI reviewer that verifies and improves educational content."},
#             {"role": "user", "content": verification_prompt}
#         ],
#         max_tokens=700
#     )
    
#     return response.choices[0].message.content if response.choices else "Verification failed."

# # Main app flow
# if not st.session_state.documents_processed:
#     if st.button("Initialize Models"):
#         initialize_models()
    
#     # File uploader
#     uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    
#     if uploaded_files and st.button("Process Documents"):
#         if not st.session_state.embedding_model:
#             initialize_models()
        
#         documents = extract_text_from_pdfs(uploaded_files)
        
#         if documents:
#             with st.spinner("Processing documents and generating embeddings..."):
#                 progress_bar = st.progress(0)
#                 for i, doc in enumerate(documents):
#                     chunks = split_text(doc["text"], chunk_size, chunk_overlap)
#                     st.write(f"Processing {doc['id']}: {len(chunks)} chunks")
                    
#                     for j, chunk in enumerate(chunks):
#                         chunk_id = f"{doc['id']}_chunk{j+1}"
#                         embedding = get_local_embedding(chunk)
#                         st.session_state.collection.upsert(ids=[chunk_id], documents=[chunk], embeddings=[embedding])
                        
#                         # Update progress bar
#                         progress_bar.progress((i / len(documents)) + (j / len(chunks)) / len(documents))
#                         time.sleep(0.01)  # Small delay to show progress
                
#                 progress_bar.progress(1.0)
#                 st.session_state.documents_processed = True
#                 st.success(f"Successfully processed {len(documents)} documents!")
#                 st.experimental_rerun()
#         else:
#             st.error("No text could be extracted from the uploaded PDFs.")

# else:
#     # Query interface
#     st.header("Query Your Documents")
    
#     query = st.text_input("Enter a topic to learn about:")
    
#     if query and st.button("Generate Learning Content"):
#         with st.spinner("Retrieving relevant content..."):
#             relevant_chunks, chunk_ids = query_documents(query, n_results)
            
#             if relevant_chunks:
#                 st.success(f"Found {len(relevant_chunks)} relevant chunks!")
                
#                 # Show source documents in expander
#                 with st.expander("Source Documents"):
#                     for i, (chunk, chunk_id) in enumerate(zip(relevant_chunks, chunk_ids)):
#                         st.markdown(f"**Source {i+1}:** {chunk_id.split('_chunk')[0]}")
#                         st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
#                         st.markdown("---")
                
#                 # Generate content
#                 with st.spinner("Generating learning content..."):
#                     learning_content = generate_learning_content(relevant_chunks, query)
#                     verified_content = verify_content(query, learning_content)
                
#                 # Display content in tabs
#                 tab1, tab2 = st.tabs(["Learning Content", "Kinesthetic Activities"])
                
#                 with tab1:
#                     st.markdown("### Generated Learning Content")
#                     st.markdown(verified_content)
                    
#                     # Download button for learning content
#                     st.download_button(
#                         label="Download Learning Content",
#                         data=verified_content,
#                         file_name=f"{query.replace(' ', '_')}_learning_content.md",
#                         mime="text/markdown"
#                     )
                
#                 with tab2:
#                     st.markdown("### Kinesthetic Activities")
                    
#                     # Generate kinesthetic activities
#                     with st.spinner("Generating kinesthetic activities..."):
#                         kinesthetic_activities = generate_kinesthetic_activities(relevant_chunks, query)
                    
#                     st.markdown(kinesthetic_activities)
                    
#                     # Download button for kinesthetic activities
#                     st.download_button(
#                         label="Download Kinesthetic Activities",
#                         data=kinesthetic_activities,
#                         file_name=f"{query.replace(' ', '_')}_kinesthetic_activities.md",
#                         mime="text/markdown"
#                     )
#             else:
#                 st.error("No relevant content found. Please try a different query or upload more documents.")
    
#     # Reset button
#     if st.button("Reset Application"):
#         # Clear session state
#         for key in list(st.session_state.keys()):
#             del st.session_state[key]
#         st.session_state.documents_processed = False
#         st.experimental_rerun()