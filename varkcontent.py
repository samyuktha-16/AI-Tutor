import os
import torch
import chromadb
import PyPDF2
from groq import Groq
from sentence_transformers import SentenceTransformer
import time
from gtts import gTTS
import graphviz
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

# Initialize common components
device = "cuda" if torch.cuda.is_available() else "cpu"
groq_api_key = "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"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=groq_api_key)
llm = ChatGroq(api_key=groq_api_key, model_name="llama3-70b-8192")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = chroma_client.get_or_create_collection(name="document_qa_collection")

# Common functions
def get_local_embedding(text):
    return embedding_model.encode(text).tolist()

def extract_text_from_pdfs(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            try:
                with open(pdf_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    if text.strip():
                        documents.append({"id": filename, "text": text})
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return documents

def split_text(text, chunk_size=1000, chunk_overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

def query_documents(topic, n_results=3):
    """
    Query ChromaDB collection for documents relevant to a topic
    
    Parameters:
    topic (str): The topic to search for
    n_results (int): Number of results to return
    
    Returns:
    list: Relevant document chunks
    """
    try:
        # Ensure ChromaDB client and collection are initialized
        chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
        try:
            collection = chroma_client.get_collection(name="document_qa_collection")
            print("Found existing ChromaDB collection")
        except Exception as e:
            print(f"Creating new ChromaDB collection: {str(e)}")
            collection = chroma_client.create_collection(name="document_qa_collection")
            
            # Add some dummy content if collection is empty
            dummy_text = f"This is a document about {topic}. Education content requires specialized teaching methods."
            collection.add(
                ids=["dummy_doc_1"],
                documents=[dummy_text],
                embeddings=[get_local_embedding(dummy_text)]
            )
            
        # Get embeddings for the topic
        topic_embedding = get_local_embedding(topic)
        
        # Query the collection
        results = collection.query(query_embeddings=[topic_embedding], n_results=n_results)
        
        # Extract relevant chunks
        relevant_chunks = []
        if results and 'documents' in results and results['documents']:
            relevant_chunks = [doc for sublist in results['documents'] for doc in sublist]
            
        # If no results, return a generic document
        if not relevant_chunks:
            print(f"No relevant documents found for topic: {topic}")
            return [f"Educational content about {topic}. This topic is important for learning and understanding key concepts."]
            
        return relevant_chunks
        
    except Exception as e:
        print(f"Error querying documents: {str(e)}")
        # Return a fallback document
        return [f"Educational content about {topic}. This topic is important for learning and understanding key concepts."]

# VARK-specific content generation functions
def generate_visual_content(topic, relevant_chunks):
    """Generate visual learning content (mind maps, diagrams)"""
    try:
        # Content generation prompt
        content_generation_prompt = ChatPromptTemplate.from_template("""
        You are an AI tutor generating educational content for visual learners on {topic}.
        Create content that emphasizes visual learning with clear structure and organization.
        
        Format:
        1. Introduction
        2. Key Concepts
        3. Visual Relationships
        4. Examples & Applications
        5. Visual Summary
        
        Based on the retrieved knowledge, generate a comprehensive tutorial for visual learners.
        
        Topic: {topic}
        """)
        
        content_generation_chain = (
            {"topic": RunnablePassthrough()}
            | content_generation_prompt
            | llm
            | StrOutputParser()
        )
        
        # Generate content
        content = content_generation_chain.invoke(topic)
        
        try:
            # Try to generate a mind map with Graphviz with enhanced error handling
            # Try multiple possible Graphviz installation paths
            possible_paths = [
                r'C:\Program Files\Graphviz\bin',
                r'C:\Program Files (x86)\Graphviz\bin',
                r'C:\Graphviz\bin',
                r'C:\Users\ShanmugaPriya\AppData\Local\Programs\Graphviz\bin',
                r'C:\Users\ShanmugaPriya\Graphviz\bin',
                # Add other possible paths where Graphviz might be installed
            ]
            
            # Add all possible paths to the PATH environment variable
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Adding Graphviz path: {path}")
                    os.environ["PATH"] += os.pathsep + path
            
            # Try to find the dot executable
            dot_cmd = 'dot'
            dot_path = None
            for path in possible_paths:
                candidate_path = os.path.join(path, "dot.exe")
                if os.path.exists(candidate_path):
                    dot_path = candidate_path
                    dot_cmd = dot_path
                    print(f"Found dot executable at: {dot_path}")
                    # Set environment variable instead of constructor parameter
                    os.environ["GRAPHVIZ_DOT"] = dot_path
                    break
            
            # Create Graphviz diagram without the executable parameter
            dot = graphviz.Digraph(format="png", engine='dot')
            
            sections = content.split("\n\n")  # Split into sections
            root = topic
            dot.node(root, shape="box", style="filled", fillcolor="lightblue")
            
            for section in sections:
                lines = section.strip().split("\n")
                if len(lines) > 1:
                    heading = lines[0]
                    dot.node(heading, shape="ellipse", style="filled", fillcolor="lightgrey")
                    dot.edge(root, heading)
                    
                    for line in lines[1:]:
                        if line.strip():
                            subheading = line[:40]  # Shorten long lines
                            dot.node(subheading, shape="rectangle", style="filled", fillcolor="white")
                            dot.edge(heading, subheading)
            
            # Save the diagram to a file in the static directory
            output_dir = os.path.join(os.getcwd(), "static/images")
            os.makedirs(output_dir, exist_ok=True)
            filename = f"visual_content_{topic.replace(' ', '_').replace('/', '_')}"
            full_path = os.path.join(output_dir, filename)
            print(f"Trying to render diagram to: {full_path}")
            
            # Render the diagram
            rendered_file = dot.render(filename=full_path, cleanup=True)
            print(f"Successfully rendered diagram to: {rendered_file}")
            
            # Return the path relative to the static directory
            return {
                "content_type": "visual",
                "text_content": content,
                "visual_asset": f"/static/images/{os.path.basename(rendered_file)}"
            }
            
        except Exception as e:
            import traceback
            print(f"Graphviz error: {str(e)}")
            print(traceback.format_exc())
            
            # If Graphviz fails, return text content only
            return {
                "content_type": "visual",
                "text_content": content,
                "visual_asset": None,
                "error": f"Unable to generate diagram: {str(e)}"
            }
            
    except Exception as e:
        print(f"Error generating visual content: {str(e)}")
        fallback_content = f"Visual learning content for {topic}. This topic can be understood through diagrams and visual representations."
        return {
            "content_type": "visual",
            "text_content": fallback_content,
            "visual_asset": None
        }

# Modified function for generate_auditory_content in varkcontent.py
def generate_auditory_content(topic, relevant_chunks):
    """Generate auditory learning content (audio explanations)"""
    try:
        context = "\n\n".join(relevant_chunks)
        
        # Audio-friendly content generation prompt
        prompt = f"""
        You are an AI tutor generating educational content on {topic} that will be converted to audio.
        Write in a conversational style suitable for listening rather than reading.
        Use simpler sentences, avoid complex lists, and write naturally for spoken content.
        
        Format your response as a script for narration with:
        1. Introduction - Begin with "Welcome to this audio tutorial on {topic}"
        2. Key Concepts - Explain core ideas clearly with transitions between points
        3. Main Explanation - Break down the topic step by step
        4. Examples - Provide simple, concrete examples
        5. Summary - Briefly recap the main points
        6. End with "Thank you for listening to this audio tutorial on {topic}"
        
        Based on this context: {context}
        """
        
        # Generate content
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",  # Using a more reliable model
            messages=[
                {"role": "system", "content": "You are an AI tutor creating audio educational content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800
        )
        
        audio_content = response.choices[0].message.content
        
        # Format content for better audio experience
        audio_content = audio_content.replace("•", "")
        audio_content = audio_content.replace("#", ",")
        audio_content = audio_content.replace("e.g.", "for example")
        audio_content = audio_content.replace("i.e.", "that is")
        
        # Create audio file
        audio_dir = "static/audio"
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
        
        # Create a more reliable filename
        safe_topic = "".join([c if c.isalnum() else "_" for c in topic])
        timestamp = str(int(time.time()))
        audio_filename = f"{safe_topic}_{timestamp}.mp3"
        audio_filepath = os.path.join(audio_dir, audio_filename)
        
        # Generate the audio file
        tts = gTTS(text=audio_content, lang='en', slow=False)
        tts.save(audio_filepath)
        print(f"Audio created successfully at: {audio_filepath}")
        
        # Return the relative path that will work with the server's routing
        return {
            "content_type": "auditory",
            "text_content": audio_content,
            "audio_file": f"/static/audio/{audio_filename}"  # Use path that matches Flask's static route
        }
    except Exception as e:
        import traceback
        print(f"Error in text to speech conversion: {str(e)}")
        print(traceback.format_exc())
        # Create a fallback version with a shorter text to reduce chance of errors
        try:
            fallback_text = f"Welcome to this audio tutorial on {topic}. This is a simplified version due to technical difficulties. {topic} is an important subject to learn about. Thank you for listening."
            
            audio_dir = "static/audio"
            if not os.path.exists(audio_dir):
                os.makedirs(audio_dir)
                
            fallback_filename = f"fallback_{int(time.time())}.mp3"
            fallback_path = os.path.join(audio_dir, fallback_filename)
            
            
            tts = gTTS(text=fallback_text, lang='en', slow=False)
            tts.save(fallback_path)
            
            return {
                "content_type": "auditory",
                "text_content": fallback_text,
                "audio_file": f"/static/audio/{fallback_filename}"
            }
        except Exception as fallback_error:
            print(f"Even fallback audio generation failed: {str(fallback_error)}")
            # Return just the text content with no audio file
            return {
                "content_type": "auditory",
                "text_content": f"Welcome to this audio tutorial on {topic}. This is a text-only version.",
                "audio_file": None,
                "error": str(e)
            }

def generate_reading_content(topic, relevant_chunks):
    """Generate reading/writing learning content (structured text)"""
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""You are an AI tutor generating educational content on {topic}.
    Ensure the content is structured, clear, and includes examples.
    
    Format:
    1. Introduction
    2. Key Concepts
    3. Step-by-Step Explanation
    4. Examples & Applications
    5. Common Mistakes & Misconceptions
    6. Summary & Key Takeaways
    
    Based on the retrieved knowledge, generate a comprehensive tutorial.
    
    Topic: {topic}
    Context: {context}
    """
    
    # Generate content
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are an AI tutor that creates concise, accurate educational content."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800
    )
    
    content = response.choices[0].message.content
    
    # Verify content
    verification_prompt = f"""You are an AI verifier. Review the generated content for factual accuracy, clarity, and completeness. Ensure it matches the topic and is free from hallucinations.
    
    Topic: {topic}
    Generated Content: {content}
    
    If needed, refine and correct errors. Provide the final reviewed content.
    """
    
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are an AI reviewer that verifies and improves educational content."},
            {"role": "user", "content": verification_prompt}
        ],
        max_tokens=800
    )
    
    verified_content = response.choices[0].message.content
    
    return {
        "content_type": "reading",
        "text_content": verified_content
    }

def generate_kinesthetic_content(topic, relevant_chunks):
    """Generate kinesthetic learning content (activities, exercises)"""
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""Based on the following educational content about "{topic}", create 6 kinesthetic learning activities that will help users physically engage with the material. Each activity should:
    1. Have a clear title
    2. Include step-by-step instructions
    3. Specify any simple materials needed (items commonly found at home or in a classroom)
    4. Explain how the activity reinforces the key concepts
    5. Be accessible for different learning environments (classroom, home, etc.)
    
    Educational content to base activities on:
    {context}
    
    Format your response as a list of 6 numbered kinesthetic activities, with clear section breaks between each activity.
    """
    
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are an AI tutor that creates engaging kinesthetic learning activities."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800
    )
    
    activities = response.choices[0].message.content
    
    return {
        "content_type": "kinesthetic",
        "activities": activities
    }

# Main function to generate VARK-based content
def generate_vark_content(topic, vark_style):
    """
    Generate educational content based on the user's VARK learning style
    
    Parameters:
    topic (str): The learning topic
    vark_style (str): The user's learning style (visual, auditory, reading, kinesthetic)
    
    Returns:
    dict: Content appropriate for the specified learning style
    """
    # Get relevant chunks for the topic
    relevant_chunks = query_documents(topic)
    
    if not relevant_chunks:
        return {
            "content_type": "error",
            "message": "No relevant content found for this topic. Please try a different topic or upload more documents."
        }
    
    if vark_style.lower() == "visual":
        return generate_visual_content(topic, relevant_chunks)
    
    elif vark_style.lower() == "auditory":
        return generate_auditory_content(topic, relevant_chunks)
    
    elif vark_style.lower() == "reading":
        return generate_reading_content(topic, relevant_chunks)
    
    elif vark_style.lower() == "kinesthetic":
        return generate_kinesthetic_content(topic, relevant_chunks)
    
    else:
        # Default to a mix of styles
        return generate_reading_content(topic, relevant_chunks)

# Load and process documents
def load_and_process_documents(directory_path):
    """Load and process PDF documents from a directory"""
    documents = extract_text_from_pdfs(directory_path)
    
    for doc in documents:
        chunks = split_text(doc["text"])
        print(f"Processing {doc['id']}: {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}_chunk{i+1}"
            embedding = get_local_embedding(chunk)
            collection.upsert(ids=[chunk_id], documents=[chunk], embeddings=[embedding])
    
    return len(documents)

# Example usage
if __name__ == "__main__":
    # Load documents
    # Use os.path.join for paths
    directory_path = os.path.join('C:', 'Users', 'ShanmugaPriya', 'Downloads','BP501T_MEDCHEM_-UNIT-_IV')
    num_docs = load_and_process_documents(directory_path)
    print(f"Processed {num_docs} documents")
    
    # Example: Generate content for different learning styles
    topic = "Medical Chemistry"
    
    # Visual learner content
    visual_content = generate_vark_content(topic, "visual")
    print(f"Visual content generated: {visual_content.get('visual_asset', 'No visual asset')}")
    
    # Auditory learner content
    auditory_content = generate_vark_content(topic, "auditory")
    print(f"Auditory content generated: {auditory_content.get('audio_file', 'No audio file')}")
    
    # Reading/writing learner content
    reading_content = generate_vark_content(topic, "reading")
    print(f"Reading content generated: {len(reading_content['text_content'])} characters")
    
    # Kinesthetic learner content
    kinesthetic_content = generate_vark_content(topic, "kinesthetic")
    print(f"Kinesthetic activities generated: {len(kinesthetic_content['activities'])} characters")