import os
from typing import List
import chromadb
import PyPDF2
from groq import Groq
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import Tool, AgentExecutor, AgentType, initialize_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq 
import gtts
from gtts import gTTS
import datetime

# Custom Groq chat model class for LangChain
class GroqChatModel(BaseChatModel):
    client: Groq
    model_name: str
    temperature: float
    
    def __init__(self, api_key, model_name="mixtral-8x7b-32768", temperature=0.3):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        super().__init__()
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message_dicts = []
        for message in messages:
            if isinstance(message, SystemMessage):
                message_dicts.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                message_dicts.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                message_dicts.append({"role": "assistant", "content": message.content})
                
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_dicts,
            temperature=self.temperature,
            max_tokens=1000,
            stop=stop,
            **kwargs
        )
        
        return {"generations": [{"text": response.choices[0].message.content, "message": AIMessage(content=response.choices[0].message.content)}]}
    
    @property
    def _llm_type(self) -> str:
        return "groq-chat"

# Audio conversion function
def convert_to_audio(text, output_filename, lang='en'):
    """Convert text content to audio file using gTTS"""
    try:
        print(f"Converting text to audio: {output_filename}")
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_filename)
        print(f"Audio file saved: {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error in text to speech conversion: {e}")
        return None

# Set up environment variables for API keys
groq_api_key = "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"

# Initialize the LLM
llm = ChatGroq(api_key=groq_api_key, model_name="llama3-70b-8192")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Extract text from PDFs
def extract_text_from_pdfs(directory_path: str) -> List[Document]:
    print("==== Extracting text from PDFs ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            try:
                with open(pdf_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    if text.strip():
                        documents.append(Document(page_content=text, metadata={"source": filename}))
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return documents

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)

# Create output directory for audio files
def ensure_audio_dir():
    """Ensure the audio output directory exists"""
    audio_dir = "./audio_content"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    return audio_dir

# Load and process documents
def load_documents(directory_path: str):
    print("==== Loading and processing documents ====")
    # Extract documents from PDFs
    raw_documents = extract_text_from_pdfs(directory_path)
    print(f"Loaded {len(raw_documents)} documents from PDFs.")
    
    if not raw_documents:
        print("No documents were loaded. Please check the path and file formats.")
        return None
    
    # Split the documents into chunks
    split_documents = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(split_documents)} chunks")
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("==== Documents stored in vector database ====")
    return vectorstore

# Format educational content for better audio experience
def format_for_audio(content):
    """Format the content to be more suitable for audio playback"""
    # Replace lists and bullet points with more natural language
    content = content.replace("•", "")
    
    # Add pauses for section headers (using commas in text-to-speech)
    content = content.replace("#", ",")
    
    # Expand common abbreviations
    content = content.replace("e.g.", "for example")
    content = content.replace("i.e.", "that is")
    
    # Break long sentences
    sentences = content.split(". ")
    reformatted = ". ".join(sentences)
    
    return reformatted

# Make sure to use the directory path, not a file path
directory_path = 'C:\\Users\\S SHARMISTHA\\OneDrive\\Desktop\\Edtech\\machine learning'

# Create a more robust loading process
try:
    vectorstore = load_documents(directory_path)
    if not vectorstore:
        raise ValueError("Vector store creation failed")
except Exception as e:
    print(f"Error loading documents: {e}")
    print("Creating a simple document for testing...")
    # Create a simple document for testing if loading fails
    test_doc = Document(
        page_content="Deep learning is a subset of machine learning based on artificial neural networks.",
        metadata={"source": "test_document"}
    )
    vectorstore = Chroma.from_documents(
        documents=[test_doc],
        embedding=embeddings,
        persist_directory="./chroma_db_test"
    )

# Create a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Define the knowledge retrieval system
retrieval_prompt = ChatPromptTemplate.from_template("""
You are an AI tutor with access to a knowledge base on machine learning topics.

Based on the following query, respond with relevant, accurate information.
If the retrieved context doesn't contain enough information, say what you know about the topic generally.

Context: {context}
Question: {question}

Your response:
""")

knowledge_retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | retrieval_prompt
    | llm
    | StrOutputParser()
)

# Content generation system - optimized for audio
audio_content_generation_prompt = ChatPromptTemplate.from_template("""
You are an AI tutor generating educational content on {topic} that will be converted to audio.
Write in a conversational style suitable for listening rather than reading.
Use simpler sentences, avoid complex lists, and write in a way that sounds natural when spoken aloud.

Format your response as a script for narration with the following sections:
1. Introduction - Begin with "Welcome to this audio tutorial on [topic]"
2. Key Concepts - Explain core ideas clearly with pauses between different points
3. Main Explanation - Break down the topic step by step
4. Examples - Provide simple, concrete examples that work well verbally
5. Common Mistakes - Mention what learners often misunderstand
6. Summary - Briefly recap the main points
7. End with "Thank you for listening to this audio tutorial on [topic]"

Based on the knowledge retrieval and context provided, generate a comprehensive audio-friendly tutorial.

Topic: {topic}
""")

audio_content_generation_chain = (
    {"topic": RunnablePassthrough()}
    | audio_content_generation_prompt
    | llm
    | StrOutputParser()
)

# Content verification system - optimized for audio content
audio_verification_prompt = ChatPromptTemplate.from_template("""
You are an AI verifier specialized in audio educational content. Review the generated content for:
- Natural speech patterns suitable for listening (not reading)
- Clear pronunciation and rhythm (avoid terms that are hard to understand when spoken)
- Logical flow that a listener can follow without visual aids
- Factual accuracy on the topic: {topic}
- Absence of phrases like "as shown in the diagram" or "in the table below"

Generated Content:
{content}

If needed, refine to make it more suitable for audio learning. Provide the final reviewed content.
""")

audio_verification_chain = (
    {"topic": lambda x: x["topic"], "content": lambda x: x["content"]}
    | audio_verification_prompt
    | llm
    | StrOutputParser()
)

# Define agent tools
tools = [
    Tool(
        name="Knowledge_Retrieval",
        func=lambda query: knowledge_retrieval_chain.invoke(query),
        description="Useful for retrieving specific information about a topic from the knowledge base. Input should be a specific question."
    ),
    Tool(
        name="Audio_Content_Generation",
        func=lambda topic: audio_content_generation_chain.invoke(topic),
        description="Generates audio-friendly educational content on a specific topic. Input should be the topic name."
    ),
    Tool(
        name="Audio_Content_Verification",
        func=lambda input_str: audio_verification_chain.invoke({"topic": input_str.split("TOPIC:")[1].split("CONTENT:")[0].strip(),
                                                              "content": input_str.split("CONTENT:")[1].strip()}),
        description="Verifies and improves generated content for audio learning. Input should be formatted as 'TOPIC: [topic] CONTENT: [content]'"
    )
]

# Create the agent using initialize_agent instead of create_react_agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Main function to create educational audio content
def create_educational_audio_content(topic):
    """Generate educational content and convert to audio"""
    try:
        # Create audio directory
        audio_dir = ensure_audio_dir()
        
        print(f"\n===== GENERATING AUDIO EDUCATIONAL CONTENT FOR: {topic} =====\n")
        
        # Run the agent to generate audio-friendly content
        response = agent.run(f"Create audio educational content about: {topic}")
        
        content = response
        print("\n===== GENERATED CONTENT =====\n")
        print(content)
        
        # Format content for better audio experience
        audio_content = format_for_audio(content)
        
        # Generate a timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to safe filename (replace spaces and special chars)
        safe_topic = "".join([c if c.isalnum() else "_" for c in topic])
        
        # Create audio file
        audio_filename = os.path.join(audio_dir, f"{safe_topic}_{timestamp}.mp3")
        
        # Split content into chunks if too long (gTTS has limits)
        max_chunk_size = 3000  # Characters
        content_chunks = [audio_content[i:i+max_chunk_size] for i in range(0, len(audio_content), max_chunk_size)]
        
        if len(content_chunks) == 1:
            # Single audio file
            audio_path = convert_to_audio(audio_content, audio_filename)
        else:
            # Multiple audio files for longer content
            audio_files = []
            for i, chunk in enumerate(content_chunks):
                chunk_filename = os.path.join(audio_dir, f"{safe_topic}_{timestamp}_part{i+1}.mp3")
                audio_path = convert_to_audio(chunk, chunk_filename)
                if audio_path:
                    audio_files.append(audio_path)
            
            # Inform about multiple files
            audio_path = ", ".join(audio_files)
        
        return {
            "text_content": content,
            "audio_file": audio_path
        }
    
    except Exception as e:
        print(f"Error creating audio content: {e}")
        # Fallback to direct audio content generation
        audio_dir = ensure_audio_dir()
        content = audio_content_generation_chain.invoke(topic)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join([c if c.isalnum() else "_" for c in topic])
        audio_filename = os.path.join(audio_dir, f"{safe_topic}_{timestamp}_fallback.mp3")
        audio_path = convert_to_audio(content, audio_filename)
        
        return {
            "text_content": content,
            "audio_file": audio_path
        }

# Example usage
if __name__ == "__main__":
    query = "Explain deep learning basics"
    result = create_educational_audio_content(query)
    
    print("\n===== AUDIO CONTENT GENERATION COMPLETE =====\n")
    print(f"Text content generated: {len(result['text_content'])} characters")
    print(f"Audio file(s) created: {result['audio_file']}")
    print("\nAudio content has been successfully created and saved to your root repository.")