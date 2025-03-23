import os
import chromadb
import PyPDF2
import graphviz
from groq import Groq
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq


# Custom Groq Chat Model
class GroqChatModel(BaseChatModel):
    def __init__(self, api_key, model_name="mixtral-8x7b-32768", temperature=0.3):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        super().__init__()

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message_dicts = [
            {"role": "system", "content": msg.content} if isinstance(msg, SystemMessage) else
            {"role": "user", "content": msg.content} if isinstance(msg, HumanMessage) else
            {"role": "assistant", "content": msg.content}
            for msg in messages
        ]

        response = self.client.chat.completions.create(
            model=self.model_name, messages=message_dicts, temperature=self.temperature, max_tokens=1000, stop=stop, **kwargs
        )

        return {"generations": [{"text": response.choices[0].message.content, "message": AIMessage(content=response.choices[0].message.content)}]}

    @property
    def _llm_type(self) -> str:
        return "groq-chat"


# Set up API key
groq_api_key = "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"

# Initialize LLM and Embeddings
llm = ChatGroq(api_key=groq_api_key, model_name="llama3-70b-8192")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Extract text from PDFs
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
                        documents.append(Document(page_content=text, metadata={"source": filename}))
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return documents


# Initialize and process documents
directory_path = 'C:\\Users\\S SHARMISTHA\\OneDrive\\Desktop\\Edtech\\machine learning'

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = extract_text_from_pdfs(directory_path)
split_documents = text_splitter.split_documents(documents)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=split_documents if split_documents else [Document(page_content="Placeholder content.")],
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Create a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Content Generation Prompt
content_generation_prompt = ChatPromptTemplate.from_template("""
You are an AI tutor generating educational content on {topic}.
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
""")

content_generation_chain = (
    {"topic": RunnablePassthrough()}
    | content_generation_prompt
    | llm
    | StrOutputParser()
)


# Mind Map Generator
def generate_mind_map(content, filename="mindmap"):
    dot = graphviz.Digraph(format="png")
    
    sections = content.split("\n\n")  # Split into sections
    root = "Learning Topic"
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

    dot.render(filename, cleanup=True)
    print(f"Mind map saved as {filename}.png")


# Generate content and save mind map
if __name__ == "__main__":
    topic = "Deep Learning Basics"
    
    # Generate structured learning content
    content = content_generation_chain.invoke(topic)
    
    # Save mind map to root directory
    generate_mind_map(content, filename="mindmap")

    # Save structured learning content as a text file
    with open("learning_content.txt", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("\n===== Learning Content Generated and Mind Map Saved =====\n")
    print("Check 'mindmap.png' for the visual representation.")
    print("Check 'learning_content.txt' for the full content.")
