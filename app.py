from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import torch
from sentence_transformers import SentenceTransformer
from learning_style import process_vark_answers
import json
from langchain_core.documents import Document
from quiz_generator import RAGQuizGenerator
import time
from gtts import gTTS

# Import your VARK content generation functions
from varkcontent import (
    generate_vark_content,
    query_documents,
    generate_visual_content,
    generate_auditory_content, 
    generate_reading_content,
    generate_kinesthetic_content
)

# Setup Graphviz paths early
possible_graphviz_paths = [
    r'C:\Program Files\Graphviz\bin',
    r'C:\Program Files (x86)\Graphviz\bin',
    r'C:\Graphviz\bin',
    r'C:\Users\ShanmugaPriya\AppData\Local\Programs\Graphviz\bin',
    r'C:\Users\ShanmugaPriya\Graphviz\bin',
]

# Add all possible paths to the PATH environment variable
for path in possible_graphviz_paths:
    if os.path.exists(path):
        print(f"Adding Graphviz path to environment: {path}")
        os.environ["PATH"] += os.pathsep + path

app = Flask(__name__, static_folder='../build')
CORS(app, resources={r"/api/*": {"origins": "*"}})

API_KEY = os.environ.get("GROQ_API_KEY", "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")  # Replace with your actual API key or use environment variable
quiz_gen = RAGQuizGenerator(api_key=API_KEY)

# Sample documents (you should replace this with your actual document loading logic)
def load_sample_documents():
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
        # Add more documents based on subjects and topics
    ]
    return sample_docs

# Initialize with sample documents (you should replace this with your document loading logic)
quiz_gen.load_documents(load_sample_documents())

@app.route('/api/generate-test', methods=['POST'])
def generate_test():
    try:
        data = request.json
        topic = data.get('topic')
        subject = data.get('subject')
        learning_style = data.get('learningStyle', 'visual')
        
        if not topic or not subject:
            return jsonify({"error": "Topic and subject are required"}), 400
        
        # Generate a quiz using the RAG Quiz Generator
        quiz_data = quiz_gen.generate_quiz(
            topic=f"{subject}: {topic}",  # Combine subject and topic for better context
            num_questions=5,  # You can adjust this or make it configurable
            difficulty="medium"  # You can map learning styles to difficulty if needed
        )
        
        if "error" in quiz_data:
            return jsonify({"error": quiz_data["error"]}), 500
        
        # Transform the quiz data to match the expected format in your React component
        questions = []
        for i, q in enumerate(quiz_data.get('questions', [])):
            question = {
                "id": str(i + 1),
                "text": q['question_text'],
                "correctAnswer": None,
                "options": []
            }
            
            if q['question_type'] == 'multiple_choice':
                question['options'] = q['options']
                question['correctAnswer'] = q['correct_answer']
            elif q['question_type'] == 'true_false':
                question['options'] = ["True", "False"]
                question['correctAnswer'] = "True" if q['correct_answer'] else "False"
            elif q['question_type'] == 'short_answer':
                # For short answer questions, we'll create dummy options
                # This is a simplification - you might want to handle this differently
                question['options'] = [q['correct_answer']] + ["Option 2", "Option 3", "Option 4"]
                question['correctAnswer'] = q['correct_answer']
                
            questions.append(question)
        
        response = {
            "quiz_title": quiz_data.get('quiz_title', f"Quiz on {topic}"),
            "topic": topic,
            "subject": subject,
            "questions": questions
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error generating test: {str(e)}")
        return jsonify({"error": f"Failed to generate test: {str(e)}"}), 500

@app.route('/api/generate-content', methods=['POST'])
def generate_content():
    data = request.json
    topic = data.get('topic', '')
    subject = data.get('subject', '')
    learning_style = data.get('learningStyle', 'visual')
    
    # Create necessary directories if they don't exist
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('static/audio', exist_ok=True)
    
    try:
        # Log the request
        print(f"Generating {learning_style} content for topic: {topic}")
        
        # Attempt to use the ML-based content generation
        content = generate_vark_content(topic, learning_style)
        
        # Check if we got an error response from the model
        if content.get('content_type') == 'error':
            print(f"Error from ML model: {content.get('message')}")
            raise Exception(content.get('message', 'Unknown error in content generation'))
        
        # Process the ML model output based on learning style
        if learning_style == 'visual':
            # Handle visual asset path
            visual_asset_path = content.get('visual_asset')
            if visual_asset_path:
                print(f"Visual asset path: {visual_asset_path}")
            else:
                print("No visual asset path returned from generation function")
                visual_asset_path = None
                
            return jsonify({
                'success': True,
                'content': {
                    'title': f'Visual Learning: {topic}',
                    'content': content.get('text_content', ''),
                    'visual_asset': visual_asset_path,  # This should be a URL path like "/static/images/filename.png"
                    'activities': [
                        "Create a mind map of key concepts",
                        "Watch video tutorials with visual demonstrations",
                        "Use color-coding for your notes"
                    ]
                }
            })
            
        elif learning_style == 'auditory':
            # For auditory content, ensure the audio file path is correctly handled
            audio_file_path = content.get('audio_file', '')
            
            if not audio_file_path:
                # If no audio file was generated, create one now as a fallback
                try:
                    # Create audio from the text content
                    audio_dir = "static/audio"
                    safe_topic = "".join([c if c.isalnum() else "_" for c in topic])
                    audio_filename = f"{safe_topic}_fallback.mp3"
                    audio_filepath = os.path.join(audio_dir, audio_filename)
                    
                    # Use content text to generate audio
                    text_to_speak = content.get('text_content', f"Audio tutorial about {topic}")
                    tts = gTTS(text=text_to_speak, lang='en', slow=False)
                    tts.save(audio_filepath)
                    
                    # Update the audio file path
                    audio_file_path = f"/static/audio/{audio_filename}"
                    print(f"Created fallback audio: {audio_file_path}")
                except Exception as e:
                    print(f"Error creating fallback audio: {str(e)}")
            
            return jsonify({
                'success': True,
                'content': {
                    'title': f'Auditory Learning: {topic}',
                    'content': content.get('text_content', ''),
                    'audio_file': audio_file_path,
                    'activities': [
                        "Listen to audio explanations and lectures",
                        "Participate in group discussions",
                        "Explain concepts out loud to yourself"
                    ]
                }
            })
            
        elif learning_style == 'reading':
            return jsonify({
                'success': True,
                'content': {
                    'title': f'Reading/Writing Learning: {topic}',
                    'content': content.get('text_content', ''),
                    'activities': [
                        "Take detailed notes and summarize key points",
                        "Read textbook chapters and articles",
                        "Write your own explanations of concepts"
                    ]
                }
            })
            
        elif learning_style == 'kinesthetic':
            # Parse activities from the content
            activities_text = content.get('activities', '')
            activities_list = []
            
            # Try to parse activities from the text
            if activities_text:
                import re
                activities = re.split(r'\d+\.\s+', activities_text)
                activities_list = [a.strip() for a in activities if a.strip()]
            
            if not activities_list:
                activities_list = [
                    "Perform hands-on experiments or activities",
                    "Use physical models or manipulatives",
                    "Apply concepts to real-world situations"
                ]
            
            return jsonify({
                'success': True,
                'content': {
                    'title': f'Kinesthetic Learning: {topic}',
                    'content': content.get('text_content', 'Learn through hands-on activities and practical application.'),
                    'activities': activities_list
                }
            })
            
    except Exception as e:
        import traceback
        print(f"Error using ML model for {learning_style} content: {str(e)}")
        print(traceback.format_exc())
        # Fall back to mock
        # Fall back to mock content but log the specific error
        
    # Fallback content if ML generation fails
    fallback_content = {
        'visual': {
            'title': f'Visual Learning: {topic}',
            'content': f'Here\'s a visual approach to learning {topic} in {subject}. Visual learners process information best through images, diagrams, and spatial understanding. This content includes visual aids and graphical representations to enhance your understanding.\n\nFor visual learners, we recommend starting with flowcharts that illustrate the key concepts and relationships within {topic}. Color-coding different elements can help you distinguish between important components and see how they interconnect.',
            'activities': [
                f"Create a colorful mind map connecting all concepts in {topic}",
                f"Watch video demonstrations about {topic} with annotations",
                f"Use color-coding when taking notes about {topic}",
                f"Visualize processes and concepts with diagrams"
            ]
        },
        'auditory': {
            'title': f'Auditory Learning: {topic}',
            'content': f'This content is structured for auditory learning of {topic} in {subject}. Auditory learners understand information best through listening and speaking. This module emphasizes verbal explanations and discussions.\n\nFor auditory learners, we recommend starting with verbal explanations of the key concepts in {topic}. Focus on understanding the terminology and being able to explain concepts in your own words. Consider recording lectures or discussions to review later.',
            'activities': [
                f"Listen to lectures or podcasts about {topic}",
                f"Discuss {topic} concepts with study partners",
                f"Record yourself explaining {topic} and listen to it later",
                f"Create mnemonics or songs to remember key information"
            ]
        },
        'reading': {
            'title': f'Reading/Writing Learning: {topic}',
            'content': f'This approach focuses on text-based learning for {topic} in {subject}. Reading/writing learners prefer information displayed as words and text. This module emphasizes written explanations, lists, and definitions.\n\nFor reading/writing learners, we recommend starting with comprehensive notes on {topic}. Focus on creating structured documents with headings, bullet points, and definitions. Rewrite key concepts in your own words to reinforce understanding.',
            'activities': [
                f"Create detailed notes and summaries about {topic}",
                f"Read textbooks and articles about {topic}",
                f"Write explanations of {topic} concepts in your own words",
                f"Create flashcards with key terms and definitions"
            ]
        },
        'kinesthetic': {
            'title': f'Kinesthetic Learning: {topic}',
            'content': f'For hands-on learners, this content provides practical ways to learn {topic} in {subject} through physical activities and real-world applications. Kinesthetic learners understand best through doing, touching, and physical experience.\n\nFor kinesthetic learners, we recommend starting with practical applications of {topic}. Focus on experiments, simulations, or real-world examples that you can interact with physically. Try to connect abstract concepts to tangible experiences.',
            'activities': [
                f"Perform hands-on experiments related to {topic}",
                f"Create physical models to represent {topic} concepts",
                f"Use role-playing to act out processes in {topic}",
                f"Apply {topic} concepts to real-world situations"
            ]
        }
    }
    
    # Get content based on learning style or default to visual
    selected_content = fallback_content.get(learning_style, fallback_content['visual'])
    
    return jsonify({
        'success': True,
        'content': selected_content
    })

@app.route('/api/assessment', methods=['POST'])
def save_assessment():
    data = request.json
    
    # Map the data from frontend format to your model's format
    learning_style = {
        'Visual': data.get('visual', 0),
        'Auditory': data.get('auditory', 0),
        'Reading/Writing': data.get('reading', 0),
        'Kinesthetic': data.get('kinesthetic', 0)
    }
    
    # Use the processing function from learning_style.py
    result = process_vark_answers(learning_style)
    
    # Convert styles to lowercase for frontend consistency
    dominant_style = result['primary_style'].lower()
    if dominant_style == "reading/writing":
        dominant_style = "reading"  # Simplify for frontend
    
    secondary_style = result.get('secondary_style', '').lower()
    if secondary_style == "reading/writing":
        secondary_style = "reading"
    
    # Return response in the format expected by the frontend
    return jsonify({
        'success': True,
        'learning_style': {
            'visual': learning_style['Visual'],
            'auditory': learning_style['Auditory'],
            'reading': learning_style['Reading/Writing'],
            'kinesthetic': learning_style['Kinesthetic']
        },
        'dominant_style': dominant_style,
        'secondary_style': secondary_style
    })

@app.route('/api/subjects', methods=['POST'])
def create_subject():
    # In a real app, you would save this to the database
    # For this example, we'll just return the data
    return jsonify({
        'success': True,
        'message': 'Subject created successfully',
        'data': request.json
    })

# Add this route to your app.py file

@app.route('/api/subjects/<subject_id>', methods=['DELETE'])
def delete_subject(subject_id):
    try:
        data = request.json
        user_id = data.get('userId')
        
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400
            
        # In a real app, you would delete from your database
        # This is a simplified example returning success
        
        return jsonify({
            'success': True,
            'message': f'Subject {subject_id} deleted successfully'
        })
        
    except Exception as e:
        print(f"Error deleting subject: {str(e)}")
        return jsonify({"error": f"Failed to delete subject: {str(e)}"}), 500

@app.route('/api/subjects/<subject_id>/progress', methods=['PUT'])
def update_progress(subject_id):
    data = request.json
    # In a real app, you would update the database
    return jsonify({
        'success': True,
        'message': f'Progress updated for subject {subject_id}',
        'progress': data.get('progress', 0)
    })

# Serve static files for generated content
@app.route('/static/<path:filename>')
def serve_static(filename):
    # Get the absolute path to the static directory
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    
    # Log the request for debugging
    print(f"Serving static file: {filename} from {static_dir}")
    
    # For audio files, set the correct MIME type
    if filename.endswith('.mp3'):
        return send_from_directory(static_dir, filename, mimetype='audio/mpeg')
    else:
        return send_from_directory(static_dir, filename)

# Serve React app
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Make sure static directory exists
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('static/audio', exist_ok=True)
    app.run(debug=True)