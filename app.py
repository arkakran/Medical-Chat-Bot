try:
    import faiss
except ImportError:
    import os
    os.system("pip install faiss-cpu==1.7.4.post2")
    import faiss



from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from utils.pdf_processor import process_pdf_complete
from utils.vector_database import (
    initialize_embeddings, load_vector_database, add_to_vector_database,
    save_vector_database, get_database_stats
)
from utils.retrieval_qa import generate_medical_response, validate_medical_query

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Groq client
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# PDF path
PDF_PATH = "data/Medical_book.pdf"

VECTOR_DB_PATH = "data/medical_vector_store"

def initialize_medical_chatbot():
    print("Initializing Medical Chatbot System.")
    
    # Initialize embeddings
    initialize_embeddings()
    
    # Try to load existing vector database
    if load_vector_database(VECTOR_DB_PATH):
        stats = get_database_stats()
        if stats['total_chunks'] > 0:
            print(f"Loaded existing medical knowledge base with {stats['total_chunks']} chunks")
            return True
    
    # Process PDF if no existing database
    print("No existing knowledge base found. Processing medical PDF.")
    return process_medical_pdf()

def process_medical_pdf():
    try:
        if not os.path.exists(PDF_PATH):
            print(f"Medical PDF not found: {PDF_PATH}")
            return False
        
        print(f"Processing medical PDF: {PDF_PATH}")
        
        # Complete PDF processing pipeline
        chunks = process_pdf_complete(PDF_PATH)
        
        if not chunks:
            print("Failed to process PDF")
            return False
        
        # Add to vector database
        add_to_vector_database(chunks)
        
        # Save vector database
        save_vector_database(VECTOR_DB_PATH)
        
        stats = get_database_stats()
        print(f"Medical knowledge base created with {stats['total_chunks']} chunks!")
        return True
        
    except Exception as e:
        print(f"Error processing medical PDF: {e}")
        return False


@app.route('/')
def index():
    """Main chat interface"""
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Please enter a message'}), 400
        
        if len(user_message) > 500:
            return jsonify({'error': 'Message too long. Please keep it under 500 characters.'}), 400
        
        # Validate medical query
        if not validate_medical_query(user_message):
            return jsonify({
                'response': "Please ask a medical-related question. I'm specialized in providing medical information from my knowledge base.",
                'timestamp': datetime.now().strftime('%H:%M')
            })
        
        # Generate medical response using RAG
        response = generate_medical_response(user_message, groq_client)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().strftime('%H:%M')
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """System health check"""
    try:
        stats = get_database_stats()
        return jsonify({
            'status': 'healthy',
            'vector_database': stats,
            'groq_model': 'llama-3.3-70b-versatile',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/reprocess_pdf', methods=['POST'])
def reprocess_pdf():
    """Reprocess PDF and recreate vector database"""
    try:
        print("Reprocessing medical PDF...")
        
        # Clear existing database
        global vector_index, vector_texts, vector_metadatas
        from utils.vector_database import vector_index, vector_texts, vector_metadatas
        vector_index = None
        vector_texts.clear()
        vector_metadatas.clear()
        
        # Process PDF
        success = process_medical_pdf()
        
        if success:
            stats = get_database_stats()
            return jsonify({
                'message': 'PDF reprocessed successfully!',
                'chunks_count': stats['total_chunks']
            })
        else:
            return jsonify({'error': 'Failed to reprocess PDF'}), 500
            
    except Exception as e:
        print(f"Error reprocessing PDF: {e}")
        return jsonify({'error': f'Failed to reprocess PDF: {str(e)}'}), 500

if __name__ == '__main__':
    initialize_medical_chatbot()
    app.run(debug=True, host='0.0.0.0', port=5000)


