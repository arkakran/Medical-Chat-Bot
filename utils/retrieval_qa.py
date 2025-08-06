from groq import Groq
from typing import List, Tuple, Dict
from utils.vector_database import similarity_search

def create_medical_prompt_template():
    return """You are a highly knowledgeable and professional medical assistant. Use the medical context provided below, along with your general medical knowledge, to generate accurate, helpful, and easy-to-understand responses to the user's question.

Instructions:
• Prioritize information found in the medical context whenever relevant.
• If specific details are not present in the context, provide a reliable and informative response based on your broader medical knowledge—without stating that the context is lacking.
• Structure your response using clear formatting, such as bullet points (•) or numbered lists, where appropriate. But dont use +.
• Use professional yet approachable language suitable for both medical professionals and laypersons.
• Explain complex medical terms in simple language when needed.
• Do not mention the absence or presence of information in the context. Focus on delivering a complete and helpful answer.

Medical Context:
{context}

User Question:
{question}

Answer:"""



def retrieve_relevant_context(query: str, top_k: int = 5) -> str:
    print(f"Retrieving relevant context for: '{query[:50]}...'")
    
    results = similarity_search(query, k=top_k)
    
    if not results:
        print("No relevant context found")
        return "No relevant medical information found in the knowledge base."
    
    # Format context with relevance scores
    contexts = []
    for i, (text, score, metadata) in enumerate(results, 1):
        context_piece = f"[Source {i} - Relevance: {score:.2f}]\n{text.strip()}"
        contexts.append(context_piece)
    
    formatted_context = "\n\n" + "="*50 + "\n\n".join(contexts)
    print(f"Retrieved {len(results)} relevant contexts")
    return formatted_context



def generate_medical_response(query: str, groq_client: Groq) -> str:
    print(f"Generating response for query...")
    try:
        top_k = 3
        max_top_k = 12
        retries = 0
        final_response = ""
        
        while top_k <= max_top_k:
            context = retrieve_relevant_context(query, top_k=top_k)
            prompt_template = create_medical_prompt_template()
            prompt = prompt_template.format(context=context, question=query)
            
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1200,
                top_p=0.9,
                stream=False
            )
            
            response = completion.choices[0].message.content
            formatted_response = format_medical_response(response)
            final_response = formatted_response
            
            # Critic Agent: Check if answer is too vague
            if is_response_complete(formatted_response):
                print(f"Response accepted with top_k = {top_k}")
                break
            else:
                print(f"Incomplete response with top_k = {top_k}. Retrying...")
                top_k += 1
                retries += 1
        
        if retries > 0:
            print(f"Critic agent improved answer in {retries} attempt(s)")
        
        return final_response

    except Exception as e:
        print(f"Error generating medical response: {e}")
        return "I apologize, but I encountered an error while processing your medical question. Please try rephrasing your question or consult a healthcare professional directly."


def is_response_complete(response: str) -> bool:
    """Critic agent checks if response is likely complete"""
    vague_phrases = [
        "I'm sorry", "I don't know", "insufficient information",
        "not enough context", "I cannot provide", "I am unable"
    ]
    
    if not response or len(response) < 80:
        return False
    
    return not any(phrase.lower() in response.lower() for phrase in vague_phrases)




def format_medical_response(response: str) -> str:
    # Remove excessive formatting
    response = response.replace("**", "")
    response = response.replace("*", "")
    
    # Ensure proper spacing
    response = response.replace("\n\n\n", "\n\n")
    response = response.strip()
    
    # Add medical disclaimer at the bottom
    disclaimer = ""
    
    if not response.endswith(disclaimer.strip()):
        response += disclaimer
    
    return response

def validate_medical_query(query: str) -> bool:
    medical_indicators = [
        'symptom', 'disease', 'condition', 'treatment', 'medicine', 'medication',
        'diagnosis', 'doctor', 'hospital', 'pain', 'fever', 'infection',
        'medical', 'health', 'illness', 'therapy', 'surgery', 'procedure',
        'blood', 'heart', 'lung', 'brain', 'liver', 'kidney', 'cancer',
        'diabetes', 'hypertension', 'pneumonia', 'what is', 'how to treat',
        'causes of', 'prevention', 'cure', 'relief'
    ]
    
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in medical_indicators)
