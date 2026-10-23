import streamlit as st # Streamlit lib
import sys
import os
import importlib.util # For importing response generator file
from typing import Dict, Any, List
import time

def load_css():
    """Load CSS from external file"""
    try:
        # Get the path to the CSS file
        css_path = os.path.join(os.path.dirname(__file__), 'styles', 'app.css')
        
        # Check if CSS file exists
        if os.path.exists(css_path):
            with open(css_path, 'r') as f:
                css_content = f.read()
            # Apply the CSS
            st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
        else:
            st.warning(f"CSS file not found at: {css_path}")
    except Exception as e:
        st.error(f"Failed to load CSS: {e}")

def load_response_generator():
    """Load response generator module using direct import"""
    try:
        # Add the project root to sys.path so Python find modules
        project_root = os.path.dirname(__file__)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Import directly
        from src.core.response_generator import GeminiResponseGenerator
        st.success("Successfully loaded response generator from src/core/!")
        return GeminiResponseGenerator
        
    except ImportError as e:
        st.error(f"Import error: {e}")
        st.info("Make sure all files are in the correct locations and imports are updated")
        return None
    except Exception as e:
        st.error(f"Failed to load response generator: {e}")
        return None

# Load the class
GeminiResponseGenerator = load_response_generator()
if GeminiResponseGenerator is None:
    st.stop()

# st.set_page_config() configures page settings
st.set_page_config(
    page_title="UCSB College of Engineering Assistant",  # Browser tab title
    page_icon="🛠️",                           # Browser tab icon
    layout="wide",                            # Layout: "centered" or "wide"
    initial_sidebar_state="expanded"          # Sidebar: "auto", "expanded", or "collapsed"
)

# Load external CSS file
load_css()

# State management
def initialize_session_state():
    """
    Initializes all session state variables
    - st.session_state -> object that persists across reruns (global var dict similar - maintains val)
    """
    
    # Check if "messages" key exists in session state, if not creates empty list
    # Stores full chat convo history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Store the RAG system generator instance
    # None initially - populated when system initializes
    if "generator" not in st.session_state:
        st.session_state.generator = None
    
    # Track whether initializing system has been attempted
    # Prevents multiple initialization attempts
    if "initialization_attempted" not in st.session_state:
        st.session_state.initialization_attempted = False
    
    # Track current system status
    #  -> "not_initialized", "ready", or "error"
    if "system_status" not in st.session_state:
        st.session_state.system_status = "not_initialized"

def initialize_rag_system():
    """
    Initialize the RAG system with error handling
    Returns bool: True if initialization successful, False o.w
    """
    
    # Does not try again if success has been achieved
    if st.session_state.initialization_attempted and st.session_state.generator is not None:
        return True
    
    # Attempting initialization mark
    st.session_state.initialization_attempted = True
    
    # Initialize the system
    try:
        # st.spinner() --> loading spinner w/ custom message
        with st.spinner("Initializing UCSB Engineering Assistant..."):
            # Create RAG system instance
            st.session_state.generator = GeminiResponseGenerator()
            # Update status to ready
            st.session_state.system_status = "ready"
            return True
    
    except Exception as e:
        # If initialization fails, update status and store error
        st.session_state.system_status = "error"
        st.session_state.initialization_error = str(e)
        return False

# Display functions
def display_system_status():
    """Display system status and diagnostics"""
    
    if st.session_state.system_status == "ready":
        # st.success() displays green success message
        st.success("UCSB Engineering Assistant is ready!")
        
        # st.expander() creates collapsible section
        # expanded=False --> starts collapsed
        with st.expander("System Information", expanded=False):
            try:
                # Get system metrics from the generator
                doc_count = st.session_state.generator.collection.count()
                collection_name = st.session_state.generator.collection_name
                
                # st.columns() creates side by side cols.
                # Returns list of column objs
                col1, col2, col3 = st.columns(3)
                
                # st.metric() displays metric w/ label and value
                with col1:
                    st.metric("Documents Loaded", doc_count)
                with col2:
                    st.metric("Collection", collection_name)
                with col3:
                    st.metric("Model", "Gemini 1.5 Flash")
            except Exception as e:
                # st.warning() displays yellow warning message
                st.warning(f"Could not retrieve system metrics: {e}")
    
    elif st.session_state.system_status == "error":
        # st.error() displays red error message
        st.error("Failed to initialize the system")
        
        with st.expander("Error Details", expanded=True):
            # Markdown w/ HTML for custom styling
            st.markdown(f"""
            <div class="error-message">
                <strong>Error:</strong> {st.session_state.get('initialization_error', 'Unknown error')}
            </div>
            """, unsafe_allow_html=True)
            
            # Display troubleshooting info
            st.markdown("""
            **Possible Solutions:**
            1. Make sure you've run the embeddings script to create the document collection
            2. Check that your `.env` file contains a valid `GOOGLE_API_KEY`
            3. Verify all required packages are installed:
               ```bash
               pip install streamlit google-generativeai chromadb python-dotenv
               ```
            4. Ensure the refactored structure is complete:
               - `src/core/response_generator.py` exists
               - `src/core/embeddings.py` exists  
               - `src/core/rag_pipeline.py` exists
            5. Check that imports within moved files are updated for new structure
            """)

def display_chat_interface():
    """
    Main chat interface for user input and response generation
    - User input via chat_input
    - Response generation
    - Message display and storage
    """
    
    # st.chat_input() creates chat input box at bottom
    # := operator assigns input to 'prompt' AND checks if truthy
    # Standard streamlit pattern handling i think 
    if prompt := st.chat_input("Ask about UCSB College of Engineering..."):
        
        # Add user message to session state (convo history)
        # Each message is a dict w/ role and content
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # st.chat_message() creates chat bubble w/ avatar
        # Context manager --> everything inside appears in the bubble
        with st.chat_message("user"):
            st.write(prompt)  # Displays user's message
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            # Loading indicator while processing
            with st.spinner("Thinking..."):
                try:
                    # Call RAG system to generate response
                    result = st.session_state.generator.generate_response(prompt)
                    
                    # Check if response generation was successful
                    if result['success']:
                        # Display main response text
                        st.write(result['response'])
                        
                        # Display sources if available and non empty
                        if result.get('sources') and len(result['sources']) > 0:
                            # Expandable section for sources
                            with st.expander("Sources Consulted", expanded=False):
                                # Loop through sources w/ enumeration
                                for i, source in enumerate(result['sources'], 1):
                                    # Custom HTML styling for each source
                                    st.markdown(f"""
                                    <div class="source-item">
                                        <strong>{i}.</strong> {source['title']}<br>
                                        <small>Department: {source['department']} | Type: {source['type'].title()}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Display additional metrics if available
                        if result.get('context_docs_used'):
                            # st.caption() displays small gray text
                            st.caption(f"Consulted {result['context_docs_used']} documents")
                    
                    else:
                        # Handle failed response generation
                        st.error("Failed to generate response. Please try again.")
                        st.caption(f"Error details: {result.get('response', 'Unknown error')}")
                    
                    # Add assistant response to conversation history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result['response'],
                        "sources": result.get('sources', []),  # Include sources in history
                        "success": result['success']
                    })
                
                except Exception as e:
                    # Handle unexpected errors during response generation
                    error_msg = f"An unexpected error occurred: {str(e)}"
                    st.error(f"{error_msg}")
                    # Still add error to convo history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "success": False
                    })

def display_chat_history():
    """
    Display existing chat history from session state
    Replays all previous messages in the convo, keeps visual chat flow
    """
    
    # Loop through all stored messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            # Display user message
            with st.chat_message("user"):
                st.write(message["content"])
        
        elif message["role"] == "assistant":
            # Display assistant message
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Display sources for successful responses if exist
                if message.get("success", False) and message.get("sources"):
                    with st.expander("Sources", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"""
                            <div class="source-item">
                                <strong>{i}.</strong> {source['title']}<br>
                                <small>{source['department']} | {source['type'].title()}</small>
                            </div>
                            """, unsafe_allow_html=True)

def display_sidebar():
    """
    Sidebar w/ additional features and controls
    st.sidebar creates sidebar panel on the left side
    """

    with st.sidebar:
        st.markdown("## 🛠️ UCSB Engineering Chatbot Assistant")
        
        # Display current system status
        if st.session_state.system_status == "ready":
            st.success("System Online")
        elif st.session_state.system_status == "error":
            st.error("System Error")
        else:
            st.warning("Initializing...")
        
        # st.markdown("---") --> horizontal divider line
        st.markdown("---")
        
        # Sample questions section - NOW WRAPPED IN EXPANDER
        with st.expander("Sample Questions"):
            # Use st.columns() to force consistent layout
            col1 = st.columns(1)[0]  # Single full-width column
            with col1:
                sample_questions = [
                    "What CS courses are available?",
                    "Tell me about the ME program",
                    "Prerequisites for ECE 10A?",
                    "Computer Engineering requirements",
                    "Materials department courses"
                ]
                
                for question in sample_questions:
                    if st.button(
                        question,
                        use_container_width=True,  # Forces full-width like Chat Controls
                        key=f"sample_{hash(question)}",
                        help="Click to ask this question",
                    ):
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.rerun()
        
        st.markdown("---")
        
        # Chat controls section
        with st.expander("Chat Controls"):
            # Clear chat history button
            if st.button("Clear Chat History"):
                st.session_state.messages = []  # Empty the messages list
                st.rerun()  # Rerun to show changes
            
            # Restart system button
            if st.button("Restart System"):
                # Reset all session state by deleting all keys
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            
            # Export chat functionality
            if len(st.session_state.messages) > 0:  # Only shows if messages are there
                if st.button("Export Chat"):
                    # Create export data structure
                    chat_export = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "messages": st.session_state.messages
                    }
                    # st.download_button() creates download button
                    st.download_button(
                        "Download Chat History",
                        data=str(chat_export),  # Convert to string for download
                        file_name=f"ucsb_chat_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"  # MIME type for text file
                    )
        
        st.markdown("---")
        
        # About section
        with st.expander("About"):
            st.markdown("""
            **UCSB Engineering Chatbot Assistant** uses RAG (Retrieval-Augmented Generation) 
            to answer questions about UCSB College of Engineering courses, programs, 
            and departments.
            
            **Features:**
            - Specific course information and prerequisites  
            - Department and respective program details
            - Accurate and credible source-backed responses
            - Semantic search across documents
            
            **Tech Stack:**
            - Frontend: Streamlit
            - Backend: Python & LangChain  
            - LLM: Google Gemini 1.5 Flash
            - Vector DB: ChromaDB
            - Embeddings: Google Embedding-001
            """)
            
# Main application
def main():
    """
    Entire app flow
    1. Initializes the app state
    2. Sets up the UI header
    3. Initializes the RAG system
    4. Displays the appropriate interface based on system status
    """
    
    # Initialize session state vars
    initialize_session_state()
    
    # App header w/ custom HTML styling
    st.markdown('<h1 class="main-header">🛠️ UCSB Engineering Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask any questions about UCSB College of Engineering. Courses, programs, and departments information! </p>', unsafe_allow_html=True)
    
    # Initialize RAG system and get status
    system_ready = initialize_rag_system()
    
    # Display system status (success/error messages)
    display_system_status()
    
    # Only show chat interface if system is ready
    if system_ready and st.session_state.system_status == "ready":
        
        # Display sidebar w/ controls and sample questions
        display_sidebar()
        
        # Main chat interface section
        st.markdown("### Chat Interface")
        
        # Display existing conversation history
        display_chat_history()
        
        # Handle new user input and responses
        display_chat_interface()
    
    else:
        # System not ready - show setup instructions
        st.markdown("### Getting Started")
        
        with st.expander("Setup Instructions", expanded=True):
            st.markdown("""
            To get the UCSB Engineering Assistant running:
            
            1. **Environment Setup:**
               ```bash
               # Install required packages
               pip install streamlit google-generativeai chromadb python-dotenv
               ```
            
            2. **API Configuration:**
               - Create a `.env` file in your project root
               - Add your Google API key: `GOOGLE_API_KEY=your_api_key_here`
            
            3. **Document Processing:**
               ```bash
               # Run the embedding script to process UCSB documents
               python src/core/embeddings.py
               ```
            
            4. **Launch Application:**
               ```bash
               # Start the Streamlit app
               streamlit run app.py
               ```
            """)

# Only runs main() if this file is executed directly
if __name__ == "__main__":
    main()