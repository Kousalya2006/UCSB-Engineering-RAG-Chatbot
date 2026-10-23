   # UCSB College of Engineering RAG Chatbot

   A conversational artificial intelligence assistant specifically designed for UCSB College of Engineering students, faculty, and prospective students. This RAG (Retrieval-Augmented Generation) chatbot provides up to date, accurate, source-backed detailed and thorough information about departments, programs, and course information.

   ## Table of Contents

   - [Why Did I Build This?](#why-did-i-build-this)
   - [Key Capabilities](#key-capabilities)
   - [Features](#features)
   - [System Architecture](#system-architecture)
   - [Data Flow](#data-flow)
   - [Demo GIFs](#demo-gifs)
   - [Technology Stack](#technology-stack)
   - [Project Structure](#project-structure)
   - [Important File Responsibilities](#important-file-responsibilities)
   - [Quick Start](#quick-start)
   - [Requirements](#requirements)
   - [Configuration](#configuration)
   - [System Requirements](#system-requirements)
   - [Usage](#usage)
   - [Testing](#testing)
   - [Performance](#performance)
   - [Troubleshooting Configurations](#troubleshooting-configurations)
   - [Contributing](#contributing)
   - [License](#license)
   - [Author](#author)
   - [Acknowledgments & References](#acknowledgments--references)

   ## Why Did I Build This?

   I’m deeply interested in artificial intelligence, large language models, and the rapidly evolving landscape of generative AI tools. I built this personal project to learn and sharpen my technical skills and explore how cutting-edge AI can be applied in a meaningful real world setting, specifically one I’m part of every day: the UCSB community.

   I noticed that information about UCSB departments, programs, and courses is often fragmented across multiple websites, buried in PDFs, or difficult to navigate for students and prospective applicants. I wanted to create something that not only streamlines this experience but also shows the potential of retrieval augmented generation systems in higher education.

   On the frontend, I chose Streamlit as a way to branch out and learn a new framework for building interactive Python applications, since I’ve already gained experience with React, HTML, and CSS. On the backend, I built a full RAG pipeline using ChromaDB, Google Gemini, LangChain and custom scraping (JavaSciprt) and preprocessing scripts to extract, clean, embed, and retrieve information from UCSB’s official academic catalog.

   This chatbot is more than just a technical demo. It is a proof of concept for how modern AI tools like Google Gemini can make academic information more accessible, verifiable, and helpful. This is also a step forward in my journey toward pursuing opportunities and knowledge in AI, machine learning, computer science, and data science.

   ### Key Capabilities 
   - **Source-Verified Responses**: Reliability and citation features for this RAG chatbot. Credible source-backed responses for each query
   - **Cross-Departmental Knowledge**: Understanding of connections across varying engineering disciplines. Numerous departments, programs, and courses in UCSB College of Engineering
   - **Intelligent Course Discovery**: Semantic search capabilities for finding relevant courses and programs
   - **Academic Information Retrieval**: Comprehensive access to official and up to date UCSB engineering information
   - **Advanced System Features**: Comprehensive chat history management with export functionality, real-time system monitoring and controls, and detailed source verification with expandable document attribution

   ## Features

   ### Core Functionality
   - **Intelligent Q&A**: Natural language queries about UCSB College of Engineering departments, programs, and courses
   - **Source Attribution**: All responses backed by most recent (2024 - 2025) official UCSB documents and catalog
   - **Real-time Chat**: Interactive Streamlit interface with full conversation history
   - **Semantic Search**: Advanced document retrieval using vector embeddings
   - **Multi-Department Support**: Covers all engineering departments (CS, ECE, ME, etc.)

   ### User Experience
   - **Responsive Design**: UCSB branded user interface with school colors and styling
   - **Full Chat History**: Persistent conversation storage within sessions
   - **Export Functionality**: Download chat transcripts for later reference
   - **Sample Questions**: Quick start prompts for common queries for user
   - **System Diagnostics**: Real-time status monitoring and error handling

   ### Technical Features
   - **RAG Pipeline**: Combines retrieval and generation for accurate responses
   - **Vector Database**: ChromaDB for efficient similarity search
   - **Modern LLM**: Google Gemini 1.5 Flash for high-quality responses with little to no relative cost
   - **Modular Architecture**: Clean separation of concerns with organized codebase
   - **Error Handling**: Graceful failures and thorough error management

   ### Advanced Features

   #### Chat History Management
   - View complete conversation history
   - Export chat transcripts
   - Clear history for new sessions

   #### System Controls
   - Monitor system status
   - Restart RAG pipeline
   - View document metrics

   #### Source Verification
   - All responses include specific source documents
   - Expandable source details
   - Department and document type attribution


   ## System Architecture

   ```
   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │   User Browser  │    │  UCSB Websites  │    │   Google APIs   │
   │                 │    │                 │    │                 │
   │ • Chat Interface│    │ • General Cat.  │    │ • Gemini 1.5    │
   │  • UCSB Styling │    │ • Dept. Pages   │    │ • Embedding-001 │
   │  • Export/Clear │    │ • Course Info   │    │ • API Keys      │
   └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
            │                      │                      │
            │ HTTP                 │ Web Scraping         │ API Calls
            │                      │ (6 Concurrent)       │
            ▼                      ▼                      ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                    Streamlit Web App (app.py)                   │
   │                                                                 │
   │  • Session Management    • Chat Interface     • Error Handling  │
   │  • CSS Loading          • History Display    • System Status    │
   │  • UI Components        • User Input         • Sidebar Controls │
   └─────────────────────────┬───────────────────────────────────────┘
                           │
                           │ Function Calls
                           │
                           ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                    Core RAG Pipeline                            │
   │                                                                 │
   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
   │  │ response_       │  │ rag_pipeline.py │  │ embeddings.py   │  │
   │  │ generator.py    │  │                 │  │                 │  │
   │  │                 │  │ • Pipeline Test │  │ • Doc Processing│  │
   │  │ • RAG Main Flow │  │ • Quality Tests │  │ • Vector Gen    │  │
   │  │ • Doc Retrieval │  │ • E2E Testing   │  │ • ChromaDB Init │  │
   │  │ • LLM Generation│  │ • Test Scoring  │  │ • Embedding API │  │
   │  │ • Context Format│  │ • Validation    │  │ • Doc Chunking  │  │
   │  │ • Error Handling│  │ • Batch Testing │  │ • Retry Logic   │  │
   │  │ • Chat Interface│  │ • Results Export│  │ • Rate Limiting │  │
   │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
   └─────────────────────────┬───────────────────────────────────────┘
                           │
                           │ Document Storage & Retrieval
                           │
                           ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                        Data Layer                               │
   │                                                                 │
   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
   │  │   ChromaDB      │  │  Data Pipeline  │  │  Configuration  │  │
   │  │                 │  │                 │  │                 │  │
   │  │ • Vector Store  │  │ • data_scraper  │  │ • prompts.py    │  │
   │  │ • Similarity    │  │ • data_cleaner  │  │ • settings.py   │  │
   │  │ • Collections   │  │ • data_processor│  │ • .env file     │  │
   │  │ • Persistence   │  │ • data_validator│  │ • API Keys      │  │
   │  │ • Query Search  │  │ • Quality Ctrl  │  │ • Model Config  │  │
   │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
   └─────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────┐
   │                    Testing & Utilities Layer                    │
   │                                                                 │
   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
   │  │   API Testing   │  │ Embeddings Test │  │ Quality Metrics │  │
   │  │                 │  │                 │  │                 │  │
   │  │ • api_tester.py │  │ • Embedding Sim │  │ • Response Eval │  │
   │  │ • Connection    │  │ • Retrieval Acc │  │ • Source Track  │  │
   │  │ • Validation    │  │ • Context Format│  │ • Performance   │  │
   │  │ • Health Checks │  │ • End-to-End    │  │ • Test Reports  │  │
   │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
   └─────────────────────────────────────────────────────────────────┘
   ```

   ## Data Flow

   ```
   ┌─────────────────────────────────────────────────────────────────┐
   │                         Data Flow                               │
   └─────────────────────────────────────────────────────────────────┘

   1. User Input → Streamlit Interface

   2. Query Processing → response_generator.py (Main RAG Pipeline)
      ├── Vector Search → ChromaDB Collection Query
      ├── Document Retrieval → Relevant Context Assembly  
      └── Context Formatting → Structured LLM Input

   3. LLM Generation → Gemini API with System Prompts

   4. Response Assembly → Source Attribution & Error Handling

   5. UI Display → Chat Interface with Source Citations

   6. Data Processing Pipeline → Automated Quality Control
      ├── Raw Data Scraping → data_scraper.js (Concurrent Engineering Focus)
      ├── Content Cleaning → data_cleaner.py (Text Processing & Filtering)
      ├── Document Processing → data_processor.py (Structure & Chunking)
      ├── Validation → data_validator.py (Quality Assurance)
      └── Embedding Generation → embeddings.py (Vector Creation)

   7. Quality Assurance → rag_pipeline.py Testing Suite
      ├── Embedding Quality Tests
      ├── Retrieval Accuracy Tests
      ├── Response Generation Tests
      └── End-to-End Pipeline Validation

   ```

   ## Demo GIFs

   ![demo2](https://github.com/user-attachments/assets/ad92a5e9-cce7-44e8-8eef-726006d0b8f6)

   ![demo1](https://github.com/user-attachments/assets/7555bcd6-0744-4a80-a063-fd4a8d943360)

   ## Technology Stack

   - **Frontend**: Streamlit
   - **Backend**: Python & LangChain  
   - **LLM**: Google Gemini 1.5 Flash
   - **Vector DB**: ChromaDB
   - **Embeddings**: Google Embedding-001

   ## Project Structure

   ```
   UCSB-RAG-CHATBOT/
   ├── src/
   │   ├── config/
   │   │   ├── prompts.py
   │   │   └── settings.py
   │   ├── core/
   │   │   ├── embeddings.py
   │   │   ├── rag_pipeline.py
   │   │   └── response_generator.py
   │   ├── data/
   │   │   ├── data_cleaner.py
   │   │   ├── data_processor.py
   │   │   ├── data_scraper.js
   │   │   └── data_validator.py
   │   └── utils/
   │       └── api_tester.py
   ├── styles/
   │   └── app.css
   ├── tests/
   │   ├── test_embeddings.py
   │   └── test_results.json
   ├── .env
   ├── .gitignore
   ├── app.py
   ├── LICENSE
   ├── package-lock.json
   ├── package.json
   ├── README.md
   └── requirements.txt
   ```

   ## Important File Responsibilities

   ### embeddings.py
   - GeminiDocumentEmbedder class
   - Text chunking with overlap (1000 chars, 100 overlap)
   - ChromaDB collection management
   - Batch processing with rate limiting
   - Document metadata preservation
   - Retrieval testing and validation

   ### response_generator.py
   - GeminiResponseGenerator class (Main RAG Implementation)
   - Query embedding with retry logic
   - Vector similarity search and ranking
   - Dynamic context formatting by document type
   - Multi-mode interfaces (chat, batch, single query)
   - Source tracking and citation generation

   ### rag_pipeline.py
   - RAGPipelineTester class (Comprehensive Testing)
   - Prerequisite validation (API keys, collections)
   - Multi-criteria quality evaluation
   - Cosine similarity calculations
   - End-to-end pipeline validation
   - JSON test result export

   ### data_cleaner.py
   - UCSBDataCleaner class (Content Quality Control)
   - Text normalization and cleaning algorithms
   - Gibberish detection and filtering
   - UCSB-specific content extraction
   - Course/program/department structure parsing
   - Quality metrics and reporting

   ### data_processor.py
   - UCSBDataProcessor class (Document Transformation)
   - Hierarchical data flattening
   - Document structure standardization
   - Metadata generation and preservation
   - Content formatting for embedding

   ### data_validator.py
   - DataValidator class (Pre-embedding Quality Assurance)
   - Structure validation and integrity checks
   - Content analysis and statistics
   - Embedding cost estimation
   - Readiness verification for RAG pipeline

   ## Quick Start

   ### Prerequisites

   - Python 3.8 or higher (Backend and Streamlit)
   - Node.js 16 or higher (Web Scraper)
   - Google API key (Gemini)
   - ChromaDB
   - LangChain

   ### Installation

   1. **Clone the Repository**
      ```bash
      git clone https://github.com/yourusername/ucsb-rag-chatbot.git
      cd ucsb-rag-chatbot
      ```

   2. **Set Up Virtual Environment**
      ```bash
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      ```

   3. **Install Dependencies**
      ```bash
      pip install -r requirements.txt
      ```

   4. **Configure Environment**
      ```bash
      cp .env.example .env
      # Edit .env and add your GOOGLE_API_KEY
      ```

   5. **Process Documents**
      ```bash
      python src/core/embeddings.py
      ```

   6. **Launch Application**
      ```bash
      streamlit run app.py
      ```

   The application will be available at `http://localhost:8501`

   ## Requirements

   ### Dependencies
   ```
   streamlit>=1.28.0
   python-dotenv>=1.0.0
   chromadb>=0.4.15
   langchain>=0.0.350
   langchain-community>=0.0.10
   google-generativeai>=0.3.0
   pandas>=2.0.0
   numpy>=1.24.0
   requests>=2.31.0
   beautifulsoup4>=4.12.0
   lxml>=4.9.0
   pathlib2>=2.3.0
   ```

   ## Configuration

   ### Environment Variables

   Create a `.env` file in the project root:

   ```env
   # Required
   GOOGLE_API_KEY=your_google_api_key_here
   ```

   ## System Requirements

   **Hardware Requirements:**
   - **CPU**: Sufficient enough for embedding data
   - **RAM**: Sufficient enough for embedding data
   - **Storage**: Sufficient amount for documents and embeddings 

   ## Usage

   **Basic Queries**
   - **Departments**: "What programs does the Computer Science department offer?"
   - **Courses**: "Tell me about CS 156 Machine Learning"
   - **Requirements**: "What are the prerequisites for ECE 153A?"
   - **Programs**: "How do I apply to the Materials Engineering graduate program?"

   **Chat Interface**
   - Type questions in natural language
   - View conversation history in the sidebar
   - Export chat transcripts using the download button
   - Clear history to start fresh sessions

   **Sample Questions**
   - Click any sample question to automatically populate the input
   - Modify sample questions to suit your specific needs

   **Source Verification**
   - All responses include citations from official UCSB documents
   - Click "Show Sources" to view document details
   - Sources are linked to specific departments and document types

   **Tips for Best Results**
   - Be specific about departments, courses, or programs
   - Ask follow-up questions for clarification
   - Use course codes (e.g., "CS 156") for precise information
   - Combine multiple topics (e.g., "CS courses related to AI")

   ## Testing

   ### Run Test Suite
   ```bash
   # All tests
   pytest

   # Specific test categories
   pytest tests/test_embeddings.py
   pytest src/core/rag_pipeline.py

   ```

   ### Manual Testing
   ```bash
   # Test embeddings generation
   python src/core/embeddings.py --test

   # Test RAG pipeline
   python src/core/rag_pipeline.py --query "test query"

   # Test API connection
   python src/utils/api_tester.py
   ```

   ## Performance

   ### System Metrics
   - **Document Collection**: 314 engineering documents indexed
   - **Response Time**: Approximately 2-5 seconds per query
   - **Response Length**: 800-1,800 characters average
   - **Sources Retrieved**: 5 relevant documents per query

   ### Test Results
   **Overall System Performance**: 85.7% success rate

   **Component Performance:**
   - **Retrieval Accuracy**: 3/3 passed (100%) - Correctly identifies relevant documents
   - **Response Generation**: 4/4 passed (100%) - High-quality responses with proper formatting
   - **End-to-End Pipeline**: 4/4 passed (100%) - Complete user query processing
   - **Embedding Quality**: 1/3 passed (33%) - Some cross-domain similarity challenges

   **Cosine Similarity Analysis:**
   - **High Similarity** (>0.80): "computer science courses" vs "CS classes available" (0.819)
   - **Medium Similarity** (0.65-0.80): "mechanical engineering program" vs "ME department overview" (0.698)
   - **Cross-Domain** (<0.70): "computer science courses" vs "mechanical engineering program" (0.683) - Expected low similarity
   - **Similarity Threshold**: 0.80 for reliable semantic matching

   **Quality Scores:**
   - Machine Learning course recommendations: 1.00
   - Computer Engineering program queries: 1.00  
   - Materials department requirements: 1.00
   - Circuit-related course searches: 1.00

   ### Known Limitations
   - Cross-departmental query similarity requires optimization (e.g., CS vs ME topics)
   - Embedding model may need fine-tuning for domain-specific engineering terminology

   ### Optimization Tips
   - Pre-generate embeddings for faster startup
   - Use SSD storage for vector database
   - Monitor API rate limits
   - Cache frequent queries

   ### Production Considerations
   - Integration with UCSB GOLD system
   - Personal academic planning assistance
   - Include academic and recreational club information
   - Expand to UCSB College of Letters & Science
   - Expand to UCSB College of Creative Studies

   ## Troubleshooting Configurations

   ### Common Issues

   #### "Failed to initialize the system"
   - **Solution**: Verify Google API key in `.env`
   - **Check**: Run embeddings script first
   - **Verify**: All dependencies installed correctly

   #### "No documents found"
   - **Solution**: Run `python src/core/embeddings.py`
   - **Check**: Data folder contains UCSB documents
   - **Verify**: ChromaDB permissions and storage

   #### "Import errors"
   - **Solution**: Activate virtual environment
   - **Check**: Install all requirements
   - **Verify**: Python version compatibility

   ## Contributing

   This project was developed as a personal learning project. For future questions and/or suggestions:

   1. Open an issue describing the enhancement or bug
   2. Fork the repository and create a feature branch
   3. Follow coding standards
   4. Write tests for new functionality
   5. Update documentation as needed
   6. Submit a pull request with detailed description of changes

   ## License

   This project is open source and available under the MIT License.

   ## Author

   **Ryan Fabrick**
   - Statistics and Data Science (B.S) Student, University of California Santa Barbara
   - GitHub: [https://github.com/RyanFabrick](https://github.com/RyanFabrick)
   - LinkedIn: [www.linkedin.com/in/ryan-fabrick](https://www.linkedin.com/in/ryan-fabrick)
   - Email: ryanfabrick@gmail.com

   ## Acknowledgments & References

   - **[UCSB General Catalog](https://catalog.ucsb.edu/)** - Official academic catalog published by UCSB's Registrar's Office, containing comprehensive course descriptions, academic requirements, and program information for all colleges and majors at UCSB
   - **[Google AI Studio](https://aistudio.google.com/)** - Google's platform for experimenting with Generative AI models including the Gemini family, providing direct API access, prototyping capabilities, and billing & usage information
   - **[Google Gemini](https://gemini.google.com/)** - Google's generative AI model family offering powerful text generation capabilities, integrated with LangChain for building GenAI applications with function calling
   - **[ChromaDB](https://www.trychroma.com/)** - An open-source vector database designed for storing and querying embeddings, enabling efficient similarity search and retrieval-augmented generation workflows
   - **[LangChain](https://www.langchain.com/)** - A framework for developing LLM-powered applications by connecting with external data sources, providing chains and agents for complex reasoning and information processing
   - **[Puppeteer Community](https://pptr.dev/)** - A Node.js library providing an API for controlling Chrome/Chromium browsers, essential for web scraping and automated data collection from dynamic web pages
   - **[Streamlit Community](https://streamlit.io/)** - An open-source Python framework for building and deploying interactive web applications with seamless integration for AI and machine learning projects

   ________________________________________
   Built with ❤️ for the UCSB community

   This personal project demonstrates my passion for AI, machine learning, and real-world problem solving. As a UCSB student, I designed this chatbot to bridge the information gap many of us face when navigating academic programs, courses, and departments. It's a full-stack portfolio piece showcasing my technical skills in large language models, retrieval-augmented generation, data engineering, custom web scraping, and modern web application development.
