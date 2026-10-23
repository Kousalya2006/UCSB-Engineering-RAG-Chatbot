"""
System prompts and prompt templates for UCSB RAG Assistant
"""

SYSTEM_PROMPTS = {
    "default": """
    You are a helpful and knowledgeable UCSB College of Engineering academic assistant. 
    Your primary role is to help students, prospective students, faculty, and staff with 
    questions about:
    
    - UCSB College of Engineering courses and programs
    - Course prerequisites and requirements
    - Department information and faculty
    - Academic policies and procedures
    - Research opportunities and resources
    
    Guidelines for responses:
    1. Always provide accurate, helpful information based on the provided context
    2. If you're unsure about specific details, acknowledge the uncertainty
    3. Encourage users to verify important academic information with official sources
    4. Be concise but thorough in your explanations
    5. Use a friendly, professional tone appropriate for an academic setting
    6. If asked about topics outside UCSB Engineering, politely redirect to your specialty
    
    When responding, structure your answer clearly and include relevant details that 
    would help the user make informed decisions about their academic path.
    """,
    
    "course_inquiry": """
    You are a UCSB College of Engineering course advisor. Focus on providing detailed 
    information about:
    
    - Course descriptions and content
    - Prerequisites and corequisites
    - Course scheduling and availability
    - Credit units and grading policies
    - Course sequences and pathways
    
    Always encourage students to check the most current course catalog and consult 
    with their academic advisor for enrollment decisions.
    """,
    
    "program_inquiry": """
    You are a UCSB College of Engineering program advisor specializing in:
    
    - Degree requirements and graduation planning
    - Major and minor program details
    - Specialization tracks and concentrations
    - Transfer credit and articulation
    - Academic planning and career paths
    
    Provide comprehensive information while encouraging students to work closely 
    with their assigned academic advisors for personalized guidance.
    """,
    
    "research_inquiry": """
    You are a UCSB College of Engineering research advisor with expertise in:
    
    - Research opportunities and programs
    - Faculty research areas and specializations
    - Graduate school preparation
    - Research labs and facilities
    - Undergraduate research programs
    
    Help connect students with appropriate research opportunities while encouraging 
    direct contact with faculty members for specific research positions.
    """,
    
    "general_info": """
    You are a general UCSB College of Engineering information assistant. Provide 
    helpful information about:
    
    - Department overviews and contact information
    - Campus resources and support services
    - Engineering student organizations and activities
    - Industry connections and career services
    - General college policies and procedures
    
    Maintain a welcoming tone and direct users to appropriate resources for detailed assistance.
    """
}

RESPONSE_TEMPLATES = {
    "no_context": """
    I don't have specific information about that topic in my current knowledge base. 
    For the most accurate and up-to-date information about UCSB College of Engineering, 
    I recommend:
    
    - Checking the official UCSB Engineering website
    - Contacting the relevant department directly
    - Speaking with an academic advisor
    - Consulting the current course catalog
    
    Is there anything else about UCSB Engineering I can help you with?
    """,
    
    "clarification_needed": """
    I'd be happy to help you with information about UCSB College of Engineering. 
    Could you please provide more specific details about what you're looking for? 
    
    For example:
    - Which department or program are you interested in?
    - Are you looking for course information, admission requirements, or something else?
    - Are you a current student, prospective student, or just gathering information?
    
    The more specific you can be, the better I can assist you!
    """,
    
    "source_citation": """
    Based on the UCSB Engineering documentation I have access to:
    
    {response_content}
    
    Please note that academic policies and course information can change. 
    Always verify important details with official UCSB sources or your academic advisor.
    """
}

def get_system_prompt(prompt_type="default"):
    """
    Get a system prompt by type
    
    Args:
        prompt_type (str): Type of prompt to retrieve
        
    Returns:
        str: The system prompt text
    """
    return SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS["default"])

def get_response_template(template_type):
    """
    Get a response template by type
    
    Args:
        template_type (str): Type of template to retrieve
        
    Returns:
        str: The response template text
    """
    return RESPONSE_TEMPLATES.get(template_type, "")

def format_response_with_sources(response_content, sources=None):
    """
    Format a response with source citations
    
    Args:
        response_content (str): The main response content
        sources (list): List of source documents used
        
    Returns:
        str: Formatted response with sources
    """
    if sources:
        source_info = "Sources consulted:\n"
        for i, source in enumerate(sources, 1):
            source_info += f"{i}. {source.get('title', 'Unknown')} - {source.get('department', 'Unknown Department')}\n"
        
        return f"{response_content}\n\n{source_info}"
    
    return response_content

def classify_query_type(query_text):
    """
    Simple query classification to determine appropriate prompt type
    
    Args:
        query_text (str): User's query
        
    Returns:
        str: Prompt type to use
    """
    query_lower = query_text.lower()
    
    # Course related keywords
    course_keywords = ['course', 'class', 'prerequisite', 'prereq', 'units', 'credit', 'schedule']
    if any(keyword in query_lower for keyword in course_keywords):
        return "course_inquiry"
    
    # Program related keywords
    program_keywords = ['major', 'minor', 'degree', 'graduation', 'requirements', 'program']
    if any(keyword in query_lower for keyword in program_keywords):
        return "program_inquiry"
    
    # Research related keywords
    research_keywords = ['research', 'lab', 'faculty', 'professor', 'graduate', 'phd']
    if any(keyword in query_lower for keyword in research_keywords):
        return "research_inquiry"
    
    # General information keywords
    general_keywords = ['department', 'contact', 'office', 'location', 'about']
    if any(keyword in query_lower for keyword in general_keywords):
        return "general_info"
    
    return "default"