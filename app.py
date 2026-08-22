import streamlit as st
import os
import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Config

load_dotenv()  # for local dev

# Prefer Streamlit secrets (used on Streamlit Cloud); fall back to .env locally
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY not found. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key
st.set_page_config(page_title="📺 YouTube Analyzer Pro", layout="wide")


st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; font-weight: 700; margin-bottom: 0.5rem; text-align: center;}
    .sub-header {font-size: 1.3rem; color: #ffffff; background: linear-gradient(135deg, #1f77b4, #4a90e2); 
                 padding: 1rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; 
                  border-radius: 15px; color: white; text-align: center;}
    .analysis-btn {height: 70px; border-radius: 20px; font-weight: 700; font-size: 1.2rem; 
                   border: none; box-shadow: 0 4px 15px rgba(0,0,0,0.2);}
    .analysis-btn:hover {transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.3);}
    .chat-container {background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                     border-radius: 25px; padding: 2.5rem; margin-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# Initialize Session State
def initialize_session_state():
    defaults = {"retriever": None, "messages": [], "video_info": None, "full_transcript": "", "video_id": None, "lang": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Core Functions
@st.cache_data(ttl=3600)
def extract_video_id(url):  
    patterns = [r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)', r'youtube\.com/shorts/([^&\n?#]+)']
    for pattern in patterns:
        match = re.search(pattern, url)
        if match: return match.group(1)
    return None

@st.cache_data(ttl=3600)
def get_transcript(video_id):
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        for lang in ['hi', 'en']:
            try:
                transcript_obj = transcript_list.find_generated_transcript([lang])
                data = transcript_obj.fetch()
                return [{"text": chunk.text, "start": chunk.start, "duration": chunk.duration} for chunk in data], lang
            except: continue
        transcript_obj = transcript_list.find_generated_transcript([transcript_list[0].language_code])
        data = transcript_obj.fetch()
        return [{"text": chunk.text, "start": chunk.start, "duration": chunk.duration} for chunk in data], transcript_list[0].language_code
    except TranscriptsDisabled:
        st.error("⚠️ No transcripts available!")
        return None, None
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

@st.cache_resource
def create_vector_store(transcript_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_text])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 8})

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


FULL_SUMMARY_PROMPT = """🚨 CRITICAL: RESPOND **EXCLUSIVELY IN ENGLISH LANGUAGE ONLY**. NO HINDI. NO OTHER LANGUAGES.

You are a professional video analyst. Create COMPREHENSIVE, DETAILED English analysis.

TRANSCRIPT: {context}

**REQUIRED ENGLISH STRUCTURE:**
**📊 VIDEO STRUCTURE**
• Introduction (0:00-X:XX): [Detailed summary]
• Main Content (X:XX-Y:YY): [Detailed breakdown]  
• Examples/Demos (Y:YY-Z:ZZ): [Specific details]
• Conclusion (Z:ZZ-end): [Key takeaways]

**🔑 5 CORE CONCEPTS** (with explanations):
1. [Concept]: [Detailed explanation]
2. [Concept]: [Detailed explanation]
...

**⏱️ TIMELINE** (Key moments):
• 0:00: [Event]
• X:XX: [Event] 
• Y:YY: [Event]

**🎯 3 MAIN TAKEAWAYS**:
1. [Critical insight]
2. [Critical insight]
3. [Critical insight]

**💡 ACTIONABLE INSIGHTS**: [3 practical lessons]

ENGLISH ONLY. PROFESSIONAL FORMAT."""

KEY_TOPICS_PROMPT = """🚨 MANDATORY: **ENGLISH LANGUAGE ONLY**. NO HINDI.

Extract PRECISE main topics from transcript.

TRANSCRIPT: {context}

**REQUIRED ENGLISH FORMAT:**
**🎯 MAIN TOPICS** (Most discussed):
1. **[XX:XX-XX:XX] TOPIC NAME**: [3-4 precise subpoints from transcript]
2. **[XX:XX-XX:XX] TOPIC NAME**: [3-4 precise subpoints from transcript]

**⚡ MOST CRITICAL TOPIC**: [Why it's most important]

**📈 TOPIC FLOW**: [How topics connect]

**100% TRANSCRIPT ACCURATE. ENGLISH ONLY.**"""

QNA_PROMPT = """🚨 **ENGLISH ONLY**. NO HINDI.

TRANSCRIPT: {context}

QUESTION: {question}

**ENGLISH ANSWER**:"""

# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<h1 class="main-header">📺 YouTube Analyzer Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Transform any YouTube video into structured insights • Always English • Professional analysis</p>', unsafe_allow_html=True)

# Video Input
st.markdown("---")
input_col1, input_col2 = st.columns([3, 1])
with input_col1:
    video_url = st.text_input("🔗 **YouTube Video URL**", placeholder="https://www.youtube.com/watch?v=Gfr50f6ZBvo")
with input_col2:
    if st.button("🚀 **ANALYZE VIDEO**", type="primary", key="analyze"):
        if video_url:
            video_id = extract_video_id(video_url)
            if video_id:
                with st.spinner("🔄 Processing transcript & building AI index..."):
                    transcript_data, lang = get_transcript(video_id)
                    if transcript_data:
                        final_text = " ".join(chunk["text"] for chunk in transcript_data)
                        retriever = create_vector_store(final_text)
                        
                        st.session_state.retriever = retriever
                        st.session_state.full_transcript = final_text[:2000]
                        st.session_state.video_id = video_id
                        st.session_state.lang = lang
                        st.session_state.video_info = {"id": video_id, "lang": lang}
                        
                        st.success(f"✅ **{lang.upper()} transcript loaded & analyzed**")
                        with st.expander("📄 **Transcript Preview**", expanded=False):
                            st.caption(final_text[:500] + "..." if len(final_text) > 500 else final_text)
                    else:
                        st.error("❌ No transcript available")
            else:
                st.error("❌ **Invalid YouTube URL**")


if st.session_state.video_info:
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🌐 Language", st.session_state.video_info["lang"].upper())
        st.markdown('</div>', unsafe_allow_html=True)


if st.session_state.retriever:
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #1f77b4; margin-bottom: 2rem;'>🚀 Instant Video Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 **FULL SUMMARY**", key="summary", use_container_width=True):
            with st.chat_message("user"): 
                st.markdown("**🔬 Generate comprehensive video analysis**")
            with st.chat_message("assistant"):
                with st.spinner("🔬 Professional analysis..."):
                    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
                    
                    context = format_docs(st.session_state.retriever.invoke("summarize this video"))
                    prompt_text = FULL_SUMMARY_PROMPT.format(context=context)
                    response = llm.invoke(prompt_text).content
                    st.markdown(response)
    
    with col2:
        if st.button("🔑 **KEY TOPICS**", key="topics", use_container_width=True):
            with st.chat_message("user"):
                st.markdown("**📋 Extract structured topics**")
            with st.chat_message("assistant"):
                with st.spinner("🔍 Topic extraction..."):
                    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
                    
                    context = format_docs(st.session_state.retriever.invoke("main topics"))
                    prompt_text = KEY_TOPICS_PROMPT.format(context=context)
                    response = llm.invoke(prompt_text).content
                    st.markdown(response)
    
    with col3:
        if st.button("🗑️ **RESET**", key="reset", use_container_width=True):
            initialize_session_state()
            st.rerun()
    
    st.markdown("---")
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #1f77b4;'>💬 Detailed Q&A</h3>", unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if question := st.chat_input("💬 Ask detailed questions about the video...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("🤖 AI Analysis..."):
                llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
                
                context = format_docs(st.session_state.retriever.invoke(question))
                
                is_summary = any(word in question.lower() for word in ['summar', 'overview', 'key', 'main', 'tl;dr'])
                if is_summary:
                    prompt_text = FULL_SUMMARY_PROMPT.format(context=context)
                else:
                    prompt_text = QNA_PROMPT.format(context=context, question=question)
                
                response = llm.invoke(prompt_text).content
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        ## ✨ **What You Get**
        
        - ✅ **English analysis** (works with Hindi/English videos)
        - ✅ **Structured summaries** with timestamps  
        - ✅ **Key topics extraction**
        - ✅ **Smart Q&A** - ask anything about video
        - ✅ **Professional AI analysis**
        """)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;'>Built with ❤️ using Streamlit + LangChain + Groq</p>", unsafe_allow_html=True)
