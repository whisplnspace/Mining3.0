import os
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from huggingface_hub import login
from sentence_transformers import SentenceTransformer, util

# Authenticate with Hugging Face
login("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")

# Streamlit page setup
st.set_page_config(page_title="MinerlexAI", page_icon="🤖", layout="wide")
load_dotenv()

# Gemini API Setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config={"temperature": 0.7, "top_p": 0.95, "top_k": 40, "max_output_tokens": 8192, "response_mime_type": "text/plain"},
)

# Embedding and RAG setup
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_store = None
rag_model = Ollama(model="deepseek-r1:1.5b")

# Prompt template for answer generation
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# PDF Processing
@st.cache_data(show_spinner=False)
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name
    docs = PDFPlumberLoader(tmp_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True).split_documents(docs)
    return chunks

def index_docs(documents):
    global vector_store
    vector_store = FAISS.from_documents(documents, embeddings)

def retrieve_docs(query):
    if vector_store is None:
        st.warning("📂 Please upload and process a PDF to enable contextual answers.")
        return []
    return vector_store.similarity_search(query)

# Repeated Prompting + Reranking
def generate_and_refine_answer(question, context, attempts=3):
    prompt = ChatPromptTemplate.from_template(template)
    candidate_answers = [(prompt | rag_model).invoke({"question": question, "context": context}) for _ in range(attempts)]

    reranker_model = SentenceTransformer("all-MiniLM-L6-v2")
    q_embed = reranker_model.encode(question, convert_to_tensor=True)
    scores = [util.cos_sim(q_embed, reranker_model.encode(ans, convert_to_tensor=True))[0].item() for ans in candidate_answers]
    best_index = scores.index(max(scores))
    return candidate_answers[best_index], candidate_answers, scores

# Translation & TTS
LANGUAGE_CODES = {
    "English": "en_XX",
    "Hindi (हिन्दी)": "hi_IN",
    "Bengali (বাংলা)": "bn_IN",
    "Tamil (தமிழ்)": "ta_IN",
    "Telugu (తెలుగు)": "te_IN",
    "Marathi (मराठी)": "mr_IN",
    "Gujarati (ગુજરાતી)": "gu_IN",
    "Malayalam (മലയാളം)": "ml_IN",
    "Kannada (ಕನ್ನಡ)": "kn_IN",
    "Odia (ଓଡ଼ିଆ)": "or_IN",
    "Urdu (اردو)": "ur_PK",
    "Assamese (অসমীয়া)": "as_IN"
}

GTTS_LANGUAGE_CODES = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "Bengali (বাংলা)": "bn",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Marathi (मराठी)": "mr",
    "Gujarati (ગુજરાતી)": "gu",
    "Malayalam (മലയാളം)": "ml",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Urdu (اردو)": "ur",
    "Assamese (অসমীয়া)": "as"
}

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
translator_model = MBartForConditionalGeneration.from_pretrained(model_name)

def translate_text(text, target_language):
    if target_language == "English":
        return text
    try:
        model_inputs = tokenizer(text, return_tensors="pt")
        tokens = translator_model.generate(**model_inputs, forced_bos_token_id=tokenizer.lang_code_to_id[LANGUAGE_CODES.get(target_language, "en_XX")])
        return tokenizer.decode(tokens[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

def generate_audio(text, language):
    try:
        tts = gTTS(text, lang=GTTS_LANGUAGE_CODES.get(language, "en"))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
        return temp_audio.name
    except Exception as e:
        st.error(f"Audio generation error: {e}")
        return None

# Dark Theme Styling and Header
st.markdown("""
    <style>
        body { background-color: #121212; color: white; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stButton > button { width: 100%; padding: 0.6em; font-size: 1.05em; background-color: #1f1f1f; color: white; border: 1px solid #444; }
        .stTextInput > div > input, .stSelectbox > div > div { background-color: #1f1f1f; color: white; border: 1px solid #444; }
        h1, h3, .stMarkdown { color: #90caf9; }
        .stSidebar { background-color: #1c1c1c !important; color: white; }
    </style>
    <div style='text-align:center; padding-bottom: 10px;'>
        <h1>🤖 MinerlexAI</h1>
        <h3>🔎 Revolutionizing Mining Law with AI-Powered Precision</h3>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload your mining law PDF", type="pdf")
    if uploaded_file:
        with st.spinner("📖 Extracting and indexing document..."):
            chunks = process_pdf(uploaded_file)
            index_docs(chunks)
            st.success("✅ PDF processed successfully!")
            st.markdown(f"**📝 File:** `{uploaded_file.name}`")
            st.markdown(f"**📚 Chunks Indexed:** `{len(chunks)}`")

st.markdown("### 🌐 Select Preferred Language")
selected_language = st.selectbox("", list(LANGUAGE_CODES.keys()), index=0)

st.markdown("---")
st.markdown("### 💬 Ask Your Legal Question")
user_input = st.text_input("Type your query or use the microphone 🎙️")
col1, col2 = st.columns(2)
with col1:
    send_clicked = st.button("📩 Submit Query")
with col2:
    speak_clicked = st.button("🎙️ Speak")

if speak_clicked:
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎧 Listening... Speak clearly.")
        try:
            audio = recognizer.listen(source, timeout=5)
            user_input = recognizer.recognize_google(audio)
            st.success(f"🗣️ You said: {user_input}")
        except sr.UnknownValueError:
            st.error("❌ Could not understand audio.")
        except sr.RequestError:
            st.error("⚠️ API unavailable. Check your internet connection.")

if user_input and (send_clicked or speak_clicked):
    with st.spinner("🧠 Generating AI-powered legal insights..."):
        docs = retrieve_docs(user_input)
        if docs:
            context = "\n\n".join([doc.page_content for doc in docs])
            best_answer, candidates, scores = generate_and_refine_answer(user_input, context, attempts=3)
        else:
            best_answer = model.start_chat(history=[]).send_message(user_input).text
            candidates = [best_answer]
            scores = [1.0]

        translated = translate_text(best_answer, selected_language)

    st.markdown("---")
    st.subheader("💡 Best AI Answer")
    st.success(best_answer)

    st.subheader(f"🌐 Translated Answer ({selected_language})")
    st.info(translated)

    st.subheader("🔊 Listen to Answer")
    audio_path_final = generate_audio(best_answer, "en")
    audio_path_translated = generate_audio(translated, selected_language)

    col3, col4 = st.columns(2)
    with col3:
        if audio_path_final:
            st.audio(audio_path_final, format="audio/mp3", start_time=0)
            st.caption("Original Answer in English")
    with col4:
        if audio_path_translated:
            st.audio(audio_path_translated, format="audio/mp3", start_time=0)
            st.caption(f"Translated Answer in {selected_language}")

    with st.expander("🔍 Show All Candidate Answers and Ranks"):
        for i, ans in enumerate(candidates):
            st.markdown(f"**Candidate {i+1} (Score: {scores[i]:.2f})**")
            st.markdown(ans)
            st.markdown("---")
