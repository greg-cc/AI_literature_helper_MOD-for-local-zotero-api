# -*- coding: utf-8 -*- version 1.4 aistudio rebuild
import streamlit as st
import requests, json, re, os, io, csv
import xml.etree.ElementTree as ET
import fitz  # PyMuPDF
from time import sleep
from requests import RequestException
import logging
import sys
import random
import numpy as np
import math
from typing import List, Dict, Any, Optional, Iterator, Tuple
from google import genai

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s', stream=sys.stdout)

# --- GLOBAL API CONSTANTS ---
MAX_RETRIES = 5
INITIAL_DELAY_SECONDS = 2
RATE_LIMIT_STATUS_CODE = 429
ZOTERO_MAX_ABSTRACT_CHARS = 10000
MAX_TAG_LENGTH = 250
PROCESSING_BATCH_SIZE = 50
DOI_MISSING_ERROR = "⚠️ ERROR: The Digital Object Identifier (DOI) is a CRITICAL identifier. This item can still be saved to Zotero because it has an abstract and url."

# --- RATE LIMITING ---
SEMANTIC_SCHOLAR_DELAY = 0.75  # Max 1 request per 0.75 seconds (Protected)
sleepgetpapers = 0.75          # Global pacing for generic robust requests

# ============================
# CONFIG
# ============================
st.set_page_config(page_title="📚 AI Literature Helper", page_icon="🤖", layout="wide")

# --- API KEYS ---
# NOTE: In production, these should be environment variables or user inputs.
SEMANTIC_SCHOLAR_API_KEY = ""
GEMINI_API_KEY = ""
NCBI_EMAIL = "@gmail.com"
NCBI_API_KEY = ""

# --- GEMINI CLIENT SETUP ---
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    client = None
    if not GEMINI_API_KEY:
        GEMINI_CLIENT_INIT_ERROR = "Client failed to initialize: GEMINI_API_KEY is empty."
    else:
        GEMINI_CLIENT_INIT_ERROR = f"Client failed to initialize: {e.__class__.__name__}: {e}"

# --- MODEL OPTIONS ---
MODEL_OPTIONS = {
    "models/gemini-2.5-flash (Default, 7 GB)": {
        "model_id": "gemini-2.5-flash",
        "description": "Stable version. General purpose, fast. 1M TPM."
    },
    "Gemini 2.5 Pro (Complex Tasks, 15 GB)": {
        "model_id": "gemini-2.5-pro",
        "description": "Highest reasoning capacity. 2M TPM."
    },
    "models/gemini-2.5-flash-lite (Quota Fallback)": {
        "model_id": "models/gemini-2.5-flash-lite",
        "description": "High throughput, cost-efficient. 4M TPM."
    },
    "models/gemini-1.5-flash": {
        "model_id": "models/gemini-1.5-flash",
        "description": "Previous Gen Flash. Good for fallback."
    },
    "models/gemini-1.5-flash-8b": {
        "model_id": "models/gemini-1.5-flash-8b",
        "description": "Fastest, lowest latency, lower intelligence."
    },
    "models/gemini-1.5-pro": {
        "model_id": "models/gemini-1.5-pro",
        "description": "Previous Gen Pro. Strong reasoning."
    },
    "models/gemini-2.0-flash-exp": {
        "model_id": "models/gemini-2.0-flash-exp",
        "description": "Experimental 2.0. Often has separate/higher quotas."
    },
    "models/gemini-experimental": {
        "model_id": "models/gemini-experimental",
        "description": "Bleeding edge experimental features."
    },
    "models/gemini-pro-experimental": {
        "model_id": "models/gemini-pro-experimental",
        "description": "Experimental Pro version."
    },
    "models/gemini-flash-experimental": {
        "model_id": "models/gemini-flash-experimental",
        "description": "Experimental Flash version."
    },
    "gemma-2-27b-it (Open Model)": {
        "model_id": "models/gemma-2-27b-it",
        "description": "Open model (27B). Hosted on Google endpoints."
    },
    "gemma-2-9b-it (Open Model)": {
        "model_id": "models/gemma-2-9b-it",
        "description": "Open model (9B). Faster, lower memory."
    },
    "gemma-2-2b-it (Open Model)": {
        "model_id": "models/gemma-2-2b-it",
        "description": "Tiny open model. Very fast, lower quality."
    },
}

# --- DEFAULTS ---
DEFAULT_AI_TAG_CATEGORIES = [
    "Clinical Trial", "Review", "Methodology", "Case Study", "In Vivo", "In Vitro", "Phytochemistry", "Pharmacokinetics", "Toxicology"
]

# Structure: (Sentence, Enabled, Pass_State, Custom_Tag)
DEFAULT_SEMANTIC_SENTENCES = [
    ("The paper discusses researching infective agents of humans using phytochemicals.", True, True, "Infective-Agents"),
    ("The paper is primarily a review or meta-analysis.", False, False, "Review"),
    ("The paper is primarily about the biology of an organism.", False, True, "Biology"),
    ("The content focuses on a medical study testing efficacy.", True, True, "Efficacy-Study"),
    ("This content contains guidelines.", True, False, "Guidelines"),
    ("This content is about how a disease progresses and how the disease itself works.", True, False, "Disease-Mechanism"),
]

# ============================
# SESSION STATE INIT (HISTORY & CACHE)
# ============================
if "results_history" not in st.session_state:
    st.session_state.results_history = []

# New: Cache for raw API results (key: query, value: list of papers)
if "search_cache" not in st.session_state:
    st.session_state.search_cache = {}

# ============================
# CORE UTILITY FUNCTIONS
# ============================

def _request_json_with_retries(url, *, method="GET", headers=None, params=None, data=None, tries=4, timeout=40):
    """
    Robust request function with retries and exponential backoff.
    """
    delay = sleepgetpapers
    for attempt in range(1, tries + 1):
        try:
            if method == "POST":
                resp = requests.post(url, headers=headers, params=params, data=data, timeout=timeout)
            else:
                resp = requests.get(url, headers=headers, params=params, timeout=timeout)

            if 200 <= resp.status_code < 300:
                return resp.json()

            if 500 <= resp.status_code < 600:
                raise RequestException(f"Server {resp.status_code}")

            resp.raise_for_status()

        except Exception as e:
            if attempt == tries:
                raise
            logging.warning(f"NETWORK BACKOFF: Sleeping for {delay:.2f}s (Attempt {attempt}/{tries}). Error: {e}")
            sleep(delay)
            delay = min(delay * 2, 3.0)
    return {}

def handle_gemini_backoff(error_msg: str):
    """Parses Gemini error message for retry time and sleeps if found."""
    match = re.search(r"retry in\s*([\d\.]+)\s*s", error_msg, re.IGNORECASE)
    if match:
        try:
            wait_time = float(match.group(1))
            logging.warning(f"GEMINI QUOTA EXHAUSTED: Pausing for {wait_time}s as requested by API...")
            sleep(wait_time + 1.5) # Sleep required time + buffer
            return True
        except ValueError:
            pass
    
    if "429" in error_msg or "exhausted" in error_msg.lower():
        logging.warning("GEMINI 429: Generic backoff (5s).")
        sleep(5)
        return True
    return False

def clean_snippet(text):
    """Removes HTML tags and normalizes whitespace."""
    if not text:
        return ""
    text = re.sub(r'<[^<]+?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def dedupe_results(results):
    """Deduplicates based on DOI (primary) and normalized Title (secondary)."""
    seen_dois = set()
    seen_titles = set()
    unique = []
    
    for r in results:
        doi = r.get("doi")
        if doi:
            doi = doi.strip().lower()
        
        title = r.get("title", "")
        norm_title = re.sub(r'\W+', '', title).lower() if title else ""
        
        if doi and doi in seen_dois:
            continue
        if not doi and norm_title and norm_title in seen_titles:
            continue
            
        if doi: seen_dois.add(doi)
        if norm_title: seen_titles.add(norm_title)
        unique.append(r)
    return unique

def _take(iterable, n):
    return list(iterable)[:n]

def parse_authors(authors_info):
    """Parses author string into Zotero-compatible format."""
    if not authors_info:
        return []
    authors_list = [a.strip() for a in authors_info.split(",") if a.strip()]
    out = []
    for nm in authors_list:
        parts = nm.split(" ")
        if len(parts) >= 2:
            out.append({"creatorType": "author", "firstName": " ".join(parts[:-1]), "lastName": parts[-1]})
        else:
            out.append({"creatorType": "author", "name": nm})
    return out

def with_ntu_proxy(url_or_doi, style=1):
    if not url_or_doi: return None
    if style == 1:
        return f"https://remotexs.ntu.edu.sg/user/login?dest={url_or_doi}"
    return f"https://remotexs.ntu.edu.sg/login?url={url_or_doi}"

def gemini_boolean_query(user_prompt, model):
    """Uses Gemini to generate a Boolean query string."""
    if not client:
        return {"boolean_query": build_boolean_query_simple(user_prompt)}
    
    prompt = f"""
    Convert this research topic into a precise Boolean search string (AND, OR, NOT) for academic databases.
    Topic: "{user_prompt}"
    Output ONLY the boolean string.
    """
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        return {"boolean_query": resp.text.replace('```', '').strip()}
    except Exception as e:
        handle_gemini_backoff(str(e))
        return {"boolean_query": build_boolean_query_simple(user_prompt)}

def build_boolean_query_simple(text: str) -> str:
    OPERATORS = {"and": "AND", "or": "OR", "not": "NOT"}
    q = text.strip()
    tokens = [t.strip() for t in re.split(r",|;|/", q) if t.strip()]
    if len(tokens) >= 2:
        q = " AND ".join([f'"{t}"' if " " in t else t for t in tokens])
    q = re.sub(r"\b(and|or|not)\b", lambda m: OPERATORS[m.group(1).lower()], q, flags=re.I)
    return q

def extract_pdf_text(url):
    """Downloads PDF and extracts text from first 8 pages using PyMuPDF."""
    if not url: return ""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        # Stream check
        r_head = requests.get(url, headers=headers, stream=True, timeout=5)
        is_pdf = "application/pdf" in r_head.headers.get("Content-Type", "").lower()
        if not is_pdf:
            chunk = next(r_head.iter_content(4), b"")
            if not chunk.startswith(b"%PDF"):
                r_head.close()
                return ""
        r_head.close()

        # Download capped at 10MB
        r = requests.get(url, headers=headers, timeout=15)
        if len(r.content) > 10 * 1024 * 1024: return ""
        
        with fitz.open(stream=io.BytesIO(r.content), filetype="pdf") as doc:
            text = []
            for i, page in enumerate(doc):
                if i >= 8: break
                text.append(page.get_text())
            return "\n".join(text)
    except Exception:
        return ""

def check_zotero_duplicate(doi: str, library_id: str, collection_id: str) -> bool:
    """Checks Zotero Cloud API for existing DOI."""
    api_key = st.session_state.get("zotero_api_key", "").strip()
    if not doi or not library_id or not api_key:
        return False
    
    url = f"https://api.zotero.org/users/{library_id}/items"
    headers = {"Zotero-API-Key": api_key}
    params = {"q": doi, "itemType": "journalArticle", "limit": 1}
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        if resp.status_code == 200:
            return len(resp.json()) > 0
    except Exception:
        pass
    return False

def save_to_zotero_local(item_data: Dict[str, Any]) -> tuple[bool, str]:
    """Posts record to local Zotero Connector."""
    connector_url = "http://127.0.0.1:23119/connector/saveItems"
    payload = {"items": [item_data]}
    try:
        # TIMEOUT INCREASED TO 5 MINUTES (300 seconds)
        resp = requests.post(connector_url, json=payload, timeout=300)
        resp.raise_for_status()
        return True, "Item sent to Zotero."
    except requests.exceptions.ConnectionError:
        return False, "Connection refused. Is Zotero Desktop running?"
    except Exception as e:
        return False, str(e)

def gemini_extract_from_text(text, model):
    """Extracts citations from text using Gemini."""
    if not client: return []
    prompt = f"""
    Extract academic references from this text into a JSON array.
    Keys: "title", "authors" (list), "year" (int), "doi" (string/null).
    Text: {text[:15000]}
    """
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        js = re.sub(r'```json|```', '', resp.text).strip()
        return json.loads(js)
    except Exception as e:
        handle_gemini_backoff(str(e))
        return []

DOI_RE = re.compile(r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b', re.IGNORECASE)

@st.cache_resource
def get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        return None

def get_embedding(text: str):
    model = get_embedding_model()
    if model and text:
        return model.encode(text)
    return None

def cosine_similarity(v1, v2):
    if v1 is None or v2 is None: return 0.0
    v1, v2 = np.array(v1), np.array(v2)
    dot = np.dot(v1, v2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return 0.0 if norm1 == 0 or norm2 == 0 else dot / (norm1 * norm2)

def generate_semantic_tags(abstract_text: str, semantic_sentences: List[tuple], semantic_model, top_n: int = 1) -> List[str]:
    """
    Generates a SINGLE tag based on the highest semantic similarity match.
    Uses the user-defined 'custom_tag' instead of the sentence text.
    """
    # Filter enabled sentences
    enabled_entries = [s for s in semantic_sentences if s[1]]
    if not abstract_text or not enabled_entries or not semantic_model:
        return []
    
    try:
        # Get embeddings for the sentences
        enabled_texts = [s[0] for s in enabled_entries]
        abs_emb = semantic_model.encode(abstract_text)
        sent_embs = semantic_model.encode(enabled_texts)
        
        # Calculate scores
        sims = []
        from numpy import dot
        from numpy.linalg import norm
        for s_emb in sent_embs:
            sim = dot(abs_emb, s_emb) / (norm(abs_emb) * norm(s_emb))
            sims.append(sim)
        
        sims = np.array(sims)
        
        # Winner Takes All: Find max score
        best_idx = sims.argmax()
        best_score = sims[best_idx]
        
        # Apply single tag if threshold met (0.4)
        if best_score > 0.4:
            # Structure is (Sentence, Enabled, Pass, CustomTag)
            custom_tag = enabled_entries[best_idx][3]
            # Clean up the tag to ensure it's Zotero/OS friendly
            clean_tag = re.sub(r'[^a-zA-Z0-9_\-]+', '', custom_tag)
            return [f"sTag-{clean_tag}"]
            
        return []
    except Exception as e:
        logging.error(f"Semantic Tag Error: {e}")
        return []

def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def gemini_annotate_paper(title, authors_info, snippet, pdf_text, url, user_query, model, suggested_tags=None, priority_topics=None):
    if not client:
        return "GEMINI API KEY MISSING", [], 0
    
    # --- UPDATED PROMPT LOGIC (CLOSED VOCABULARY + TOPIC RELEVANCE) ---
    tags_instruction = ""
    if suggested_tags and len(suggested_tags) > 0:
        tags_instruction = f"""
    OFFICIAL TAG LIST: {', '.join(suggested_tags)}
    
    INSTRUCTION: Classify this paper using ONLY tags from the 'OFFICIAL TAG LIST' above.
    1. Select all tags from the list that accurately describe the paper.
    2. Do NOT generate novel tags. Do NOT use synonyms.
    3. If the paper fits none of the tags, output 'Uncategorized'.
    """
    else:
        tags_instruction = "INSTRUCTION: Generate 3-5 relevant tags based on the content."

    # --- TOPICS INSTRUCTION FOR GRADING (ADDITIVE) ---
    topics_instruction = ""
    if priority_topics:
        topics_instruction = f"""
    PRIORITY INTERESTS: {priority_topics}
    INSTRUCTION: Evaluate the paper's relevance against the 'User Query' AND these 'PRIORITY INTERESTS'.
    Papers that align closely with the Priority Interests should receive higher relevance scores.
    """

    prompt = f"""
    Analyze this paper.
    Title: {title}
    Authors: {authors_info}
    Abstract: {snippet}
    PDF Extract: {pdf_text[:2000] if pdf_text else 'N/A'}
    User Query: {user_query}

    {topics_instruction}
    {tags_instruction}

    Output format:
    Abstract: [Summary]
    Tags: [tag1, tag2]
    Score: [0-3]
    """
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        text = resp.text
        
        abs_match = re.search(r"Abstract:\s*(.*?)Tags:", text, re.DOTALL)
        tags_match = re.search(r"Tags:\s*\[(.*?)\]", text)
        score_match = re.search(r"Score:\s*(\d)", text)
        
        abstract = abs_match.group(1).strip() if abs_match else text[:500]
        tags = [t.strip() for t in tags_match.group(1).split(',')] if tags_match else []
        score = int(score_match.group(1)) if score_match else 0
        
        return abstract, tags, score
    except Exception as e:
        handle_gemini_backoff(str(e))
        return f"API FAILURE: {e}", [], 0

def gemini_abstract_fallback(title, authors_info, current_snippet, model):
    if client is None: return current_snippet
    prompt = f"""
    Generate a 3-sentence abstract for this paper based on metadata.
    Title: {title}
    Authors: {authors_info}
    Output ONLY text.
    """
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        return resp.text.strip()
    except Exception as e:
        handle_gemini_backoff(str(e))
        return current_snippet

def prerank_papers(papers_meta, user_prompt, semantic_model):
    if not papers_meta or not user_prompt or not semantic_model:
        return papers_meta
    
    q_vec = get_embedding(user_prompt)
    for p in papers_meta:
        doc = f"{p.get('title','')} {p.get('snippet','')}"
        d_vec = get_embedding(doc)
        p['vector_score'] = cosine_similarity(q_vec, d_vec)
    
    papers_meta.sort(key=lambda x: x.get('vector_score', -1.0), reverse=True)
    return papers_meta

# ============================
# LOGGING
# ============================
def log_gemini_status(client, model_id):
    if not client:
        logging.error("GEMINI INIT FAILED.")
        return
    try:
        logging.info(f"GEMINI AUTH OK. Model: {model_id}")
    except Exception:
        logging.error("GEMINI CONNECT FAIL.")

# ============================
# PREFERENCES
# ============================
PREFS_FILE = "prefs.json"

def load_prefs():
    default_prefs = {
        "topics": ["Herbal drug discovery", "Phytochemicals"],
        "authors": [],
        "collection_id": "",
        "library_id": "",
        "allow_duplicates": False,
        "min_abstract_length_chars": 150,
        "semantic_sentences": DEFAULT_SEMANTIC_SENTENCES,
        "ai_tag_categories_list": DEFAULT_AI_TAG_CATEGORIES,
        "automated_queries": [],
        "max_results_value": 20,
        "min_score3_value": 2,
        "selected_model_key": "models/gemini-2.5-flash (Default, 7 GB)",
        "search_mode_value": "Keyword Search",
        "search_source_value": "Semantic Scholar",
        "vector_score_min_value": 0.5,
        "query_mode_value": "Single Query",
        "add_to_zotero_state_value": True,
        "zotero_api_key_value": "",
        "ai_tag_post_filter_values": [],
        "enable_speedup_value": False,
        "speedup_threshold_value": 9
    }
    if not os.path.exists(PREFS_FILE):
        return default_prefs
    try:
        data = json.load(open(PREFS_FILE))
        # Merge defaults
        for k, v in default_prefs.items():
            if k not in data:
                data[k] = v
        
        # MIGRATION: Fix legacy sentences structure (from size 3 to 4)
        sents = []
        raw_sents = data.get("semantic_sentences", [])
        for s in raw_sents:
            if len(s) == 3: 
                # Create a default tag from the first few words of the sentence
                slug = s[0].split()[0] + "-" + s[0].split()[1] if len(s[0].split()) > 1 else "Tag"
                sents.append((s[0], s[1], s[2], slug))
            elif len(s) == 4:
                sents.append(s)
            else:
                continue
                
        data["semantic_sentences"] = sents or DEFAULT_SEMANTIC_SENTENCES
        return data
    except Exception:
        return default_prefs

def save_prefs(data):
    with open(PREFS_FILE, "w") as f:
        json.dump(data, f)

prefs = load_prefs()

# Init Session State
for k, v in prefs.items():
    if k == "topics": k = "topics_txt"; v = ", ".join(v)
    if k == "authors": k = "authors_txt"; v = ", ".join(v)
    if k == "collection_id": k = "user_zotero_collection"
    if k == "library_id": k = "user_zotero_id"
    if k == "zotero_api_key_value": k = "zotero_api_key" 
    
    if k not in st.session_state and "_value" not in k:
        st.session_state[k] = v

if "automated_queries" not in st.session_state:
    st.session_state.automated_queries = prefs["automated_queries"]
if "ai_tag_categories_list" not in st.session_state:
    st.session_state.ai_tag_categories_list = prefs["ai_tag_categories_list"]
if "semantic_sentences" not in st.session_state:
    st.session_state.semantic_sentences = prefs["semantic_sentences"]
if "ai_tag_post_filter_values" not in st.session_state:
    st.session_state.ai_tag_post_filter_values = prefs.get("ai_tag_post_filter_values", [])


def save_current_settings():
    new_prefs = {
        "topics": [t.strip() for t in st.session_state.topics_txt.split(",") if t.strip()],
        "authors": [a.strip() for a in st.session_state.authors_txt.split(",") if a.strip()],
        "collection_id": st.session_state.user_zotero_collection,
        "library_id": st.session_state.user_zotero_id,
        "allow_duplicates": st.session_state.allow_duplicates,
        "semantic_sentences": st.session_state.semantic_sentences,
        "min_abstract_length_chars": st.session_state.abstract_length_slider,
        "ai_tag_categories_list": st.session_state.ai_tag_categories_list,
        "automated_queries": st.session_state.automated_queries,
        "max_results_value": st.session_state.max_results_slider,
        "min_score3_value": st.session_state.min_score3_slider,
        "selected_model_key": st.session_state.model_key_selector,
        "search_mode_value": st.session_state.search_mode_selector,
        "search_source_value": st.session_state.search_source_selector,
        "vector_score_min_value": st.session_state.vector_score_min_slider,
        "query_mode_value": st.session_state.query_mode_selector,
        "add_to_zotero_state_value": st.session_state.add_to_zotero_state,
        "zotero_api_key_value": st.session_state.zotero_api_key,
        "ai_tag_post_filter_values": st.session_state.ai_tag_post_filter_values,
        "enable_speedup_value": st.session_state.enable_speedup_checkbox,
        "speedup_threshold_value": st.session_state.speedup_threshold_slider
    }
    save_prefs(new_prefs)
    st.sidebar.success("✅ Preferences saved.")

# --- UI Helpers (UPDATED FOR CUSTOM TAGS) ---
def add_new_sentence():
    ns = st.session_state.new_sentence_input.strip()
    if ns:
        # Default tag is first word
        default_tag = ns.split()[0][:15]
        st.session_state.semantic_sentences.append((ns, True, True, default_tag))
        st.session_state.new_sentence_input = ""
        save_current_settings()

def delete_sentence(idx):
    st.session_state.semantic_sentences.pop(idx)
    save_current_settings()

def toggle_sentence(idx):
    s, e, p, t = st.session_state.semantic_sentences[idx]
    st.session_state.semantic_sentences[idx] = (s, not e, p, t)
    save_current_settings()

def toggle_pass_state(idx):
    s, e, p, t = st.session_state.semantic_sentences[idx]
    st.session_state.semantic_sentences[idx] = (s, e, not p, t)
    save_current_settings()

def edit_custom_tag(idx):
    # Callback to update the custom tag from text input
    new_tag = st.session_state[f"tag_edit_{idx}"]
    s, e, p, _ = st.session_state.semantic_sentences[idx]
    st.session_state.semantic_sentences[idx] = (s, e, p, new_tag)
    save_current_settings()

def add_new_category(cat):
    if cat and cat not in st.session_state.ai_tag_categories_list:
        st.session_state.ai_tag_categories_list.append(cat)
        save_current_settings()

def delete_category(idx):
    st.session_state.ai_tag_categories_list.pop(idx)
    save_current_settings()

def add_new_query():
    q = st.session_state.new_query_input.strip()
    if q:
        st.session_state.automated_queries.append(q)
        st.session_state.new_query_input = ""
        save_current_settings()

def delete_query(idx):
    st.session_state.automated_queries.pop(idx)
    save_current_settings()

# ============================
# SEARCH PROVIDERS
# ============================

def search_semantic_scholar(query: str, limit: int) -> Iterator[List[Dict[str, Any]]]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
    offset = 0
    seen_dois = set()
    
    st.info(f"🔎 Starting Search for: '{query}'")

    while offset < limit:
        page_limit = min(100, limit - offset)
        if page_limit <= 0: break
        
        params = {
            "query": query, "limit": page_limit, "offset": offset,
            "fields": "title,authors,url,abstract,openAccessPdf,externalIds,venue,year,citationCount,publicationDate,publicationTypes"
        }
        
        try:
            # Check Cache First
            cache_key = f"{query}_{offset}_{page_limit}"
            if cache_key in st.session_state.search_cache:
                data = st.session_state.search_cache[cache_key]
            else:
                sleep(SEMANTIC_SCHOLAR_DELAY)
                data = _request_json_with_retries(url, params=params, headers=headers)
                if data and data.get("data"):
                    st.session_state.search_cache[cache_key] = data
            
            if not data or not data.get("data"):
                if offset == 0: st.warning("Semantic Scholar returned 0 results.")
                break
            
            raw_papers = data.get("data", [])
            
            page_recs = []
            for paper in raw_papers:
                doi = paper.get("externalIds", {}).get("DOI")
                if doi in seen_dois and doi is not None: continue
                if doi: seen_dois.add(doi)
                
                page_recs.append({
                    "title": paper.get("title", ""),
                    "url": paper.get("url", "") or (f"https://doi.org/{doi}" if doi else ""),
                    "authors_info": ", ".join([a.get("name","") for a in paper.get("authors",[])]),
                    "snippet": clean_snippet(paper.get("abstract","")),
                    "pdf_url": (paper.get("openAccessPdf") or {}).get("url",""),
                    "doi": doi,
                    "venue": paper.get("venue"),
                    "year": paper.get("year"),
                    "original_abstract": clean_snippet(paper.get("abstract",""))
                })
            
            for chunk in _chunks(page_recs, PROCESSING_BATCH_SIZE):
                yield chunk
            
            offset += page_limit
            if data.get('total') and offset >= data.get('total'): break
            
        except Exception as e:
            logging.error(f"S2 Search Error: {e}")
            st.error(f"Network Error: {e}")
            break

def search_semantic_scholar_by_doi(doi: str):
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
    params = {"fields": "title,authors,url,abstract,openAccessPdf,externalIds,venue,year"}
    try:
        sleep(SEMANTIC_SCHOLAR_DELAY) # ENFORCED DELAY
        p = _request_json_with_retries(url, params=params, headers=headers)
        if not p: return None
        return {
            "title": p.get("title", ""),
            "url": p.get("url", "") or f"https://doi.org/{doi}",
            "authors_info": ", ".join([a.get("name","") for a in p.get("authors",[])]),
            "snippet": clean_snippet(p.get("abstract", "")),
            "pdf_url": (p.get("openAccessPdf") or {}).get("url", ""),
            "doi": (p.get("externalIds") or {}).get("DOI") or doi,
            "year": p.get("year"),
            "venue": p.get("venue")
        }
    except Exception:
        return None

def search_pubmed_paged(query, limit=10):
    """
    Unchained PubMed search - No artificial delays.
    Includes simple caching.
    """
    # Cache key for the whole pubmed search
    cache_key = f"PUBMED_{query}_{limit}"
    if cache_key in st.session_state.search_cache:
        return st.session_state.search_cache[cache_key]

    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    term = (query or "")[:300]
    
    # 1. ESearch
    all_pmids = []
    retstart = 0
    while len(all_pmids) < limit:
        retmax = min(1000, limit - len(all_pmids))
        params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax, "retstart": retstart, "email": NCBI_EMAIL, "api_key": NCBI_API_KEY}
        try:
            r = requests.get(f"{base}/esearch.fcgi", params=params, timeout=30).json()
            ids = r.get("esearchresult", {}).get("idlist", [])
            if not ids: break
            all_pmids.extend(ids)
            retstart += 1000
        except Exception:
            break
            
    ids = all_pmids[:limit]
    if not ids: return []

    # 2. ESummary
    sum_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json", "email": NCBI_EMAIL, "api_key": NCBI_API_KEY}
    try:
        sm = requests.get(f"{base}/esummary.fcgi", params=sum_params, timeout=30).json()
    except:
        return []

    # 3. EFetch (XML for abstracts)
    abstracts = {}
    for chunk in _chunks(ids, 99):
        try:
            ef_params = {"db": "pubmed", "retmode": "xml", "email": NCBI_EMAIL, "api_key": NCBI_API_KEY}
            ef = requests.post(f"{base}/efetch.fcgi", params=ef_params, data={"id": ",".join(chunk)}, timeout=40)
            root = ET.fromstring(ef.text)
            for art in root.findall(".//PubmedArticle"):
                pmid = art.findtext(".//PMID")
                nodes = art.findall(".//Abstract/AbstractText")
                if nodes:
                    txt = " ".join([n.text for n in nodes if n.text])
                else:
                    txt = art.findtext(".//Abstract") or ""
                if pmid: abstracts[pmid] = clean_snippet(txt)
        except Exception:
            pass
            
    # 4. Consolidate
    out = []
    block = sm.get("result", {}) or {}
    for pmid in ids:
        r = block.get(pmid, {}) or {}
        doi = None
        for aid in r.get("articleids", []):
            if aid.get("idtype") == "doi": doi = aid.get("value"); break
            
        out.append({
            "title": r.get("title", ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "authors_info": ", ".join([a.get("name","") for a in r.get("authors", [])]),
            "snippet": abstracts.get(pmid, ""),
            "doi": doi,
            "venue": r.get("fulljournalname"),
            "year": int(r.get("pubdate", "0")[:4]) if r.get("pubdate") else None
        })
    
    # Save to cache
    st.session_state.search_cache[cache_key] = out
    return out

def search_by_url_doi_pdf(url_or_doi):
    if not url_or_doi: return []
    
    # 1. DOI Check
    m = DOI_RE.search(url_or_doi)
    if m:
        p = search_semantic_scholar_by_doi(m.group(0))
        if p: return [p]

    # 2. URL/PDF Check
    if url_or_doi.startswith("http"):
        is_pdf, text = False, ""
        if url_or_doi.endswith(".pdf"):
             text = extract_pdf_text(url_or_doi)
             is_pdf = True
        else:
            try:
                h = requests.head(url_or_doi, timeout=5)
                if "pdf" in h.headers.get("Content-Type", ""):
                    text = extract_pdf_text(url_or_doi)
                    is_pdf = True
            except: pass
        
        if is_pdf and text:
            m = DOI_RE.search(text)
            if m:
                p = search_semantic_scholar_by_doi(m.group(0))
                if p: 
                    p["pdf_text"] = text
                    return [p]
            
            return [{
                "title": "Extracted PDF",
                "url": url_or_doi,
                "snippet": text[:500],
                "pdf_text": text,
                "snippet_source": "PDF_EXTRACT"
            }]
            
    return []

# ============================
# DISPLAY & PROCESSING
# ============================

def _display_paper_details(paper_data):
    """Render a single paper details card from dictionary."""
    paper = paper_data.get('paper', {})
    score = paper_data.get('score', 0)
    tags = paper_data.get('tags', [])
    abstract_ai = paper_data.get('abstract_ai', "")
    passes = paper_data.get('passes', False)
    
    # Optional logic variables (used for status text)
    z_thresh = paper_data.get('z_thresh', 0)
    
    title = paper.get("title", "Untitled")
    doi = paper.get("doi")
    
    status_msg = paper_data.get('status_msg', "") 
    
    icon = "✅ SAVED" if status_msg == "SAVED" else ("🚫 DUPLICATE" if status_msg == "DUPLICATE" else "🚫 SKIPPED")
    
    with st.expander(f"[{icon}] {title} (Score: {score})", expanded=True):
        st.markdown(f"**Authors:** {paper.get('authors_info','')}")
        if doi: st.markdown(f"**DOI:** `{doi}`")
        if paper.get("snippet"): st.markdown(f"**Source Abstract:** {paper['snippet']}")
        if abstract_ai: st.markdown(f"**AI Analysis:**\n{abstract_ai}")
        if tags: st.markdown(f"**Tags:** {', '.join(tags)}")
        
        if status_msg == "DUPLICATE":
            st.warning("Skipped: Duplicate in Zotero.")
        elif status_msg == "SKIPPED":
            st.info(f"Skipped: Score {score} < {z_thresh} or Value Filter mismatch.")

def process_chunk_and_save(chunk, query, total_papers, papers_processed, status_ph, prog_ph, query_stats):
    annotated = 0
    saved = 0
    
    # UI Settings
    min_len = st.session_state.abstract_length_slider
    z_thresh = st.session_state.min_score3_slider
    vec_min = st.session_state.vector_score_min_slider
    
    # Filter Config
    req_vals = [v.lower() for v in st.session_state.ai_tag_post_filter_values]
    suggested_tags = st.session_state.selected_tag_categories
    
    # Speed Up Settings
    enable_speedup = st.session_state.enable_speedup_checkbox
    speedup_threshold = st.session_state.speedup_threshold_slider
    
    # AI Model (Fixed ID Lookup)
    selected_key = st.session_state.model_key_selector
    actual_model_id = MODEL_OPTIONS[selected_key]["model_id"]
    
    # Context (ADDITIVE LOGIC)
    topics_context = st.session_state.get("topics_txt", "")

    # 1. Preranking & Vector Filter
    sem_model = get_embedding_model()
    if sem_model:
        chunk = prerank_papers(chunk, query, sem_model)
        
        # DEBUG: Only filter if slider > 0.0 to prevent accidental filtering
        if vec_min > 0.0:
            original_count = len(chunk)
            chunk = [p for p in chunk if p.get('vector_score', 0) >= vec_min]
            dropped = original_count - len(chunk)
            if dropped > 0:
                logging.info(f"Vector Filter: Dropped {dropped} papers (Score < {vec_min})")
    
    # If chunk is empty after filter, return early
    if not chunk:
        return 0, 0

    for i, paper in enumerate(chunk):
        idx = papers_processed + i
        
        # --- CHECK SPEEDUP STATUS ---
        bypass_ai = query_stats.get('bypass_ai', False)
        
        # 2. Duplicate Check
        doi = paper.get("doi")
        is_dup = False
        if doi and (not st.session_state.allow_duplicates) and check_zotero_duplicate(doi, st.session_state.user_zotero_id, st.session_state.user_zotero_collection):
             is_dup = True

        # 3. Abstract Fallback
        snip = paper.get("snippet", "")
        if len(snip) < min_len:
            # If skipping AI, we also skip abstract fallback? Assuming yes to save time.
            # But snippet is needed for zotero.
            if not bypass_ai:
                snip = gemini_abstract_fallback(paper.get("title"), paper.get("authors_info"), snip, actual_model_id)
                paper["snippet"] = snip
                paper["snippet_source"] = "AI_FALLBACK"

        # 4. PDF Text (Skip if bypassed)
        pdf_text = ""
        if not bypass_ai:
            pdf_text = paper.get("pdf_text") or extract_pdf_text(paper.get("pdf_url") or paper.get("url"))

        # 5. AI Annotation
        if is_dup:
            ai_abs, tags, score = "Duplicate - Skipped Analysis", [], 0
        elif bypass_ai:
            ai_abs, tags, score = "Auto-Saved (Speed Mode)", ["Speed-Save"], 3
        else:
            # ADDITIVE CONTEXT PASSED HERE
            ai_abs, tags, score = gemini_annotate_paper(
                paper.get("title"), paper.get("authors_info"), snip, pdf_text, paper.get("url"), query, actual_model_id,
                suggested_tags=suggested_tags, priority_topics=topics_context
            )
            if "API FAILURE" not in ai_abs: annotated += 1

        # 6. Semantic Tags (Winner Takes All Logic)
        if sem_model and "API FAILURE" not in ai_abs and not is_dup:
            stags = generate_semantic_tags(ai_abs if not bypass_ai else snip, st.session_state.semantic_sentences, sem_model)
            tags.extend(stags)

        # 7. Save Logic (Value Filter)
        passes = True
        if req_vals:
            t_lower = {t.lower() for t in tags}
            passes = any(v in t for t in t_lower for v in req_vals)

        status_msg = "SKIPPED"
        if is_dup:
            status_msg = "DUPLICATE"
        elif st.session_state.add_to_zotero_state and score >= z_thresh and passes:
            item = {
                "itemType": "journalArticle",
                "title": paper.get("title"),
                "creators": parse_authors(paper.get("authors_info")),
                "abstractNote": (snip or ai_abs)[:ZOTERO_MAX_ABSTRACT_CHARS],
                "tags": [{"tag": t[:MAX_TAG_LENGTH]} for t in tags],
                "url": with_ntu_proxy(doi or paper.get("url")),
                "DOI": doi
            }
            ok, msg = save_to_zotero_local(item)
            if ok: 
                saved += 1
                status_msg = "SAVED"
            else: 
                st.error(f"Zotero Error: {msg}")
        
        # --- UPDATE QUERY STATS FOR SPEEDUP LOGIC ---
        if not is_dup and not bypass_ai:
            query_stats['processed'] += 1
            if status_msg == "SAVED":
                query_stats['saved'] += 1
            
            # Check trigger
            if enable_speedup and query_stats['processed'] == 10:
                if query_stats['saved'] >= speedup_threshold:
                    query_stats['bypass_ai'] = True
                    st.success("⚡ Smart Speed-Up Triggered! Remaining papers in this query will be auto-saved.")

        # --- PACK DATA FOR HISTORY AND DISPLAY ---
        result_entry = {
            'paper': paper,
            'score': score,
            'tags': tags,
            'abstract_ai': ai_abs,
            'z_thresh': z_thresh,
            'passes': passes,
            'status_msg': status_msg
        }
        
        # Append to session state history
        st.session_state.results_history.append(result_entry)
        
        # Display immediately
        _display_paper_details(result_entry)
        
        prog_ph.progress(int((idx / total_papers) * 100) if total_papers else 0)

    return annotated, saved

# ============================
# GUI LAYOUT
# ============================
st.title("📚 AI Literature Helper")
col_mode, col_max = st.columns([1, 0.5])

# --- MAIN CONTROLS ---
with col_mode:
    search_mode = st.radio("🔍 Mode", ["Keyword Search", "Paste citation / page text", "Lookup by URL / PDF"], horizontal=True, key="search_mode_selector")
with col_max:
    max_res = st.slider("📄 Max articles", 5, 1000, value=prefs.get("max_results_value", 20), key="max_results_slider")

if search_mode == "Keyword Search":
    src = st.selectbox("📡 Source", ["Semantic Scholar", "PubMed", "Both"], index=0, key="search_source_selector")
    
    st.markdown("---")
    st.subheader("🤖 AI Filters")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.slider("📏 Min Abstract Len", 50, 500, value=prefs.get("min_abstract_length_chars", 150), key="abstract_length_slider")
        st.slider("✅ Vector Score Min", 0.0, 1.0, value=prefs.get("vector_score_min_value", 0.5), step=0.05, key="vector_score_min_slider")
    with c2:
        st.slider("⭐ Min AI Score", 0, 3, value=prefs.get("min_score3_value", 2), key="min_score3_slider")
    with c3:
        with st.expander("➕ Manage Target Tags"):
            for i, cat in enumerate(st.session_state.ai_tag_categories_list):
                c_a, c_b = st.columns([4, 1])
                c_a.text(cat)
                c_b.button("🗑️", key=f"del_cat_{i}", on_click=delete_category, args=(i,))
            ncat = st.text_input("New Tag")
            st.button("Add", on_click=add_new_category, args=(ncat,))
            
        st.multiselect("🎯 Target Tags (Instruct AI to use these)", st.session_state.ai_tag_categories_list, key="selected_tag_categories")
        st.multiselect("🌪️ Value Filter (Only save papers with these tags)", st.session_state.ai_tag_categories_list, key="ai_tag_post_filter_values")

    # --- SMART SPEED-UP SECTION (MOVED HERE FOR VISIBILITY) ---
    with st.expander("⚡ Smart Speed-Up (Auto-Save High Relevance)"):
        st.checkbox("Enable Smart Speed-Up", value=prefs.get("enable_speedup_value", False), key="enable_speedup_checkbox", help="If enabled, the system checks the first 10 papers. If enough are saved, it skips AI for the rest.")
        st.slider("Threshold (Saved/10)", 5, 10, value=prefs.get("speedup_threshold_value", 9), key="speedup_threshold_slider", help="How many of the first 10 must be saved to trigger speed-up.")

    q_mode = st.radio("📝 Query Mode", ["Single Query", "Automated Cycle"], horizontal=True, key="query_mode_selector")
    if q_mode == "Single Query":
        st.text_input("🔍 Topic", key="user_prompt_input")
        st.checkbox("🔤 Use Boolean", key="use_boolean_checkbox")
    else:
        if st.session_state.automated_queries:
            st.multiselect("⬇️ Select Queries", st.session_state.automated_queries, default=st.session_state.automated_queries, key="cycle_queries_selector")
        
        with st.expander("➕ Manage Queries"):
            for i, q in enumerate(st.session_state.automated_queries):
                c_a, c_b = st.columns([4, 1])
                c_a.text(q)
                c_b.button("🗑️", key=f"del_q_{i}", on_click=delete_query, args=(i,))
            nq = st.text_input("New Query", key="new_query_input")
            st.button("Add Query", on_click=add_new_query)

elif search_mode == "Paste citation / page text":
    st.text_area("📋 Paste text", height=200, key="paste_text")

else:
    st.text_input("🔗 Paste URL/DOI", key="url_or_doi")

# --- SEMANTIC SENTENCES (UPDATED UI) ---
st.markdown("---")
with st.expander("🎯 Semantic Classification Sentences"):
    col_t, col_e, col_p, col_c, col_d = st.columns([5, 1, 1, 3, 1])
    col_t.caption("**Classification Sentence**")
    col_e.caption("**Enabled**")
    col_p.caption("**Pass**")
    col_c.caption("**Custom Tag (Edit)**")
    col_d.caption("**Del**")

    for i, (sent, en, pas, tag) in enumerate(st.session_state.semantic_sentences):
        c1, c2, c3, c4, c5 = st.columns([5, 1, 1, 3, 1])
        c1.text(sent)
        c2.button("✅" if en else "❌", key=f"tog_s_{i}", on_click=toggle_sentence, args=(i,))
        c3.button("➡️" if pas else "🚫", key=f"tog_p_{i}", on_click=toggle_pass_state, args=(i,))
        
        # Editable Custom Tag
        c4.text_input("Tag", value=tag, key=f"tag_edit_{i}", label_visibility="collapsed", on_change=edit_custom_tag, args=(i,))
        
        c5.button("🗑️", key=f"del_s_{i}", on_click=delete_sentence, args=(i,))
        
    st.text_input("New Sentence", key="new_sentence_input")
    st.button("Add Sentence", on_click=add_new_sentence)

# --- MODEL SELECTOR ---
st.markdown("---")
m_keys = list(MODEL_OPTIONS.keys())
st.selectbox("🤖 Model", m_keys, index=0, key="model_key_selector")

# --- SIDEBAR ---
with st.sidebar:
    st.checkbox("📥 Add to Zotero", key="add_to_zotero_state", value=prefs.get("add_to_zotero_state_value", True))
    st.checkbox("⚠️ Allow Duplicates", key="allow_duplicates", value=prefs.get("allow_duplicates", False))
    st.markdown("---")
    st.text_input("Zotero Cloud API Key (for duplicate check)", type="password", key="zotero_api_key", value=prefs.get("zotero_api_key_value", ""))
    st.text_input("Zotero Library ID", key="user_zotero_id", value=prefs.get("library_id", ""))
    st.text_input("Zotero Collection ID", key="user_zotero_collection", value=prefs.get("collection_id", ""))
    st.text_area("Topics (comma-sep)", value=st.session_state.get("topics_txt", ""), key="topics_txt", help="Priority Interests: These topics are sent to the AI to help it determine if a paper is relevant (giving it a higher score). They are also used to populate the 'Automated Cycle' query list.")
    st.text_area("Authors (comma-sep)", value=st.session_state.get("authors_txt", ""), key="authors_txt")
    if st.button("💾 Save Settings"):
        save_current_settings()

# ============================
# MAIN EXECUTION
# ============================

def run_cycle_logic(queries):
    prog = st.progress(0)
    status = st.empty()
    
    total_ann = 0
    total_save = 0
    
    # Get model ID for Boolean Query
    selected_key = st.session_state.model_key_selector
    actual_model_id = MODEL_OPTIONS[selected_key]["model_id"]
    
    for q_idx, query in enumerate(queries):
        status.info(f"Processing: {query}")
        
        # Initialize stats for this query cycle
        query_stats = {
            'processed': 0,
            'saved': 0,
            'bypass_ai': False
        }
        
        # 1. Prepare Query
        final_query = query
        if search_mode == "Keyword Search" and st.session_state.get("use_boolean_checkbox"):
            res = gemini_boolean_query(query, actual_model_id)
            final_query = res.get("boolean_query", query)
            st.info(f"Boolean Query: {final_query}")

        # 2. Acquire
        papers = []
        if search_mode == "Keyword Search":
            source = st.session_state.search_source_selector
            if source in ["Semantic Scholar", "Both"]:
                for chunk in search_semantic_scholar(final_query, st.session_state.max_results_slider):
                    a, s = process_chunk_and_save(chunk, final_query, 100, 0, status, prog, query_stats)
                    total_ann += a; total_save += s
            if source in ["PubMed", "Both"]:
                 p_res = search_pubmed_paged(final_query, st.session_state.max_results_slider)
                 papers.extend(p_res)

        elif search_mode == "Paste citation / page text":
            refs = gemini_extract_from_text(st.session_state.paste_text, actual_model_id)
            for r in refs:
                if r.get("doi"): 
                    p = search_semantic_scholar_by_doi(r["doi"])
                    if p: papers.append(p)
                elif r.get("title"):
                    p = search_pubmed_paged(r["title"], 1)
                    if p: papers.append(p[0])

        elif search_mode == "Lookup by URL / PDF":
            papers = search_by_url_doi_pdf(st.session_state.url_or_doi)

        # 3. Process Non-Chunked Papers (PubMed/Paste/URL)
        if papers:
            # Dedupe
            papers = dedupe_results(papers)
            a, s = process_chunk_and_save(papers, final_query, len(papers), 0, status, prog, query_stats)
            total_ann += a; total_save += s

    st.success(f"Run Complete. Annotated: {total_ann}, Saved: {total_save}")

# --- RESULTS AREA ---
st.markdown("---")
st.subheader("📝 Results Log (Persists across runs)")

# Render existing history first
if st.session_state.results_history:
    for item in st.session_state.results_history:
        _display_paper_details(item)
else:
    st.info("No results yet. Click Go to start.")

# --- ACTION BUTTONS ---
c_go, c_pause, c_clear = st.columns([1, 1, 4])
start_run = False

with c_go:
    if st.button("🚀 Go"):
        start_run = True

with c_pause:
    if st.button("⏸️ Pause"):
        st.warning("Process paused/stopped by user. Click 'Go' to restart.")
        st.stop()

with c_clear:
    if st.button("🗑️ Clear Results"):
        st.session_state.results_history = []
        st.session_state.search_cache = {} # Also clear cache if desired
        st.rerun()

# --- RUN LOGIC ---
if start_run:
    qs = []
    if search_mode == "Keyword Search":
        if st.session_state.query_mode_selector == "Single Query":
            qs = [st.session_state.user_prompt_input]
        else:
            qs = st.session_state.cycle_queries_selector
    else:
        qs = ["Single Execution"] 
        
    if qs:
        run_cycle_logic(qs)
    else:
        st.error("No query selected.")
