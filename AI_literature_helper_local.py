# -*- coding: utf-8 -*- version 1.5
import streamlit as st
import pandas as pd  # Required for the Data Editor
import requests, json, re, os, io, csv
import xml.etree.ElementTree as ET
# import fitz  # Replaced by PDFMiner
from pdfminer.high_level import extract_text # PDFMiner.six
from time import sleep, time
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
OLLAMA_BASE_URL = "http://localhost:11434/api"

# --- RATE LIMITING ---
SEMANTIC_SCHOLAR_DELAY = 1.25  # Max 1 request per 0.75 seconds (Protected)
GEMINI_DELAY = 3.5             # Max 1 request per 0.5 seconds (Prevent Hammering)
sleepgetpapers = 1.75          # Global pacing for generic robust requests

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
    "models/gemini-2.0-flash (Default, 7 GB)": {
        "model_id": "gemini-2.0-flash",
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
    "Local Ollama (Offline)": {
        "model_id": "LOCAL_OLLAMA",
        "description": "Use a local model running via Ollama. Requires Ollama to be running."
    }
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

# New: Cycle State for Resuming
if "cycle_state" not in st.session_state:
    st.session_state.cycle_state = {
        "active": False,
        "query_idx": 0,
        "paper_offset": 0,
        "query_stats": {'processed': 0, 'saved': 0, 'bypass_ai': False}
    }

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

# --- OLLAMA HELPERS ---
def get_ollama_models():
    """Fetches available models from local Ollama instance."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/tags", timeout=2)
        if resp.status_code == 200:
            models = resp.json().get('models', [])
            return [m['name'] for m in models]
    except Exception:
        pass
    return []

def query_ollama_chat(model, prompt, temperature=0.3):
    """Queries local Ollama chat API."""
    url = f"{OLLAMA_BASE_URL}/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature}
    }
    try:
        resp = requests.post(url, json=payload, timeout=120) # Long timeout for local inference
        if resp.status_code == 200:
            return resp.json().get('message', {}).get('content', '')
        else:
            return f"OLLAMA ERROR: {resp.status_code} - {resp.text}"
    except Exception as e:
        return f"OLLAMA CONNECTION ERROR: {e}"

# --- GEMINI HELPERS ---
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
    """Uses Gemini or Ollama to generate a Boolean query string."""
    use_ollama = st.session_state.get("use_ollama", False)
    
    if not client and not use_ollama:
        return {"boolean_query": build_boolean_query_simple(user_prompt)}
    
    prompt = f"""
    Convert this research topic into a precise Boolean search string (AND, OR, NOT) for academic databases.
    Topic: "{user_prompt}"
    Output ONLY the boolean string.
    """
    
    # --- OLLAMA PATH ---
    if use_ollama:
        try:
            resp_text = query_ollama_chat(model, prompt)
            if "OLLAMA" in resp_text: return {"boolean_query": build_boolean_query_simple(user_prompt)}
            return {"boolean_query": resp_text.replace('```', '').strip()}
        except Exception:
            return {"boolean_query": build_boolean_query_simple(user_prompt)}

    # --- GEMINI PATH ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            sleep(GEMINI_DELAY) # Rate limit
            resp = client.models.generate_content(model=model, contents=prompt)
            return {"boolean_query": resp.text.replace('```', '').strip()}
        except Exception as e:
            if attempt < max_retries - 1 and handle_gemini_backoff(str(e)):
                continue # Retry loop
            logging.error(f"Gemini Boolean Query Failed: {e}")
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
    """
    Downloads PDF and extracts text using PDFMiner.six.
    Limits to first 24 pages.
    """
    if not url: return ""
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        
        # 1. HEAD request to check file size first (Save bandwidth)
        try:
            head_resp = requests.head(url, headers=headers, timeout=3)
            file_size = int(head_resp.headers.get('Content-Length', 0))
            if file_size > 15 * 1024 * 1024: # Skip if > 15MB
                logging.warning(f"PDF too large ({file_size} bytes), skipping: {url}")
                return ""
        except:
            pass

        # 2. GET request
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200: return ""
        
        # 3. Check content type
        if "application/pdf" not in r.headers.get("Content-Type", "").lower():
             if not r.content.startswith(b"%PDF"):
                 return ""

        # 4. Extract Text using PDFMiner
        # maxpages=24 ensures we don't process huge books
        with io.BytesIO(r.content) as pdf_stream:
            text = extract_text(pdf_stream, maxpages=24)
            return text

    except Exception as e:
        logging.error(f"PDFMiner Extract Error: {e}")
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
    """Extracts citations from text using Gemini or Ollama."""
    use_ollama = st.session_state.get("use_ollama", False)
    if not client and not use_ollama: return []
    
    prompt = f"""
    Extract academic references from this text into a JSON array.
    Keys: "title", "authors" (list), "year" (int), "doi" (string/null).
    Text: {text[:15000]}
    """
    
    # --- OLLAMA PATH ---
    if use_ollama:
        try:
            resp_text = query_ollama_chat(model, prompt)
            js = re.sub(r'```json|```', '', resp_text).strip()
            return json.loads(js)
        except:
            return []

    # --- GEMINI PATH ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            sleep(GEMINI_DELAY)
            resp = client.models.generate_content(model=model, contents=prompt)
            js = re.sub(r'```json|```', '', resp.text).strip()
            return json.loads(js)
        except Exception as e:
            if attempt < max_retries - 1 and handle_gemini_backoff(str(e)):
                continue
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
    use_ollama = st.session_state.get("use_ollama", False)
    
    if not client and not use_ollama:
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

    # INCREASED CONTEXT LIMIT TO 40000 CHARS FOR PDF TEXT (approx 24 pages)
    prompt = f"""
    Analyze this paper.
    Title: {title}
    Authors: {authors_info}
    Abstract: {snippet}
    PDF Extract: {pdf_text[:40000] if pdf_text else 'N/A'}
    User Query: {user_query}

    {topics_instruction}
    {tags_instruction}

    Output format:
    Abstract: [Summary]
    Tags: [tag1, tag2]
    Score: [0-3]
    """
    
    # --- OLLAMA PATH ---
    if use_ollama:
        try:
            text = query_ollama_chat(model, prompt)
            if "OLLAMA" in text: # Error caught
                 return text, [], 0
                 
            # Parse
            abs_match = re.search(r"Abstract:\s*(.*?)Tags:", text, re.DOTALL)
            tags_match = re.search(r"Tags:\s*\[(.*?)\]", text)
            score_match = re.search(r"Score:\s*(\d)", text)
            
            abstract = abs_match.group(1).strip() if abs_match else text[:500]
            tags = [t.strip() for t in tags_match.group(1).split(',')] if tags_match else []
            score = int(score_match.group(1)) if score_match else 0
            
            return abstract, tags, score
        except Exception as e:
            return f"OLLAMA FAIL: {e}", [], 0

    # --- GEMINI PATH (RETRY LOOP) ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            sleep(GEMINI_DELAY) # Rate limit
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
            # Check for backoff condition
            if attempt < max_retries - 1 and handle_gemini_backoff(str(e)):
                # handle_gemini_backoff already sleeps, so we just continue the loop to retry
                logging.info(f"Retrying Gemini Annotation (Attempt {attempt+2}/{max_retries})...")
                continue
            
            # If we are here, it's either not a 429 or we ran out of retries
            return f"API FAILURE: {e}", [], 0

def gemini_abstract_fallback(title, authors_info, current_snippet, model):
    use_ollama = st.session_state.get("use_ollama", False)
    if not client and not use_ollama: return current_snippet
    
    prompt = f"""
    Generate a 3-sentence abstract for this paper based on metadata.
    Title: {title}
    Authors: {authors_info}
    Output ONLY text.
    """
    
    # --- OLLAMA PATH ---
    if use_ollama:
        try:
            return query_ollama_chat(model, prompt)
        except:
            return current_snippet

    # --- GEMINI PATH ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            sleep(GEMINI_DELAY) # Rate limit
            resp = client.models.generate_content(model=model, contents=prompt)
            return resp.text.strip()
        except Exception as e:
            if attempt < max_retries - 1 and handle_gemini_backoff(str(e)):
                logging.info(f"Retrying Abstract Fallback (Attempt {attempt+2}/{max_retries})...")
                continue
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
    if st.session_state.get("use_ollama"):
        logging.info(f"USING LOCAL OLLAMA. Model: {model_id}")
        return

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
        "topics": [
            "Plant-Derived Chemicals", 
            "herbal extracts", 
            "phytochemicals", 
            "Plant Bioactive Compound", 
            "Phytonutrient", 
            "Dietary Phytochemicals", 
            "Biologically Active Compounds, Plant", 
            "PLANT ALKALOIDS", 
            "TCM"
        ],
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
        "enable_speedup_value": True, # Enabled by default
        "speedup_threshold_value": 9  # 9/10 default
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

def save_prefs_silent(data):
    """Helper to save preferences without triggering the UI toast notification."""
    with open(PREFS_FILE, "w") as f:
        json.dump(data, f)

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

# ============================
# SEARCH PROVIDERS
# ============================

def search_semantic_scholar(query: str, limit: int, start_offset: int = 0) -> Iterator[List[Dict[str, Any]]]:
    # --- FIX 1: Safety check for empty queries ---
    if not query or not query.strip():
        # Log this internally but don't show user error unless debugging
        logging.warning("Skipping Semantic Scholar search: Query is empty.")
        return # Yields nothing, safe exit
    
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
    offset = start_offset # Start from RESUME point
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
    
    # Retrieve details
    vector_score = paper.get('vector_score', 0.0)
    process_time = paper_data.get('process_time', 0.0)
    
    # Optional logic variables (used for status text)
    z_thresh = paper_data.get('z_thresh', 0)
    
    title = paper.get("title", "Untitled")
    doi = paper.get("doi")
    
    status_msg = paper_data.get('status_msg', "") 
    
    # Icon logic
    icon = "✅ SAVED" if status_msg == "SAVED" else ("🚫 DUPLICATE" if status_msg == "DUPLICATE" else "🚫 SKIPPED")
    
    with st.expander(f"[{icon}] {title}", expanded=True):
        
        # --- BIG COLORFUL STATUS BAR ---
        status_color = "green" if status_msg == "SAVED" else ("orange" if status_msg == "DUPLICATE" else "red")
        
        # Construct status text
        if status_msg == "SAVED":
            bar_text = f"**SAVED** | AI Score: {score}/3 | Vector Sim: {vector_score:.2f} | Time: {process_time:.2f}s | Tags: {len(tags)}"
            st.success(bar_text)
        elif status_msg == "DUPLICATE":
            bar_text = f"**DUPLICATE** | DOI: {doi} | Time: {process_time:.2f}s"
            st.warning(bar_text)
        else:
            reason = "Score too low" if score < z_thresh else "Filters failed"
            bar_text = f"**SKIPPED** | Reason: {reason} | AI Score: {score}/3 | Vector Sim: {vector_score:.2f}"
            st.error(bar_text)
        # -------------------------------

        st.markdown(f"**Authors:** {paper.get('authors_info','')}")
        if doi: st.markdown(f"**DOI:** `{doi}`")
        if paper.get("snippet"): st.markdown(f"**Source Abstract:** {paper['snippet']}")
        if abstract_ai: st.markdown(f"**AI Analysis:**\n{abstract_ai}")
        if tags: st.markdown(f"**Tags:** {', '.join(tags)}")

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
    
    # AI Model Resolution (LOCAL AWARE)
    if st.session_state.get("use_ollama"):
        actual_model_id = st.session_state.get("ollama_local_model_selector", "llama3")
    else:
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
        start_time = time() # START TIMER
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

        end_time = time()
        duration = end_time - start_time

        # --- PACK DATA FOR HISTORY AND DISPLAY ---
        result_entry = {
            'paper': paper,
            'score': score,
            'tags': tags,
            'abstract_ai': ai_abs,
            'z_thresh': z_thresh,
            'passes': passes,
            'status_msg': status_msg,
            'process_time': duration # Added Duration
        }
        
        # Append to session state history
        st.session_state.results_history.append(result_entry)
        
        # Display immediately
        _display_paper_details(result_entry)
        
        prog_ph.progress(int((idx / total_papers) * 100) if total_papers else 0)
        
        # UPDATE CYCLE STATE (FOR RESUME)
        st.session_state.cycle_state['paper_offset'] += 1

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
            st.button("Add", on_click=add_new_category, args=(ncat,), key="btn_add_cat")
            
        st.multiselect("🎯 Target Tags (Instruct AI to use these)", st.session_state.ai_tag_categories_list, key="selected_tag_categories")
        st.multiselect("🌪️ Value Filter (Only save papers with these tags)", st.session_state.ai_tag_categories_list, key="ai_tag_post_filter_values")

    # --- SMART SPEED-UP SECTION (MOVED HERE FOR VISIBILITY) ---
    with st.expander("⚡ Smart Speed-Up (Auto-Save High Relevance)"):
        st.checkbox("Enable Smart Speed-Up", value=prefs.get("enable_speedup_value", False), key="enable_speedup_checkbox", help="If enabled, the system checks the first 10 papers. If enough are saved, it skips AI for the rest.")
        st.slider("Threshold (Saved/10)", 5, 10, value=prefs.get("speedup_threshold_value", 9), key="speedup_threshold_slider", help="How many of the first 10 must be saved to trigger speed-up.")

    # --- QUERY MODE SECTION ---
    q_mode = st.radio("📝 Query Mode", ["Single Query", "Automated Cycle"], horizontal=True, key="query_mode_selector")
    
    if q_mode == "Single Query":
        st.text_input("🔍 Topic", key="user_prompt_input")
        st.checkbox("🔤 Use Boolean", key="use_boolean_checkbox")
    else:
        # Show selected queries if any exist
        if st.session_state.automated_queries:
            # REMOVED DEFAULT=... TO FIX "HELL BREAKING LOOSE" BUG
            st.multiselect("⬇️ Select Queries", st.session_state.automated_queries, key="cycle_queries_selector")
        
        # --- MANAGE QUERIES BOX (WITH DATA EDITOR TABLE) ---
        with st.expander("➕ Manage Queries (Table View)", expanded=True):
            # 1. Prepare Data
            # Convert list of strings to DF with a 'Select' column
            current_queries = st.session_state.automated_queries
            df = pd.DataFrame(
                {
                    "Select": [False] * len(current_queries),
                    "Queries": current_queries
                }
            )

            # 2. Render Editor
            edited_df = st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                key="query_table",
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Select",
                        help="Check to delete",
                        default=False,
                        width="small"
                    ),
                    "Queries": st.column_config.TextColumn(
                        "Search Queries",
                        required=True,
                        width="large"
                    )
                },
                hide_index=True
            )

            # 3. Logic for Updates and Deletes
            # Detect if "Delete Selected" is clicked
            col_del, col_del_all, col_space = st.columns([1, 1, 3])
            with col_del:
                delete_btn = st.button("🗑️ Delete Selected", type="primary", use_container_width=True)
            with col_del_all:
                delete_all_btn = st.button("💥 Delete All", type="secondary", use_container_width=True)
            
            # Logic
            if delete_all_btn:
                # Clear everything
                st.session_state.automated_queries = []
                # Silent Save
                new_prefs = {
                    "topics": [t.strip() for t in st.session_state.topics_txt.split(",") if t.strip()],
                    "authors": [a.strip() for a in st.session_state.authors_txt.split(",") if a.strip()],
                    "collection_id": st.session_state.user_zotero_collection,
                    "library_id": st.session_state.user_zotero_id,
                    "allow_duplicates": st.session_state.allow_duplicates,
                    "semantic_sentences": st.session_state.semantic_sentences,
                    "min_abstract_length_chars": st.session_state.abstract_length_slider,
                    "ai_tag_categories_list": st.session_state.ai_tag_categories_list,
                    "automated_queries": [], # CLEARED
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
                save_prefs_silent(new_prefs)
                st.rerun()

            elif delete_btn:
                # Filter out selected rows
                # Normalize 'Select' to bool (handle NaNs from new rows)
                edited_df["Select"] = edited_df["Select"].fillna(False).astype(bool)
                
                # Keep rows where Select is False
                kept_df = edited_df[~edited_df["Select"]]
                
                # Extract queries
                new_list = kept_df["Queries"].astype(str).tolist()
                new_list = [q.strip() for q in new_list if q.strip()]
                
                # Update State
                st.session_state.automated_queries = new_list
                
                new_prefs = {
                    "topics": [t.strip() for t in st.session_state.topics_txt.split(",") if t.strip()],
                    "authors": [a.strip() for a in st.session_state.authors_txt.split(",") if a.strip()],
                    "collection_id": st.session_state.user_zotero_collection,
                    "library_id": st.session_state.user_zotero_id,
                    "allow_duplicates": st.session_state.allow_duplicates,
                    "semantic_sentences": st.session_state.semantic_sentences,
                    "min_abstract_length_chars": st.session_state.abstract_length_slider,
                    "ai_tag_categories_list": st.session_state.ai_tag_categories_list,
                    "automated_queries": st.session_state.automated_queries, # UPDATED LIST
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
                save_prefs_silent(new_prefs)
                st.rerun()
            
            else:
                # Just Sync Edits (Text changes, additions)
                # If the user just edited text, we update state so it persists if they navigate away
                # We do this check to avoid unnecessary saves/reruns loop, 
                # but with st.data_editor, it only returns new df on interaction.
                
                # Extract current visible list
                current_visible_list = edited_df["Queries"].astype(str).tolist()
                current_visible_list = [q.strip() for q in current_visible_list if q.strip()]
                
                if current_visible_list != st.session_state.automated_queries:
                    st.session_state.automated_queries = current_visible_list
                    new_prefs = {
                        "topics": [t.strip() for t in st.session_state.topics_txt.split(",") if t.strip()],
                        "authors": [a.strip() for a in st.session_state.authors_txt.split(",") if a.strip()],
                        "collection_id": st.session_state.user_zotero_collection,
                        "library_id": st.session_state.user_zotero_id,
                        "allow_duplicates": st.session_state.allow_duplicates,
                        "semantic_sentences": st.session_state.semantic_sentences,
                        "min_abstract_length_chars": st.session_state.abstract_length_slider,
                        "ai_tag_categories_list": st.session_state.ai_tag_categories_list,
                        "automated_queries": st.session_state.automated_queries, # UPDATED LIST
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
                    save_prefs_silent(new_prefs)

            # C. CSV Upload Area
            st.markdown("---")
            st.caption("📂 **Bulk Import via CSV** (Drag & Drop)")
            uploaded_file = st.file_uploader("Upload CSV (First column will be imported)", type=['csv'], key="csv_query_uploader")
            
            if uploaded_file is not None:
                try:
                    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                    reader = csv.reader(stringio)
                    
                    added_count = 0
                    for row in reader:
                        if not row: continue
                        val = row[0].strip()
                        if not val or val.lower() in ["query", "queries", "keyword"]: continue
                        if val not in st.session_state.automated_queries:
                            st.session_state.automated_queries.append(val)
                            added_count += 1
                    
                    if added_count > 0:
                        save_current_settings()
                        st.success(f"✅ Imported {added_count} queries!")
                        sleep(1)
                        st.rerun()
                except Exception as e:
                    st.error(f"CSV Error: {e}")


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

# -------------------------------------------------------
# NEW LOGIC FOR LOCAL OLLAMA MODEL SELECTION
# -------------------------------------------------------

# 1. Primary Selectbox (Provider)
selected_option = st.selectbox("🤖 Model", m_keys, index=0, key="model_key_selector")

# 2. Check if user picked Ollama
is_local_ollama = (MODEL_OPTIONS[selected_option]["model_id"] == "LOCAL_OLLAMA")
st.session_state.use_ollama = is_local_ollama

# 3. If Local Ollama, show secondary dropdown for actual model
if is_local_ollama:
    local_models = get_ollama_models()
    if local_models:
        local_model_name = st.selectbox("💻 Select Local Ollama Model", local_models, key="ollama_local_model_selector")
    else:
        st.error("Could not fetch models from Ollama (localhost:11434). Is it running?")
# -------------------------------------------------------


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
    
    # --- RESOLVE MODEL ID (GEMINI vs OLLAMA) ---
    if st.session_state.get("use_ollama"):
        actual_model_id = st.session_state.get("ollama_local_model_selector", "llama3") # Default fallback
    else:
        selected_key = st.session_state.model_key_selector
        actual_model_id = MODEL_OPTIONS[selected_key]["model_id"]
    # -------------------------------------------
    
    # Cycle Management from State
    start_q_idx = st.session_state.cycle_state['query_idx']
    
    for q_idx in range(start_q_idx, len(queries)):
        query = queries[q_idx]
        
        # Update state for resume (Query Level)
        st.session_state.cycle_state['query_idx'] = q_idx
        
        status.info(f"Processing: {query}")
        
        # --- FIX 2: Check for empty query inside loop ---
        if not query or not query.strip():
            st.warning(f"Skipping empty query at index {q_idx}")
            continue

        # Initialize stats for this query cycle (Recalculated or Resumed?)
        # For simplicity, stats reset on resume for the current query to ensure safety
        query_stats = st.session_state.cycle_state.get('query_stats', {
                       
            'processed': 0, 'saved': 0, 'bypass_ai': False
        })
        
        # 1. Prepare Query
        final_query = query
        if search_mode == "Keyword Search" and st.session_state.get("use_boolean_checkbox"):
            res = gemini_boolean_query(query, actual_model_id)
            final_query = res.get("boolean_query", query)
            st.info(f"Boolean Query: {final_query}")

        # 2. Acquire
        # Determine offset for resume
        start_offset = st.session_state.cycle_state['paper_offset'] if q_idx == start_q_idx else 0
        # Reset offset for new queries
        if q_idx != start_q_idx: 
            st.session_state.cycle_state['paper_offset'] = 0
            st.session_state.cycle_state['query_stats'] = {'processed': 0, 'saved': 0, 'bypass_ai': False}
            query_stats = st.session_state.cycle_state['query_stats']

        if search_mode == "Keyword Search":
            source = st.session_state.search_source_selector
            if source in ["Semantic Scholar", "Both"]:
                for chunk in search_semantic_scholar(final_query, st.session_state.max_results_slider, start_offset):
                    a, s = process_chunk_and_save(chunk, final_query, 100, 0, status, prog, query_stats)
                    total_ann += a; total_save += s
            if source in ["PubMed", "Both"]:
                 # Pubmed paging doesn't support fine-grained offset resume in this implementation easily without cache check
                 # Relying on cache to skip network, process_chunk handles processing
                 p_res = search_pubmed_paged(final_query, st.session_state.max_results_slider)
                 # Slicing for resume
                 p_res = p_res[start_offset:]
                 if p_res:
                    a, s = process_chunk_and_save(p_res, final_query, len(p_res), 0, status, prog, query_stats)
                    total_ann += a; total_save += s

        elif search_mode == "Paste citation / page text":
            refs = gemini_extract_from_text(st.session_state.paste_text, actual_model_id)
            papers = []
            for r in refs:
                if r.get("doi"): 
                    p = search_semantic_scholar_by_doi(r["doi"])
                    if p: papers.append(p)
                elif r.get("title"):
                    p = search_pubmed_paged(r["title"], 1)
                    if p: papers.append(p[0])
            if papers:
                a, s = process_chunk_and_save(papers[start_offset:], final_query, len(papers), 0, status, prog, query_stats)
                total_ann += a; total_save += s

        elif search_mode == "Lookup by URL / PDF":
            papers = search_by_url_doi_pdf(st.session_state.url_or_doi)

                                                          
            if papers:
                    
                                           
                a, s = process_chunk_and_save(papers[start_offset:], final_query, len(papers), 0, status, prog, query_stats)
                total_ann += a; total_save += s

    st.success(f"Run Complete. Annotated: {total_ann}, Saved: {total_save}")
    # Reset Cycle State on Completion
    st.session_state.cycle_state = {
        "active": False,
        "query_idx": 0,
        "paper_offset": 0,
        "query_stats": {'processed': 0, 'saved': 0, 'bypass_ai': False}
    }

# --- RESULTS AREA ---
st.markdown("---")
st.subheader("📝 Results Log (Persists across runs)")

                               
if st.session_state.results_history:
    for item in st.session_state.results_history:
        _display_paper_details(item)
else:
    st.info("No results yet. Click Go to start.")

# --- ACTION BUTTONS ---
c_go, c_pause, c_reset, c_clear = st.columns([2, 1, 1, 2])
start_run = False

# State-Aware Button Logic
is_active = st.session_state.cycle_state['active']

with c_go:
    label = "▶️ Resume Cycle" if is_active else "🚀 Go"
    if st.button(label):
        # --- FIX 3: Prevent starting empty searches ---
        if search_mode == "Keyword Search" and st.session_state.query_mode_selector == "Single Query":
             if not st.session_state.user_prompt_input.strip():
                 st.error("Please enter a topic.")
             else:
                 st.session_state.cycle_state['active'] = True
                 start_run = True
        else:
            st.session_state.cycle_state['active'] = True
            start_run = True

with c_pause:
    if st.button("⏸️ Pause"):
        st.warning("Cycle Paused. You can adjust settings and click Resume.")
        # No st.stop() needed, just don't trigger start_run. 
        # State remains 'active' so Resume appears.

with c_reset:
    if is_active:
        if st.button("⏹️ Reset"):
            st.session_state.cycle_state = {
                "active": False,
                "query_idx": 0,
                "paper_offset": 0,
                "query_stats": {'processed': 0, 'saved': 0, 'bypass_ai': False}
            }
            st.rerun()

with c_clear:
    if st.button("🗑️ Clear Results"):
        st.session_state.results_history = []
        st.session_state.search_cache = {} 
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
        qs = ["Single Execution"] # Dummy for other modes
        
    if qs:
        run_cycle_logic(qs)
    else:
        st.error("No query selected.")
