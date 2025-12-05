# -*- coding: utf-8 -*- version 3   AIstudio v3 complete rebuild with many updates and improvements. Stable with good output.
# -*- coding: utf-8 -*-
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
from bs4 import BeautifulSoup
from urllib.parse import urljoin 

# ============================
# LOGGING CONFIGURATION
# ============================
# 1. Set Root Logger to INFO
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s', stream=sys.stdout)

# 2. SILENCE PDFMINER & URLLIB3
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("http.client").setLevel(logging.WARNING)

# --- GLOBAL API CONSTANTS ---
MAX_RETRIES = 5
INITIAL_DELAY_SECONDS = 2
RATE_LIMIT_STATUS_CODE = 429
ZOTERO_MAX_ABSTRACT_CHARS = 10000
MAX_TAG_LENGTH = 250
PROCESSING_BATCH_SIZE = 50
DOI_MISSING_ERROR = "⚠️ ERROR: The Digital Object Identifier (DOI) is a CRITICAL identifier. This item can still be saved to Zotero because it has an abstract and url."
OLLAMA_BASE_URL = "http://localhost:11434/api"
CONFIG_DIR = "saved_configs" # Directory for user configs

# --- RATE LIMITING ---
SEMANTIC_SCHOLAR_DELAY = 1.25  # Max 1 request per 0.75 seconds (Protected)
GEMINI_DELAY = 3.5             # Max 1 request per 0.5 seconds (Prevent Hammering)
sleepgetpapers = 1.75          # Global pacing for generic robust requests

# ============================
# CONFIG
# ============================
st.set_page_config(page_title="📚 AI Literature Helper", page_icon="🤖", layout="wide")

if not os.path.exists(CONFIG_DIR): os.makedirs(CONFIG_DIR)

# --- API KEYS ---
# NOTE: In production, these should be environment variables or user inputs.
SEMANTIC_SCHOLAR_API_KEY = "It2pKMHpTK7l5lnOhPUKE4ldBA3Lzeq82hHEsbnB"
GEMINI_API_KEY = "AIzaSyDFlxX_6iRUmSX8A3bvQDyChf8Fdl9EKZA"
NCBI_EMAIL = "reggcrowmell@gmail.com"
NCBI_API_KEY = "89fb3103db9bd0586c75a45d0c6a65618108"

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
    "models/gemini-2.0-flash-lite-preview-02-05": {
        "model_id": "gemini-2.0-flash-lite-preview-02-05",
        "description": "Newest Lite model. Fast and cost-effective."
    },
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
    "herbal drug discovery", "pharmaceutical drug discovery", "biology", "phytochemical", 
    "Not drug development", "non herbal drug development", "other -  non-medical", "traditional medicine"
]

# Structure: (Sentence, Enabled, Pass_State, Custom_Tag)
DEFAULT_SEMANTIC_SENTENCES = [
    ("The study evaluates the pharmacological activity or therapeutic efficacy of isolated phytochemicals or herbal extracts.", True, True, "Bioactivity"),
    ("This research identifies bioactive secondary metabolites or chemical constituents derived from medicinal plants.", True, True, "Chemistry"),
    ("The paper reports on the dose-dependent effects or mechanisms of action of natural products in biological models.", True, True, "Mechanism")
]

# ============================
# SESSION STATE INIT (HISTORY & CACHE)
# ============================
if "results_history" not in st.session_state:
    st.session_state.results_history = []

if "search_cache" not in st.session_state:
    st.session_state.search_cache = {}

if "cycle_state" not in st.session_state:
    st.session_state.cycle_state = {
        "active": False,
        "query_idx": 0,
        "paper_offset": 0,
        "query_stats": {'processed': 0, 'saved': 0, 'bypass_ai': False}
    }

# ============================
# CONFIGURATION MANAGER (SAVE/LOAD)
# ============================
def save_full_state(filename):
    """Saves every user-accessible setting and cycle state to a named JSON file."""
    if not filename.endswith(".json"): filename += ".json"
    path = os.path.join(CONFIG_DIR, filename)
    
    # Snapshot Session State
    state_dump = {
        "ver": "2.3", # Version Bump
        "cycle_state": st.session_state.get("cycle_state", {
            "active": False, "query_idx": 0, "paper_offset": 0,
            "query_stats": {'processed': 0, 'saved': 0, 'bypass_ai': False}
        }),
        "topics_txt": st.session_state.get("topics_txt", ""),
        "authors_txt": st.session_state.get("authors_txt", ""),
        "user_zotero_id": st.session_state.get("user_zotero_id", ""),
        "user_zotero_collection": st.session_state.get("user_zotero_collection", ""),
        "zotero_api_key": st.session_state.get("zotero_api_key", ""),
        "allow_duplicates": st.session_state.get("allow_duplicates", False),
        "add_to_zotero_state": st.session_state.get("add_to_zotero_state", True),
        "enable_speedup_checkbox": st.session_state.get("enable_speedup_checkbox", False),
        "use_boolean_checkbox": st.session_state.get("use_boolean_checkbox", False),
        "skip_low_yield_checkbox": st.session_state.get("skip_low_yield_checkbox", False),
        "abstract_length_slider": st.session_state.get("abstract_length_slider", 150),
        "min_score3_slider": st.session_state.get("min_score3_slider", 2),
        "speedup_threshold_slider": st.session_state.get("speedup_threshold_slider", 9),
        "vector_score_min_slider": st.session_state.get("vector_score_min_slider", 1.50),
        "composite_score_min_slider": st.session_state.get("composite_score_min_slider", 0.0),
        "max_results_slider": st.session_state.get("max_results_slider", 20),
        "semantic_sentences": st.session_state.get("semantic_sentences", DEFAULT_SEMANTIC_SENTENCES),
        "ai_tag_categories_list": st.session_state.get("ai_tag_categories_list", DEFAULT_AI_TAG_CATEGORIES),
        "ai_tag_post_filter_values": st.session_state.get("ai_tag_post_filter_values", []),
        "automated_queries": st.session_state.get("automated_queries", []),
        "model_key_selector": st.session_state.get("model_key_selector", ""),
        "ollama_local_model_selector": st.session_state.get("ollama_local_model_selector", ""),
        "search_mode_selector": st.session_state.get("search_mode_selector", "Keyword Search"),
        "search_source_selector": st.session_state.get("search_source_selector", "Semantic Scholar"),
        "query_mode_selector": st.session_state.get("query_mode_selector", "Single Query")
    }
    
    try:
        with open(path, "w") as f:
            json.dump(state_dump, f, indent=2)
        st.toast(f"✅ Configuration saved: {filename}")
    except Exception as e:
        st.error(f"Save failed: {e}")

def load_full_state(filename):
    """Restores the full GUI state."""
    path = os.path.join(CONFIG_DIR, filename)
    if not os.path.exists(path): return
    
    try:
        data = json.load(open(path))
        for k, v in data.items():
            if k in ["ver"]: continue
            if k == "cycle_state" and isinstance(v, dict):
                st.session_state.cycle_state = v
                st.session_state.cycle_state['active'] = False 
            else:
                st.session_state[k] = v
        st.session_state.current_config_file = filename
        st.toast(f"♻️ Loaded: {filename}")
    except Exception as e:
        st.error(f"Load failed: {e}")

# ============================
# PREFERENCES (LEGACY LOADER)
# ============================
PREFS_FILE = "prefs.json"

def load_prefs():
    default_prefs = {
        "topics": [], "authors": [], "collection_id": "CPEYXPTI", "library_id": "18781930",
        "allow_duplicates": False,
        "automated_queries": [],
        "ai_tag_categories_list": DEFAULT_AI_TAG_CATEGORIES,
        "semantic_sentences": DEFAULT_SEMANTIC_SENTENCES,
        "max_results_value": 1000, "min_score3_value": 2, 
        "min_abstract_length_chars": 68, "vector_score_min_value": 1.50, 
        "composite_score_min_value": 0.0,
        "model_key_selector": "models/gemini-2.0-flash-lite-preview-02-05",
        "search_source_selector": "Semantic Scholar", "query_mode_selector": "Automated Cycle",
        "add_to_zotero_state_value": True, "zotero_api_key_value": "",
        "ai_tag_post_filter_values": [], "enable_speedup_value": True,
        "speedup_threshold_value": 7, "skip_low_yield_value": False
    }
    if not os.path.exists(PREFS_FILE): return default_prefs
    try:
        return json.load(open(PREFS_FILE))
    except:
        return default_prefs

def save_prefs(data):
    try:
        with open(PREFS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save preferences: {e}")

def save_current_settings():
    ollama_sel = st.session_state.get("ollama_local_model_selector", "")
    new_prefs = {
        "topics": [t.strip() for t in st.session_state.get("topics_txt", "").split(",") if t.strip()],
        "authors": [a.strip() for a in st.session_state.get("authors_txt", "").split(",") if a.strip()],
        "collection_id": st.session_state.get("user_zotero_collection", ""),
        "library_id": st.session_state.get("user_zotero_id", ""),
        "allow_duplicates": st.session_state.get("allow_duplicates", False),
        "semantic_sentences": st.session_state.get("semantic_sentences", DEFAULT_SEMANTIC_SENTENCES),
        "min_abstract_length_chars": st.session_state.get("abstract_length_slider", 150),
        "ai_tag_categories_list": st.session_state.get("ai_tag_categories_list", DEFAULT_AI_TAG_CATEGORIES),
        "automated_queries": [
            {**q, 'semantic_sentence': q.get('semantic_sentence', "")} 
            for q in st.session_state.get("automated_queries", [])
        ],
        "max_results_value": st.session_state.get("max_results_slider", 20),
        "min_score3_value": st.session_state.get("min_score3_slider", 2),
        "selected_model_key": st.session_state.get("model_key_selector", "models/gemini-2.0-flash-lite-preview-02-05"), 
        "selected_ollama_model": ollama_sel,
        "search_source_selector": st.session_state.get("search_source_selector", "Semantic Scholar"),
        "search_mode_value": st.session_state.get("search_mode_selector", "Keyword Search"),
        "vector_score_min_value": st.session_state.get("vector_score_min_slider", 1.50),
        "composite_score_min_value": st.session_state.get("composite_score_min_slider", 0.0),
        "query_mode_value": st.session_state.get("query_mode_selector", "Single Query"),
        "add_to_zotero_state_value": st.session_state.get("add_to_zotero_state", True),
        "zotero_api_key_value": st.session_state.get("zotero_api_key", ""),
        "ai_tag_post_filter_values": st.session_state.get("ai_tag_post_filter_values", []),
        "enable_speedup_value": st.session_state.get("enable_speedup_checkbox", False),
        "speedup_threshold_value": st.session_state.get("speedup_threshold_slider", 9),
        "skip_low_yield_value": st.session_state.get("skip_low_yield_checkbox", False)
    }
    save_prefs(new_prefs)
    st.toast("✅ Settings saved.")

# -----------------------------------------------------------

prefs = load_prefs()

# Init Session State
for k, v in prefs.items():
    if k == "topics": k = "topics_txt"; v = ", ".join(v) if isinstance(v, list) else v
    if k == "authors": k = "authors_txt"; v = ", ".join(v) if isinstance(v, list) else v
    if k == "collection_id": k = "user_zotero_collection"
    if k == "library_id": k = "user_zotero_id"
    if k == "zotero_api_key_value": k = "zotero_api_key" 
    
    if k not in st.session_state and "_value" not in k:
        st.session_state[k] = v

if "model_key_selector" not in st.session_state:
    st.session_state.model_key_selector = prefs.get("selected_model_key", "models/gemini-2.0-flash-lite-preview-02-05")
if "ollama_local_model_selector" not in st.session_state:
    st.session_state.ollama_local_model_selector = prefs.get("selected_ollama_model", "")

if "automated_queries" not in st.session_state:
    st.session_state.automated_queries = []

migrated_queries = []
for q in st.session_state.automated_queries:
    if isinstance(q, dict):
        if "vector_min" not in q: q["vector_min"] = 0.50
        if "start_rec" not in q: q["start_rec"] = 0
        if "stop_rec" not in q: q["stop_rec"] = 12 
        if "semantic_sentence" not in q: q["semantic_sentence"] = "" 
        migrated_queries.append(q)
    elif isinstance(q, str):
        migrated_queries.append({
            "query": q, "folder": "", "semantic_sentence": "",
            "vector_min": 0.50, "start_rec": 0, "stop_rec": 12 
        })
st.session_state.automated_queries = migrated_queries

if "ai_tag_categories_list" not in st.session_state:
    st.session_state.ai_tag_categories_list = prefs.get("ai_tag_categories_list", DEFAULT_AI_TAG_CATEGORIES)
if "semantic_sentences" not in st.session_state:
    st.session_state.semantic_sentences = prefs.get("semantic_sentences", DEFAULT_SEMANTIC_SENTENCES)
if "ai_tag_post_filter_values" not in st.session_state:
    st.session_state.ai_tag_post_filter_values = []

# ============================
# CORE UTILITY FUNCTIONS
# ============================

def save_cycle_results_to_disk(query_str, results, status_tag=""):
    """Saves results to disk with an optional status tag (e.g., _INCOMPLETE)."""
    if not os.path.exists("saved_results"):
        os.makedirs("saved_results")
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_query = "".join([c for c in query_str if c.isalpha() or c.isdigit() or c==' ']).strip().replace(" ", "_")
    
    suffix = f"_{status_tag}" if status_tag else ""
    filename = f"saved_results/{timestamp}_{safe_query}{suffix}.json"
    
    try:
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        logging.info(f"✅ Saved cycle results to {filename}")
        return True
    except Exception as e:
        logging.error(f"Failed to save cycle results: {e}")
        return False

def _request_json_with_retries(url, *, method="GET", headers=None, params=None, data=None, tries=4, timeout=40):
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
            if attempt == tries: raise
            logging.warning(f"NETWORK BACKOFF: Sleeping for {delay:.2f}s (Attempt {attempt}/{tries}). Error: {e}")
            sleep(delay)
            delay = min(delay * 2, 3.0)
    return {}

# --- OLLAMA HELPERS ---
def get_ollama_models():
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/tags", timeout=2)
        if resp.status_code == 200:
            models = resp.json().get('models', [])
            return [m['name'] for m in models]
    except Exception:
        pass
    return []

def query_ollama_chat(model, prompt, temperature=0.3):
    url = f"{OLLAMA_BASE_URL}/chat"
    payload = {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "stream": True, "options": {"temperature": temperature}
    }
    GLOBAL_TIMEOUT_BUDGET = 300 
    MAX_GEN_TIME_PER_ATTEMPT = 180 
    global_start_time = time()

    while (time() - global_start_time) < GLOBAL_TIMEOUT_BUDGET:
        attempt_start_time = time()
        full_response = ""
        generation_complete = False
        try:
            with requests.post(url, json=payload, stream=True, timeout=10) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if (time() - attempt_start_time) > MAX_GEN_TIME_PER_ATTEMPT: break
                    if (time() - global_start_time) > GLOBAL_TIMEOUT_BUDGET: return "ERROR: OLLAMA TIMEOUT"
                    if line:
                        try:
                            body = json.loads(line)
                            content = body.get('message', {}).get('content', '')
                            print(content, end="", flush=True)
                            full_response += content
                            if body.get('done', False):
                                generation_complete = True
                                break
                        except json.JSONDecodeError: pass
            if generation_complete: return full_response
        except Exception as e:
            remaining = GLOBAL_TIMEOUT_BUDGET - (time() - global_start_time)
            if remaining > 0: sleep(5)
            else: return f"OLLAMA ERROR: {e}"
    return "ERROR: OLLAMA TIMEOUT"

# --- GEMINI HELPERS ---
def handle_gemini_backoff(error_msg: str):
    match = re.search(r"retry in\s*([\d\.]+)\s*s", error_msg, re.IGNORECASE)
    if match:
        try:
            wait_time = float(match.group(1))
            logging.warning(f"GEMINI QUOTA EXHAUSTED: Pausing for {wait_time}s...")
            sleep(wait_time + 1.5)
            return True
        except ValueError: pass
    if "429" in error_msg or "exhausted" in error_msg.lower():
        sleep(5)
        return True
    return False

def clean_snippet(text):
    if not text: return ""
    text = re.sub(r'<[^<]+?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_authors(authors_info):
    if not authors_info: return []
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
    if style == 1: return f"https://remotexs.ntu.edu.sg/user/login?dest={url_or_doi}"
    return f"https://remotexs.ntu.edu.sg/login?url={url_or_doi}"

def gemini_boolean_query(user_prompt, model):
    use_ollama = st.session_state.get("use_ollama", False)
    if not client and not use_ollama:
        return {"boolean_query": build_boolean_query_simple(user_prompt)}
    
    prompt = f"Convert this research topic into a precise Boolean search string (AND, OR, NOT). Topic: \"{user_prompt}\" Output ONLY the boolean string."
    if use_ollama:
        try:
            resp_text = query_ollama_chat(model, prompt)
            if "OLLAMA" in resp_text: return {"boolean_query": build_boolean_query_simple(user_prompt)}
            return {"boolean_query": resp_text.replace('```', '').strip()}
        except Exception:
            return {"boolean_query": build_boolean_query_simple(user_prompt)}

    for attempt in range(3):
        try:
            sleep(GEMINI_DELAY)
            resp = client.models.generate_content(model=model, contents=prompt)
            return {"boolean_query": resp.text.replace('```', '').strip()}
        except Exception as e:
            if attempt < 2 and handle_gemini_backoff(str(e)): continue
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
    if not url: return ""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Referer": "https://www.google.com/"
        }
        try:
            head_resp = requests.head(url, headers=headers, timeout=5)
            if int(head_resp.headers.get('Content-Length', 0)) > 15 * 1024 * 1024: return ""
        except: pass

        r = None
        for attempt in range(1, 5):
            try:
                r = requests.get(url, headers=headers, timeout=20) 
                if r.status_code == 200 and len(r.content) > 500: break
            except Exception: pass
            if attempt < 4: sleep(7)

        if not r or r.status_code != 200: return ""
        
        content_type = r.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type and "html" in content_type:
            if "ncbi.nlm.nih.gov/pmc/articles/PMC" in url or "pmc" in url.lower():
                soup = BeautifulSoup(r.content, 'html.parser')
                pdf_link = soup.find('a', href=re.compile(r'\.pdf$', re.I))
                if pdf_link: return extract_pdf_text(urljoin(url, pdf_link['href']))

        if "application/pdf" in content_type or r.content.startswith(b"%PDF"):
            with io.BytesIO(r.content) as pdf_stream:
                return extract_text(pdf_stream, maxpages=24)
    except Exception: return ""
    return ""

def extract_webpage_text(url):
    if not url or url.endswith('.pdf'): return ""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        if "semanticscholar.org" in url:
            try:
                s2_resp = requests.get(url, headers=headers, timeout=10)
                if s2_resp.status_code == 200:
                    s2_soup = BeautifulSoup(s2_resp.content, 'html.parser')
                    ext_link = s2_soup.find("a", attrs={"data-test-id": "paper-link"})
                    if not ext_link:
                        wrapper = s2_soup.find("div", class_=lambda x: x and "alternate-sources__paperlink-wrapper" in x)
                        if wrapper: ext_link = wrapper.find("a", href=True)
                    if not ext_link:
                        ext_link = s2_soup.find("a", attrs={"data-heap-id": "paper_link_target"})
                    if ext_link and ext_link.get('href'):
                        new_target = ext_link['href']
                        if new_target.startswith('/'): new_target = "https://www.semanticscholar.org" + new_target
                        return extract_webpage_text(new_target)
            except Exception: pass
        
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        if resp.status_code != 200: return ""
        
        soup = BeautifulSoup(resp.content, 'html.parser')
        pdf_meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
        if pdf_meta and pdf_meta.get("content"):
            extracted_pdf_text = extract_pdf_text(pdf_meta["content"])
            if extracted_pdf_text and len(extracted_pdf_text) > 200: return extracted_pdf_text

        for element in soup(["script", "style", "nav", "footer", "header", "meta", "noscript", "aside", "form", "button", "input", "iframe", "svg"]):
            element.decompose()
        
        garbage_triggers = ["copyright", "all rights reserved", "log in", "sign up", "et al.", "doi:", "google scholar", "pubmed"]
        parent_scores = {}
        all_paragraphs = soup.find_all('p')
        
        for p in all_paragraphs:
            text = p.get_text(" ", strip=True)
            word_count = len(text.split())
            if word_count < 20: continue 
            if any(trigger in text.lower() for trigger in garbage_triggers): continue
            sentences = re.split(r'[.!?]\s+(?=[A-Z])', text)
            if len(sentences) < 2: continue 
            
            parent = p.parent
            if parent.name in ['span', 'strong', 'em', 'a', 'b', 'i']: parent = parent.parent
            score = len(text)
            if word_count > 60: score *= 1.5 
            parent_scores[parent] = parent_scores.get(parent, 0) + score

        if not parent_scores: return ""
        best_parent = max(parent_scores, key=parent_scores.get)
        final_blocks = []
        for child in best_parent.descendants:
            if child.name in ['h1', 'h2', 'h3', 'h4', 'h5']:
                header_text = child.get_text().lower()
                if "reference" in header_text or "bibliography" in header_text: break
            if child.name == 'p':
                t = child.get_text(" ", strip=True)
                if len(t.split()) > 15: final_blocks.append(t)
        
        seen = set()
        unique_blocks = []
        for b in final_blocks:
            if b not in seen:
                unique_blocks.append(b)
                seen.add(b)
        return "\n\n".join(unique_blocks)[:40000]
    except Exception: pass
    return ""

def check_zotero_duplicate(doi: str, library_id: str, collection_id: str) -> bool:
    api_key = st.session_state.get("zotero_api_key", "").strip()
    if not doi or not library_id or not api_key: return False
    url = f"https://api.zotero.org/users/{library_id}/items"
    headers = {"Zotero-API-Key": api_key}
    params = {"q": doi, "itemType": "journalArticle", "limit": 1}
    if collection_id: params['collection'] = collection_id
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        if resp.status_code == 200: return len(resp.json()) > 0
    except Exception: pass
    return False

def upload_to_zotero(payload_dict: Dict[str, Any], target_coll_id: str = None) -> tuple[bool, str]:
    connector_endpoint = "http://127.0.0.1:23119/connector/saveItems"
    final_payload = {"items": [payload_dict]}
    if target_coll_id: final_payload["collectionId"] = target_coll_id 
    start_time = time()
    while (time() - start_time) < 600:
        try:
            sleep(0.5)
            z_resp = requests.post(connector_endpoint, json=final_payload, timeout=60)
            z_resp.raise_for_status()
            return True, "Item sent to Zotero."
        except Exception as e:
            sleep(5)
            last_error = str(e)
    return False, f"Zotero Timeout (10m): {last_error}"

def delete_zotero_item(doi):
    library_id = st.session_state.get("user_zotero_id")
    api_key = st.session_state.get("zotero_api_key")
    if not library_id or not api_key: return False, "Missing Library ID or API Key in settings."
    base_url = f"https://api.zotero.org/users/{library_id}/items"
    headers = {"Zotero-API-Key": api_key}
    try:
        search_params = {"q": doi, "itemType": "journalArticle", "limit": 1}
        search_resp = requests.get(base_url, headers=headers, params=search_params, timeout=10)
        search_data = search_resp.json()
        if not search_data: return False, "Item not found in Zotero."
        item_key = search_data[0]['key']
        version = search_data[0]['version']
        headers["If-Match"] = str(version)
        del_resp = requests.delete(f"{base_url}/{item_key}", headers=headers, timeout=10)
        if del_resp.status_code == 204: return True, "Item deleted."
        else: return False, f"Delete failed: {del_resp.status_code} {del_resp.text}"
    except Exception as e: return False, str(e)

def gemini_extract_from_text(text, model):
    use_ollama = st.session_state.get("use_ollama", False)
    if not client and not use_ollama: return []
    prompt = f"Extract academic references from this text into a JSON array. Keys: \"title\", \"authors\" (list), \"year\" (int), \"doi\" (string/null). Text: {text[:15000]}"
    if use_ollama:
        try:
            resp_text = query_ollama_chat(model, prompt)
            js = re.sub(r'```json|```', '', resp_text).strip()
            return json.loads(js)
        except: return []

    for attempt in range(3):
        try:
            sleep(GEMINI_DELAY)
            resp = client.models.generate_content(model=model, contents=prompt)
            js = re.sub(r'```json|```', '', resp.text).strip()
            return json.loads(js)
        except Exception as e:
            if attempt < 2 and handle_gemini_backoff(str(e)): continue
            return []

DOI_RE = re.compile(r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b', re.IGNORECASE)

@st.cache_resource
def get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('allenai/specter2_aug2023refresh_base')
    except ImportError: return None

def get_embedding(text: str):
    model = get_embedding_model()
    if model and text: return model.encode(text)
    return None

def cosine_similarity(v1, v2):
    if v1 is None or v2 is None: return 0.0
    v1, v2 = np.array(v1), np.array(v2)
    dot = np.dot(v1, v2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return 0.0 if norm1 == 0 or norm2 == 0 else dot / (norm1 * norm2)

@st.dialog("Search Vector Inspection")
def show_vector_inspection(query_text, default_min):
    st.caption(f"Inspecting Query: **{query_text}**")
    st.markdown("### 🧪 Test Similarity")
    col_input, col_score = st.columns([3, 1])
    test_text = col_input.text_area("Sample Abstract", height=200)
    if test_text:
        model = get_embedding_model()
        if model:
            q_vec = get_embedding(query_text)
            d_vec = get_embedding(test_text)
            score = cosine_similarity(q_vec, d_vec)
            col_score.metric("Vector Score", f"{score:.4f}")
            if score >= default_min: col_score.success(f"PASS (>= {default_min})")
            else: col_score.error(f"FAIL (< {default_min})")

@st.dialog("📊 Vector Tuning Report")
def run_vector_tuning_report():
    """
    Runs a rapid 'Zero-Shot' vector search.
    A.1 Composite Score Calculation: Sum of Top 3 Net + Sum of Top 3 Raw.
    """
    st.caption("Running semantic comparison on defined range (No AI Analysis / No Saving)...")
    qs = []
    if st.session_state.search_mode_selector == "Keyword Search" and st.session_state.query_mode_selector == "Single Query":
         if st.session_state.user_prompt_input.strip():
             max_r = st.session_state.get('max_results_slider', 20)
             qs = [{'query': st.session_state.user_prompt_input, 'vector_min': st.session_state.vector_score_min_slider, 'start_rec': 0, 'stop_rec': max_r}]
    elif st.session_state.search_mode_selector == "Keyword Search":
        qs = st.session_state.automated_queries

    if not qs:
        st.error("No queries defined or selected.")
        return

    model = get_embedding_model()
    if not model: 
        st.error("Embedding model not loaded.")
        return

    all_results_grouped = {}
    prog_bar = st.progress(0)
    status_text = st.empty()
    
    for i, q_obj in enumerate(qs):
        query_str = q_obj.get('query')
        if not query_str: continue
        try:
            start_rec = int(q_obj.get('start_rec', 0))
            stop_rec = int(q_obj.get('stop_rec', 20))
            limit = stop_rec - start_rec
            if limit <= 0: limit = 20
        except:
            start_rec = 0; limit = 20
            
        status_text.text(f"Scanning: {query_str} (Recs {start_rec}-{stop_rec})...")
        source = st.session_state.search_source_selector
        current_query_results = []

        if source in ["Semantic Scholar", "Both"]:
            for chunk in search_semantic_scholar(query_str, limit, start_rec):
                if chunk:
                    ranked = prerank_papers(chunk, query_str, model)
                    for p in ranked:
                        # A.2 Semantic Score Retrieval
                        sem_tags_raw = generate_semantic_tags(p.get('snippet', ''), st.session_state.semantic_sentences, model, top_n=3)
                        
                        s1_net = sem_tags_raw[0][0] if len(sem_tags_raw) > 0 else 0.0
                        s1_raw = sem_tags_raw[0][1] if len(sem_tags_raw) > 0 else 0.0
                        s2_net = sem_tags_raw[1][0] if len(sem_tags_raw) > 1 else 0.0
                        s2_raw = sem_tags_raw[1][1] if len(sem_tags_raw) > 1 else 0.0
                        s3_net = sem_tags_raw[2][0] if len(sem_tags_raw) > 2 else 0.0
                        s3_raw = sem_tags_raw[2][1] if len(sem_tags_raw) > 2 else 0.0
                        
                        # A.1 Composite Score Calculation
                        composite_score = (s1_net + s2_net + s3_net) + (s1_raw + s2_raw + s3_raw)
                        
                        p['Sem_Top1_Tag'] = sem_tags_raw[0][2] if len(sem_tags_raw) > 0 else "N/A"
                        current_query_results.append({
                            "Score": p.get('vector_score', 0),
                            "Composite_Score": composite_score, # A.3 Data Injection
                            "Sem_Top1_Score_Net": s1_net, "Sem_Top1_Raw": s1_raw,
                            "Sem_Top2_Score_Net": s2_net, "Sem_Top2_Raw": s2_raw,
                            "Sem_Top3_Score_Net": s3_net, "Sem_Top3_Raw": s3_raw,
                            "Title": p.get('title', "Untitled"),
                            "Abstract": p.get('snippet', 'N/A'),
                            "Source": "S2", "Query": query_str,
                        })
        
        if source in ["PubMed", "Both"]:
            chunk = search_pubmed_paged(query_str, limit, start_rec)
            if chunk:
                 ranked = prerank_papers(chunk, query_str, model)
                 for p in ranked:
                    sem_tags_raw = generate_semantic_tags(p.get('snippet', ''), st.session_state.semantic_sentences, model, top_n=3)
                    s1_net = sem_tags_raw[0][0] if len(sem_tags_raw) > 0 else 0.0
                    s1_raw = sem_tags_raw[0][1] if len(sem_tags_raw) > 0 else 0.0
                    s2_net = sem_tags_raw[1][0] if len(sem_tags_raw) > 1 else 0.0
                    s2_raw = sem_tags_raw[1][1] if len(sem_tags_raw) > 1 else 0.0
                    s3_net = sem_tags_raw[2][0] if len(sem_tags_raw) > 2 else 0.0
                    s3_raw = sem_tags_raw[2][1] if len(sem_tags_raw) > 2 else 0.0
                    
                    composite_score = (s1_net + s2_net + s3_net) + (s1_raw + s2_raw + s3_raw)
                    p['Sem_Top1_Tag'] = sem_tags_raw[0][2] if len(sem_tags_raw) > 0 else "N/A"

                    current_query_results.append({
                        "Score": p.get('vector_score', 0),
                        "Composite_Score": composite_score,
                        "Sem_Top1_Score_Net": s1_net, "Sem_Top1_Raw": s1_raw,
                        "Sem_Top2_Score_Net": s2_net, "Sem_Top2_Raw": s2_raw,
                        "Sem_Top3_Score_Net": s3_net, "Sem_Top3_Raw": s3_raw,
                        "Title": p.get('title', "Untitled"),
                        "Abstract": p.get('snippet', 'N/A'),
                        "Source": "PubMed", "Query": query_str,
                    })
        
        if current_query_results:
            final_query_results = current_query_results[:limit]
            final_query_results.sort(key=lambda x: x['Score'], reverse=True)
            all_results_grouped[query_str] = final_query_results
        
        prog_bar.progress((i + 1) / len(qs))

    status_text.empty()
    prog_bar.empty()

    if not all_results_grouped:
        st.warning("No papers found in the specified range for any query.")
        return

    flat_results = []
    for query, results in all_results_grouped.items():
        for res in results:
            flat_results.append({
                "Query_String": query,
                "Overall_Vector_Score": res['Score'],
                "Composite_Score": res['Composite_Score'], # C.5 CSV Export
                "Semantic_Top1_Net_Score": res['Sem_Top1_Score_Net'],
                "Semantic_Top1_Raw_Score": res['Sem_Top1_Raw'],
                "Semantic_Top2_Net_Score": res['Sem_Top2_Score_Net'],
                "Semantic_Top2_Raw_Score": res['Sem_Top2_Raw'],
                "Semantic_Top3_Score_Net": res['Sem_Top3_Score_Net'],
                "Semantic_Top3_Raw_Score": res['Sem_Top3_Raw'],
                "Title": res['Title'],
                "Abstract": res['Abstract'],
                "Source": res['Source']
            })
            
    df_csv = pd.DataFrame(flat_results)
    csv_output = df_csv.to_csv(index=False).encode('utf-8')
    st.download_button(label="⬇️ Download Full Vector Tuning Report (CSV)", data=csv_output, file_name="vector_tuning_report.csv", mime="text/csv", key="vector_report_download")
    st.markdown("---")

    for query, results in all_results_grouped.items():
        st.markdown(f"#### ➡️ Query: `{query}`")
        df = pd.DataFrame(results)
        report_cols = ["Score", "Composite_Score", "Sem_Top1_Score_Net", "Sem_Top1_Raw", "Sem_Top2_Score_Net", "Sem_Top2_Raw", "Sem_Top3_Score_Net", "Sem_Top3_Raw", "Title", "Abstract", "Source"]
        st.dataframe(
            df[report_cols],
            column_config={
                "Score": st.column_config.ProgressColumn("Vector Score", format="%.4f", min_value=0, max_value=1),
                "Composite_Score": st.column_config.ProgressColumn("COMPOSITE SCORE (0-6)", format="%.3f", min_value=-3.0, max_value=6.0),
                "Sem_Top1_Score_Net": st.column_config.NumberColumn("S1 Net", format="%.3f"),
                "Sem_Top1_Raw": st.column_config.NumberColumn("S1 Raw", format="%.3f"),
                "Sem_Top2_Score_Net": st.column_config.NumberColumn("S2 Net", format="%.3f"),
                "Sem_Top2_Raw": st.column_config.NumberColumn("S2 Raw", format="%.3f"),
                "Sem_Top3_Score_Net": st.column_config.NumberColumn("S3 Net", format="%.3f"),
                "Sem_Top3_Raw": st.column_config.NumberColumn("S3 Raw", format="%.3f"),
                "Title": st.column_config.TextColumn("Paper Title", width="medium"),
                "Abstract": st.column_config.TextColumn("Snippet", width="large"),
                "Source": st.column_config.TextColumn("DB", width="small")
            },
            use_container_width=True, height=min(len(results) * 35 + 50, 500), hide_index=True
        )


def generate_semantic_tags(abstract_text: str, semantic_sentences: List[tuple], semantic_model, top_n: int = 3):
    """
    A.2 Semantic Score Retrieval: Returns tuples of (Net_Contribution, Raw_Similarity, CustomTag, MatchSentence)
    """
    enabled_entries = [s for s in semantic_sentences if s[1]]
    if not abstract_text or not enabled_entries or not semantic_model: return []
    try:
        enabled_texts = [s[0] for s in enabled_entries]
        abs_emb = semantic_model.encode(abstract_text)
        sent_embs = semantic_model.encode(enabled_texts)
        sims_with_tags = []
        from numpy import dot
        from numpy.linalg import norm
        POSITIVE_WEIGHT = 1.0
        NEGATIVE_WEIGHT = 0.5 
        
        for idx, s_emb in enumerate(sent_embs):
            sim = dot(abs_emb, s_emb) / (norm(abs_emb) * norm(s_emb))
            _, _, pass_state, custom_tag = enabled_entries[idx]
            match_sentence = enabled_entries[idx][0]
            clean_tag = re.sub(r'[^a-zA-Z0-9_\-]+', '', custom_tag)
            
            if pass_state: contribution = sim * POSITIVE_WEIGHT
            else: contribution = - (sim * NEGATIVE_WEIGHT)
            
            sims_with_tags.append((contribution, sim, f"sTag-{clean_tag}", match_sentence))
            
        sims_with_tags.sort(key=lambda x: x[0], reverse=True)
        return sims_with_tags[:top_n]
    except Exception as e:
        logging.error(f"Semantic Tag Error: {e}")
        return []


def _chunks(seq, n):
    for i in range(0, len(seq), n): yield seq[i:i+n]

def remove_think_tags(text):
    if not text: return ""
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    if '</think>' in clean_text:
        parts = clean_text.split('</think>')
        if parts[-1].strip(): clean_text = parts[-1]
        else:
            if len(parts) > 1: clean_text = parts[0]
    return clean_text.strip()

def gemini_annotate_paper(title, authors_info, snippet, pdf_text, url, user_query, model, suggested_tags=None, priority_topics=None):
    use_ollama = st.session_state.get("use_ollama", False)
    tags_instruction = ""
    if suggested_tags and len(suggested_tags) > 0: tags_instruction = f"OFFICIAL TAG LIST: {', '.join(suggested_tags)}\nSelect relevant tags from this list."
    else: tags_instruction = "Generate 2-8 relevant tags."
    context_text = pdf_text[:40000] if pdf_text else 'N/A'

    prompt = f"""
    Analyze this scientific paper for a structured report.
    METADATA: Title: {title} Abstract: {snippet} Full Text Segment: {context_text} User Query: {user_query}
    INSTRUCTIONS:
    1. ANALYZE RELEVANCE (0-3). Score based on alignment with phytochemicals, herbs, natural products.
    2. EXTRACT TAGS (Comma separated). {tags_instruction}
    3. SUMMARIZE findings (Bullet points).
    4. EXTRACT SUBSTANCES (Phytochemicals and Plants).
    OUTPUT FORMAT (STRICTLY USE THESE HEADERS): ###SCORE### <0-3> ###TAGS### <tags> ###SUMMARY### <text> ###PHYTOCHEMICALS### <text> ###PLANTS### <text> ###possible_PLANTS### <text>
    """
    raw_text = ""
    try:
        if use_ollama: raw_text = query_ollama_chat(model, prompt)
        elif client:
            for attempt in range(3):
                try:
                    sleep(GEMINI_DELAY)
                    resp = client.models.generate_content(model=model, contents=prompt)
                    raw_text = resp.text
                    break
                except Exception as e:
                    if attempt < 2 and handle_gemini_backoff(str(e)): continue
                    return f"API FAILURE: {e}", [], 0, "None", "None", "None"
        else: return "API KEY MISSING", [], 0, "None", "None", "None"
    except Exception as e: return f"EXECUTION ERROR: {e}", [], 0, "None", "None", "None"

    # --- PREAMBLE RECOVERY LOGIC ---
    # Capture text before the first header, which often contains the AI's generated abstract
    # if the input snippet was empty.
    preamble_match = re.search(r"^(.*?)(?=###\s*SCORE\s*###)", remove_think_tags(raw_text), re.DOTALL | re.IGNORECASE)
    preamble_text = preamble_match.group(1).strip() if preamble_match else ""

    text = remove_think_tags(raw_text)
    if "###SCORE###" not in text and "### SCORE ###" not in text: return "API FAILURE: No Score Returned", [], 0, "None", "None", "None"

    # Extract Score
    # UPDATED REGEX: Handles cases where AI adds text like "<0-3>" or "The score is..." before the number
    score_matches = re.findall(r"(?:\*\*|)?###\s*SCORE\s*###(?:[^\d]*?)(\d)", text, re.DOTALL | re.IGNORECASE)
    score = 0
    if score_matches:
        try:
            score = int(score_matches[-1])
            if score > 3: score = 3
        except: pass

    tags_matches = re.findall(r"(?:\*\*|)?###\s*TAGS\s*###(?:\*\*|)?\s*(.*?)(?=\n(?:\*\*|)?###|$)", text, re.DOTALL | re.IGNORECASE)
    tags = []
    if tags_matches: tags = [t.strip() for t in tags_matches[-1].strip().split(',') if t.strip()]

    summary_matches = re.findall(r"(?:\*\*|)?###\s*SUMMARY\s*###(?:\*\*|)?\s*(.*?)(?=\n(?:\*\*|)?###|$)", text, re.DOTALL | re.IGNORECASE)
    summary_txt = summary_matches[-1].strip() if summary_matches else "Analysis unavailable."

    chems_matches = re.findall(r"(?:\*\*|)?###\s*PHYTOCHEMICALS\s*###(?:\*\*|)?\s*(.*?)(?=\n(?:\*\*|)?###|$)", text, re.DOTALL | re.IGNORECASE)
    chemicals = chems_matches[-1].strip() if chems_matches else "None listed"

    plants_matches = re.findall(r"(?:\*\*|)?###\s*PLANTS\s*###(?:\*\*|)?\s*(.*?)(?=\n(?:\*\*|)?###|$)", text, re.DOTALL | re.IGNORECASE)
    plants = plants_matches[-1].strip() if plants_matches else "None listed"
    
    possplants_matches = re.findall(r"(?:\*\*|)?###\s*possible_PLANTS\s*###(?:\*\*|)?\s*(.*?)(?=\n(?:\*\*|)?###|$)", text, re.DOTALL | re.IGNORECASE)
    possplants = possplants_matches[-1].strip() if possplants_matches else "None listed"

    clean_report = f"**AI Abstract Summary**\n{summary_txt}\n\n---\n**\U0001F9EA Substances & Plants**\n*   **PhytoChemicals:** {chemicals}\n*   **Plants:** {plants}\n*   **possible_Plants:** {possplants}\n"
    
    # Return 7 values (including preamble)
    return clean_report, tags, score, chemicals, plants, possplants, preamble_text


def gemini_abstract_fallback(title, authors_info, current_snippet, model, full_text=None):
    use_ollama = st.session_state.get("use_ollama", False)
    if full_text and len(full_text) > 500:
        prompt = f"Summarize the following academic text into a concise abstract. TEXT SOURCE: {full_text[:15000]}"
    else:
        prompt = f"Generate a 3-sentence abstract for this paper based on metadata. Title: {title} Authors: {authors_info} Output ONLY text."
    
    if use_ollama:
        try: return query_ollama_chat(model, prompt)
        except Exception: return current_snippet
    if client:
        for attempt in range(3):
            try:
                sleep(GEMINI_DELAY)
                resp = client.models.generate_content(model=model, contents=prompt)
                return resp.text.strip()
            except Exception as e:
                if attempt < 2 and handle_gemini_backoff(str(e)): continue
                return current_snippet
    return current_snippet

def prerank_papers(papers_meta, user_prompt, semantic_model):
    if not papers_meta or not user_prompt or not semantic_model: return papers_meta
    q_vec = get_embedding(user_prompt)
    for p in papers_meta:
        doc = f"{p.get('title','')} {p.get('snippet','')}"
        d_vec = get_embedding(doc)
        p['vector_score'] = cosine_similarity(q_vec, d_vec)
    papers_meta.sort(key=lambda x: x.get('vector_score', -1.0), reverse=True)
    return papers_meta

# ============================
# UI Helper functions 
# ============================
def add_new_sentence():
    ns = st.session_state.new_sentence_input.strip()
    if ns:
        default_tag = ns.split()[0][:15]
        st.session_state.semantic_sentences.append((ns, True, True, default_tag))
        st.session_state.new_sentence_input = ""
        save_current_settings()

def set_view_expand_all(): st.session_state.results_view_mode = "expand_all"
def set_view_collapse_all(): st.session_state.results_view_mode = "collapse_all"
def set_view_expand_saved(): st.session_state.results_view_mode = "expand_saved"
def set_view_expand_others(): st.session_state.results_view_mode = "expand_others"

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
        st.session_state.automated_queries.append({"query": q, "folder": ""})
        st.session_state.new_query_input = ""
        save_current_settings()

# ============================
# SEARCH PROVIDERS
# ============================
def search_semantic_scholar(query: str, limit: int, start_offset: int = 0) -> Iterator[List[Dict[str, Any]]]:
    if not query or not query.strip(): return 
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
    offset = start_offset
    target_stop = start_offset + limit
    st.info(f"🔎 Starting Search for: '{query}' (Records {start_offset} to {target_stop})")
    seen_dois = set()

    while offset < target_stop:
        remaining = target_stop - offset
        if remaining <= 0: break
        page_limit = min(100, remaining)
        params = {
            "query": query, "limit": page_limit, "offset": offset,
            "fields": "title,authors,url,abstract,openAccessPdf,externalIds,venue,year,citationCount,publicationDate,publicationTypes"
        }
        try:
            cache_key = f"{query}_{offset}_{page_limit}"
            if cache_key in st.session_state.search_cache: data = st.session_state.search_cache[cache_key]
            else:
                sleep(SEMANTIC_SCHOLAR_DELAY)
                data = _request_json_with_retries(url, params=params, headers=headers)
                if data and data.get("data"): st.session_state.search_cache[cache_key] = data
            
            if not data or not data.get("data"): break
            raw_papers = data.get("data", [])
            page_recs = []
            for paper in raw_papers:
                doi = paper.get("externalIds", {}).get("DOI")
                if doi in seen_dois and doi is not None: continue
                if doi: seen_dois.add(doi)
                paper_title = paper.get("title") or "Untitled Paper"
                abstract_text = clean_snippet(paper.get("abstract",""))
                page_recs.append({
                    "title": paper_title,
                    "url": paper.get("url", "") or (f"https://doi.org/{doi}" if doi else ""),
                    "authors_info": ", ".join([a.get("name","") for a in paper.get("authors",[])]),
                    "snippet": abstract_text,
                    "pdf_url": (paper.get("openAccessPdf") or {}).get("url",""),
                    "doi": doi,
                    "venue": paper.get("venue"),
                    "year": paper.get("year"),
                    "original_abstract": abstract_text
                })
            for chunk in _chunks(page_recs, PROCESSING_BATCH_SIZE): yield chunk
            offset += page_limit
            if data.get('total') and offset >= data.get('total'): break
        except Exception as e:
            logging.error(f"S2 Search Error: {e}")
            break

def search_semantic_scholar_by_doi(doi: str):
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
    params = {"fields": "title,authors,url,abstract,openAccessPdf,externalIds,venue,year"}
    try:
        sleep(SEMANTIC_SCHOLAR_DELAY)
        p = _request_json_with_retries(url, params=params, headers=headers)
        if not p: return None
        abstract_text = clean_snippet(p.get("abstract", ""))
        return {
            "title": p.get("title") or "Untitled Paper",
            "url": p.get("url", "") or f"https://doi.org/{doi}",
            "authors_info": ", ".join([a.get("name","") for a in p.get("authors",[])]),
            "snippet": abstract_text,
            "pdf_url": (p.get("openAccessPdf") or {}).get("url", ""),
            "doi": (p.get("externalIds") or {}).get("DOI") or doi,
            "year": p.get("year"),
            "venue": p.get("venue"),
            "original_abstract": abstract_text
        }
    except Exception: return None

def search_pubmed_paged(query, limit=10, start_offset=0):
    cache_key = f"PUBMED_{query}_{limit}_{start_offset}"
    if cache_key in st.session_state.search_cache: return st.session_state.search_cache[cache_key]
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    term = (query or "").strip() 
    all_pmids = []
    retstart = start_offset 
    while len(all_pmids) < limit:
        retmax = min(1000, limit - len(all_pmids))
        params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax, "retstart": retstart, "email": NCBI_EMAIL, "api_key": NCBI_API_KEY}
        try:
            r = requests.post(f"{base}/esearch.fcgi", data=params, timeout=30).json()
            ids = r.get("esearchresult", {}).get("idlist", [])
            if not ids: break
            all_pmids.extend(ids)
            retstart += 1000
        except Exception: break
    ids = all_pmids[:limit]
    if not ids: return []
    all_summaries = {}
    for chunk in _chunks(ids, 200):
        sum_params = {"db": "pubmed", "id": ",".join(chunk), "retmode": "json", "email": NCBI_EMAIL, "api_key": NCBI_API_KEY}
        try:
            resp = requests.post(f"{base}/esummary.fcgi", data=sum_params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                result_block = data.get("result", {})
                if "uids" in result_block: del result_block["uids"]
                all_summaries.update(result_block)
        except: pass
    abstracts = {}
    for chunk in _chunks(ids, 200):
        try:
            ef_params = {"db": "pubmed", "retmode": "xml", "email": NCBI_EMAIL, "api_key": NCBI_API_KEY}
            ef = requests.post(f"{base}/efetch.fcgi", data={"id": ",".join(chunk)}, params=ef_params, timeout=60)
            root = ET.fromstring(ef.text)
            for art in root.findall(".//PubmedArticle"):
                pmid = art.findtext(".//PMID")
                nodes = art.findall(".//Abstract/AbstractText")
                if nodes: txt = " ".join([n.text for n in nodes if n.text])
                else: txt = art.findtext(".//Abstract") or ""
                if pmid: abstracts[pmid] = clean_snippet(txt)
        except Exception: pass
    out = []
    for pmid in ids:
        r = all_summaries.get(pmid, {})
        doi = None
        for aid in r.get("articleids", []):
            if aid.get("idtype") == "doi": doi = aid.get("value"); break
        abstract_text = abstracts.get(pmid, "")
        out.append({
            "title": r.get("title") or "Untitled Paper",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "authors_info": ", ".join([a.get("name","") for a in r.get("authors", [])]),
            "snippet": abstract_text,
            "doi": doi,
            "venue": r.get("fulljournalname"),
            "year": int(r.get("pubdate", "0")[:4]) if r.get("pubdate") else None,
            "original_abstract": abstract_text
        })
    st.session_state.search_cache[cache_key] = out
    return out

def search_by_url_doi_pdf(url_or_doi):
    if not url_or_doi: return []
    m = DOI_RE.search(url_or_doi)
    if m:
        p = search_semantic_scholar_by_doi(m.group(0))
        if p: return [p]
    if url_or_doi.startswith("http"):
        is_pdf, text = False, ""
        if url_or_doi.endswith(".pdf"):
             text = extract_pdf_text(url_or_doi); is_pdf = True
        else:
            try:
                h = requests.head(url_or_doi, timeout=5)
                if "pdf" in h.headers.get("Content-Type", ""): text = extract_pdf_text(url_or_doi); is_pdf = True
            except: pass
        if is_pdf and text:
            m = DOI_RE.search(text)
            if m:
                p = search_semantic_scholar_by_doi(m.group(0))
                if p: 
                    p["pdf_text"] = text
                    return [p]
            extracted_snippet = text[:500]
            return [{
                "title": "Extracted PDF", "url": url_or_doi, "snippet": extracted_snippet,
                "pdf_text": text, "snippet_source": "PDF_EXTRACT", "original_abstract": extracted_snippet
            }]
    return []

# ============================
# DISPLAY & PROCESSING
# ============================
def _display_cycle_header(item):
    if item.get('type') == 'separator':
        st.markdown("<div style='margin-top: 30px; margin-bottom: 30px;'><hr style='border: 0; border-top: 6px solid #000000; border-radius: 5px;'><h3 style='text-align: center; color: #333;'>🚀 NEW RUN SERIES INITIATED</h3></div>", unsafe_allow_html=True)
    elif item.get('type') == 'cycle_header':
        with st.container():
            st.markdown(f"<div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 6px solid #4CAF50; margin-bottom: 10px;'><h2 style='margin:0; color: #1E1E1E;'>🔄 CYCLE: {item['query']}</h2></div>", unsafe_allow_html=True)
            s = item['settings']
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Vector Min", s['vec_min'])
            c2.metric("Comp Min", s['comp_min'])
            c3.metric("Start Rec", s['start'])
            c4.metric("Stop Rec", s['stop'])
            c5.metric("Source", s['source'])
            st.caption(f"**Semantic Sentence:** {s.get('semantic_sentence', 'None')}")
            st.markdown("---")

def _display_paper_details(paper_data, idx_key):
    paper = paper_data.get('paper', {})
    score = paper_data.get('score', 0)
    tags = paper_data.get('tags', [])
    ai_report_body = paper_data.get('abstract_ai', "")
    vec_min_used = paper_data.get('vec_min_used', 0.0)
    comp_min_used = paper_data.get('comp_min_used', 0.0)
    sem_top3 = paper_data.get('sem_top3', []) 
    extra_sent = paper_data.get('extra_sent', "")
    coll_id = paper_data.get('collection_id', "N/A")
    range_info = paper_data.get('range_info', "N/A")
    title = paper.get("title") or "Untitled Paper"
    url = paper.get("url", "#")
    doi = paper.get("doi", "N/A")
    status_msg = paper_data.get('status_msg', "") 
    reason = paper_data.get('reason', "")
    
    icon = "✅" if status_msg == "SAVED" else ("🚫" if status_msg == "DUPLICATE" else "⚠️")
    is_expanded = (status_msg == "SAVED")
    view_mode = st.session_state.get("results_view_mode", "default")
    if view_mode == "expand_all": is_expanded = True
    elif view_mode == "collapse_all": is_expanded = False
    elif view_mode == "expand_saved": is_expanded = (status_msg == "SAVED")
    elif view_mode == "expand_others": is_expanded = (status_msg != "SAVED")

    header_label = f"{icon} [Score: {score}/3] {title} | [VecMin: {vec_min_used:.2f} | CompMin: {comp_min_used:.2f}]"
    if status_msg != "SAVED" and status_msg != "DUPLICATE": header_label += f" | {reason}"
    elif status_msg == "SAVED": header_label += " | SAVED"

    st.markdown("<style>.streamlit-expanderHeader {font-size: 1.1rem !important; font-weight: bold !important;}</style>", unsafe_allow_html=True)
    with st.expander(header_label, expanded=is_expanded):
        st.markdown("### 1. Citation Data")
        st.markdown(f"**Authors:** {paper.get('authors_info', 'N/A')}")
        st.markdown(f"**DOI:** [{doi}](https://doi.org/{doi})")
        if url: st.markdown(f"**Link:** [Open Source]({url})")
        
        st.markdown("### 2. Abstracts")
        col_src, col_ai = st.columns(2)
        with col_src:
            st.caption("Source Abstract")
            st.info(paper.get('original_abstract', paper.get('snippet', 'No abstract available.'))) 
        with col_ai:
            st.caption("AI Abstract Summary")
            st.markdown(ai_report_body if ai_report_body and "Duplicate" not in ai_report_body else "Analysis unavailable.")
        
        st.markdown("### 3. Categorization")
        safe_tags = [str(t[0]) if isinstance(t, tuple) and len(t) > 0 else str(t) for t in tags if t]
        st.markdown(f"**Tags:** {', '.join(safe_tags) if safe_tags else 'None detected'}")
        
        st.markdown("---")
        st.markdown("### 4. Tuning & Debug Info")
        t1, t2, t3 = st.columns(3)
        t1.caption(f"**Collection:** {coll_id}")
        t2.caption(f"**Range:** {range_info}")
        t3.caption(f"**Extra Sentence:** {extra_sent if extra_sent else 'None'}")
        if sem_top3:
            st.caption("**Top 3 Zero-Shot Matches:**")
            for i, item in enumerate(sem_top3):
                if len(item) >= 4:
                    net, raw, tag, sent = item
                    st.text(f"{i+1}. [Net: {net:.3f} | Raw: {raw:.3f}] {tag} \n   \"{sent}\"")
        else: st.text("No semantic matches found.")

        st.markdown("---")
        if st.button("📤 Upload to Zotero", key=f"btn_save_{idx_key}"):
            clean_tags = []
            for t in tags:
                val = str(t[0]) if isinstance(t, tuple) and len(t) > 0 else str(t)
                clean_tags.append({"tag": val[:MAX_TAG_LENGTH]})
            tuning_text = f"\n\n[TUNING REPORT]\nCollection: {coll_id}\nRange: {range_info}\nVecMin: {vec_min_used} | CompMin: {comp_min_used}\n"
            if extra_sent: tuning_text += f"Extra Sent: {extra_sent}\n"
            if sem_top3:
                tuning_text += "Top Matches:\n"
                for item in sem_top3:
                    if len(item) >= 4: tuning_text += f"- {item[2]} (Net:{item[0]:.3f}): {item[3]}\n"
            
            # Zotero upload fix: Prioritize original_abstract if snip is empty
            best_abstract = paper.get("original_abstract")
            if not best_abstract: best_abstract = paper.get('snippet')
            
            final_abstract = (best_abstract or "")[:ZOTERO_MAX_ABSTRACT_CHARS] + tuning_text
            item = {
                "itemType": "journalArticle", "title": paper.get("title"), "creators": parse_authors(paper.get("authors_info")),
                "abstractNote": final_abstract, "date": str(paper.get("year") or ""), "tags": clean_tags,
                "url": with_ntu_proxy(doi or paper.get("url")), "DOI": doi
            }
            ok, msg = upload_to_zotero(item, target_coll_id=st.session_state.user_zotero_collection)
            if ok: st.toast(f"✅ Saved to Zotero: {title}")
            else: st.error(f"Zotero Error: {msg}")


def process_chunk_and_save(chunk, query, total_papers, papers_processed, status_ph, prog_ph, query_stats, collection_override=None, vec_min_override=None, composite_min_override=None, extra_semantic_sentence=None, start_rec=0, stop_rec=0):
    annotated = 0
    saved = 0
    target_collection = collection_override if collection_override else st.session_state.user_zotero_collection
    min_len = st.session_state.abstract_length_slider
    z_thresh = st.session_state.min_score3_slider
    
    # B.4 Variable Mapping Fix
    vec_min = float(vec_min_override) if vec_min_override is not None else st.session_state.vector_score_min_slider
    comp_min = float(composite_min_override) if composite_min_override is not None else 0.0
    
    req_vals = [v.lower() for v in st.session_state.ai_tag_post_filter_values]
    suggested_tags = st.session_state.selected_tag_categories
    enable_speedup = st.session_state.enable_speedup_checkbox
    speedup_threshold = st.session_state.speedup_threshold_slider
    skip_low_yield = st.session_state.get("skip_low_yield_checkbox", False)
    
    if st.session_state.get("use_ollama"): actual_model_id = st.session_state.get("ollama_local_model_selector", "llama3")
    else: actual_model_id = MODEL_OPTIONS[st.session_state.model_key_selector]["model_id"]
    topics_context = st.session_state.get("topics_txt", "")
    
    sem_model = get_embedding_model()
    # B.1 CRITICAL FILTER DECOUPLING: No Early Drop Here
    if sem_model: chunk = prerank_papers(chunk, query, sem_model)

    if not chunk: return 0, 0

    for i, paper in enumerate(chunk):
        if query_stats.get('abort', False): break
        start_time = time()
        idx = papers_processed + i
        bypass_ai = query_stats.get('bypass_ai', False)
        doi = paper.get("doi")
        is_dup = False
        if doi and (not st.session_state.allow_duplicates) and check_zotero_duplicate(doi, st.session_state.user_zotero_id, target_collection):
             is_dup = True

        full_text_context = ""
        current_snippet = paper.get("snippet", "")
        
        # Determine if we need to fetch text. 
        # If abstract is missing, we MUST fetch, even in Turbo mode, otherwise Zotero entry is empty.
        needs_fetch = False
        if current_snippet and len(current_snippet) >= min_len:
             paper["snippet_source"] = "METADATA"
        else:
             # If missing/short, we need to fetch. 
             # We allow fetch if: (Not Turbo) OR (Turbo AND Missing Abstract)
             needs_fetch = True
        
        if needs_fetch:
            target_url = paper.get("url") or (f"https://doi.org/{paper.get('doi')}" if paper.get("doi") else "")
            if target_url:
                # We fetch regardless of bypass_ai if the abstract is truly missing
                if not bypass_ai or not current_snippet: 
                    full_text_context = extract_webpage_text(target_url)
                    if not paper.get("original_abstract") and full_text_context: 
                        paper["original_abstract"] = full_text_context[:3000]

        snip = paper.get("snippet", "")
        # Fallback AI summarization is ONLY for non-turbo mode
        if len(snip) < min_len and not bypass_ai:
             snip = gemini_abstract_fallback(paper.get("title"), paper.get("authors_info"), snip, actual_model_id, full_text=full_text_context)
             paper["snippet"] = snip

        sem_tags_results = [] 
        ai_abs, tags, score, chemicals, plants, possplants = "", [], 0, "None", "None", "None"
        preamble_text = ""

        if is_dup:
            ai_abs, tags, score, chemicals, plants = "Duplicate", [], 0, "None", "None"
        elif bypass_ai:
            ai_abs, tags, score, chemicals, plants = "Auto-Saved", ["Speed-Save"], 3, "Skipped", "Skipped"
        else:
            # UPDATED: Capture 7 values (including preamble text)
            ai_abs, tags, score, chemicals, plants, possplants, preamble_text = gemini_annotate_paper(
                paper.get("title"), paper.get("authors_info"), snip, full_text_context, paper.get("url"), query, actual_model_id,
                suggested_tags=suggested_tags, priority_topics=topics_context
            )
            if "API FAILURE" not in ai_abs: annotated += 1
        
        # A.1 & A.4 Data Injection: Calculate Composite Score
        current_composite_score = 0.0
        if sem_model and "API FAILURE" not in ai_abs and not is_dup:
             # Use snippet. If snippet empty and bypass_ai is True, fallback to Title
             text_to_analyze = ai_abs if not bypass_ai else snip
             if bypass_ai and not text_to_analyze.strip():
                 text_to_analyze = paper.get("title", "")
             
             sem_tags_results = generate_semantic_tags(text_to_analyze, st.session_state.semantic_sentences, sem_model, top_n=3)
             for item in sem_tags_results:
                 current_composite_score += (item[0] + item[1]) # Net + Raw
             # Add top tag to tags list
             if sem_tags_results: tags.append(sem_tags_results[0])

        v_score = paper.get('vector_score', 0.0)
        # B.6 Debugging Output & Turbo Logging
        prefix = "[TURBO]" if bypass_ai else "[PAPER]"
        print(f"   {prefix} {paper.get('title')[:40]}... | Vector: {v_score:.4f} (Min: {vec_min}) | Composite: {current_composite_score:.4f} (Min: {comp_min})")
        
        # 1. Capture Source Abstract
        snip = paper.get("snippet", "")
        
        # 3. Check for Empty Abstract in Turbo Mode
        if bypass_ai:
            # If snip is still empty (Source had none, Web fetch failed)
            if not snip or not snip.strip():
                # 3a. Construct Warning
                warning_msg = "no text found from source, no abstract found in source download"
                
                # 3b. FORCE AI GENERATION (Rescue)
                print(f"\n[TURBO RESCUE] Attempting AI generation for: {paper.get('title')}")
                ai_generated_text = gemini_abstract_fallback(
                    paper.get("title"), 
                    paper.get("authors_info"), 
                    "", 
                    actual_model_id
                )
                
                # 3c. Combine & Update
                snip = f"{warning_msg}\n\n[AI GENERATED CONTENT]:\n{ai_generated_text}"
                paper['snippet'] = snip 
                print(f"   -> Generated: {ai_generated_text[:100]}...")
        else:
            # If NOT in Turbo mode, check if we recovered text in the preamble
            if (not snip or not snip.strip()) and preamble_text:
                warning_msg = "no text found from source, no abstract found in source download"
                snip = f"{warning_msg}\n\n[AI GENERATED CONTENT (Preamble Recovery)]:\n{preamble_text}"
                paper['snippet'] = snip
                print(f"   -> Recovered from Preamble: {preamble_text[:100]}...")

        # B.2 & B.3 Primary Filter Change & Qualification Logic (Rescue Logic)
        vector_pass = (v_score >= vec_min)
        composite_pass = (current_composite_score >= comp_min)
        
        passes_value_filter = True
        if req_vals:
            search_text = (str(paper.get("title", "")) + " " + str(paper.get("original_abstract", "")) + " " + str(snip) + " " + str(ai_abs) + " " + " ".join([str(t) for t in tags])).lower()
            passes_value_filter = any(v in search_text for v in req_vals)

        status_msg = "SKIPPED"
        reason = ""
        has_chems = "none" not in chemicals.lower()
        has_plants = "none" not in plants.lower()
        has_poss = "none" not in possplants.lower()
        substance_rejected = not (has_chems or has_plants or has_poss)
            
        if "API FAILURE" in ai_abs:
            status_msg = "SKIPPED"; reason = f"AI Analysis Failed (No Score)"
        elif is_dup: status_msg = "DUPLICATE"
        elif not passes_value_filter: reason = "Failed Value Filter"
        # B.7 Filter Logging Fix
        elif not (vector_pass or composite_pass):
            reason = f"Failed Both Filters (Vec: {v_score:.2f}<{vec_min}, Comp: {current_composite_score:.2f}<{comp_min})"
        elif st.session_state.add_to_zotero_state and score >= z_thresh:
            if substance_rejected: status_msg = "REJECTED"; reason = "No Substances/Plants Found"
            else:
                clean_tags = []
                for t in tags:
                    val = str(t[0]) if isinstance(t, tuple) and len(t) > 0 else str(t)
                    clean_tags.append({"tag": val[:MAX_TAG_LENGTH]})
                tuning_text = f"\n\n[TUNING REPORT]\nCollection: {target_collection}\nRange: {start_rec}-{stop_rec}\nVecMin: {vec_min} | CompMin: {comp_min}\n"
                if extra_semantic_sentence: tuning_text += f"Extra Sent: {extra_semantic_sentence}\n"
                if sem_tags_results:
                    tuning_text += "Top Matches:\n"
                    for item in sem_tags_results:
                        if len(item) >= 4: tuning_text += f"- {item[2]} (Net:{item[0]:.3f}): {item[3]}\n"
                
                # Zotero upload fix: Prioritize original_abstract if snip is empty
                best_abstract = paper.get("original_abstract")
                if not best_abstract: best_abstract = snip
                # Final check for Zotero upload
                if not best_abstract or not best_abstract.strip():
                     best_abstract = "no text found from source, no abstract found in source download"

                final_abstract = (best_abstract or "")[:ZOTERO_MAX_ABSTRACT_CHARS] + tuning_text
                item = {
                    "itemType": "journalArticle", "title": paper.get("title"), "creators": parse_authors(paper.get("authors_info")),
                    "abstractNote": final_abstract, "date": str(paper.get("year") or ""), "tags": clean_tags,
                    "url": with_ntu_proxy(doi or paper.get("url")), "DOI": doi
                }
                ok, msg = upload_to_zotero(item, target_coll_id=target_collection)
                if ok: 
                    saved += 1; status_msg = "SAVED"
                else: st.error(f"Zotero Error: {msg}")
        else:
            if score < z_thresh: reason = f"AI Score ({score}) too low"
            else: reason = "Skipped (Logic Fallthrough)"
        
        if not is_dup and not bypass_ai:
            if "API FAILURE" not in ai_abs: query_stats['processed'] += 1
            if status_msg == "SAVED": query_stats['saved'] += 1
            if enable_speedup and query_stats['processed'] == 10:
                if query_stats['saved'] >= speedup_threshold:
                    query_stats['bypass_ai'] = True
                    st.success("⚡ Smart Speed-Up Triggered!")
                elif skip_low_yield:
                    st.warning(f"🛑 Fail Fast Triggered. Aborting.")
                    query_stats['abort'] = True
                    break

        duration = time() - start_time
        result_entry = {
            'paper': paper, 'score': score, 'tags': tags, 'abstract_ai': ai_abs,
            'z_thresh': z_thresh, 'passes': passes_value_filter, 'status_msg': status_msg,
            'process_time': duration, 'reason': reason, 'vec_min_used': vec_min, 'comp_min_used': comp_min,
            'sem_top3': sem_tags_results, 'extra_sent': extra_semantic_sentence,
            'collection_id': target_collection, 'range_info': f"{start_rec} - {stop_rec}"
        }
        st.session_state.results_history.append(result_entry)
        _display_paper_details(result_entry, len(st.session_state.results_history))
        prog_ph.progress(int((idx / total_papers) * 100) if total_papers else 0)
        st.session_state.cycle_state['paper_offset'] += 1
    return annotated, saved


# ============================
# GUI LAYOUT
# ============================
st.title("📚 AI Literature Helper")
col_mode, col_max = st.columns([1, 0.5])
with col_mode:
    search_mode = st.radio("🔍 Mode", ["Keyword Search", "Paste citation / page text", "Lookup by URL / PDF"], horizontal=True, key="search_mode_selector")
with col_max:
    max_res = st.slider("📄 Max articles (Manual Mode Only)", 5, 1000, value=prefs.get("max_results_value", 20), key="max_results_slider")

if search_mode == "Keyword Search":
    src = st.selectbox("📡 Source", ["Semantic Scholar", "PubMed", "Both"], index=0, key="search_source_selector")
    st.markdown("---")
    st.subheader("🤖 AI Filters")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.slider("📏 Min Abstract Len", 50, 500, value=prefs.get("min_abstract_length_chars", 150), key="abstract_length_slider")
        # C.3 Slider Max Range (Manual)
        st.slider("✅ Vector Score (Manual Mode)", 0.00, 6.00, value=prefs.get("vector_score_min_value", 1.50), step=0.01, format="%.2f", key="vector_score_min_slider")
    with c2:
        st.slider("⭐ Min AI Score", 0, 3, value=prefs.get("min_score3_value", 2), key="min_score3_slider")
        # C.1 New Slider (Manual Mode)
        st.slider("✨ Composite Score Min", 0.00, 6.00, value=prefs.get("composite_score_min_value", 0.0), step=0.01, key="composite_score_min_slider", help="Sum of Top 3 Net + Top 3 Raw Semantic Scores")
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

    with st.expander("⚡ Smart Speed-Up (Auto-Save High Relevance)"):
        c_enable, c_failfast = st.columns(2)
        with c_enable: st.checkbox("Enable Smart Speed-Up", value=prefs.get("enable_speedup_value", False), key="enable_speedup_checkbox", help="If enabled, the system checks the first 10 papers. If enough are saved, it skips AI for the rest.")
        with c_failfast: st.checkbox("❌ Fail Fast (Skip Query)", value=prefs.get("skip_low_yield_value", False), key="skip_low_yield_checkbox", help="If the first 10 papers DO NOT meet the threshold below, the entire query is aborted immediately.")
        st.slider("Threshold (Saved/10)", 5, 10, value=prefs.get("speedup_threshold_value", 9), key="speedup_threshold_slider", help="How many of the first 10 must be saved to trigger speed-up (or skip).")
    
    q_mode = st.radio("📝 Query Mode", ["Single Query", "Automated Cycle"], horizontal=True, key="query_mode_selector")
    if q_mode == "Single Query":
        st.text_input("🔍 Topic", key="user_prompt_input")
        st.checkbox("🔤 Use Boolean", key="use_boolean_checkbox")
    else:
        st.caption("🛠️ **Global Bulk Adjustments**")
        gc1, gc2, gc_reset = st.columns([2, 1, 1])
        with gc1: vec_adjust = st.number_input("Global Vector Adjust (+/-)", value=0.00, step=0.01, format="%.2f", key="global_vec_adj_unique_v2")
        with gc2:
            if st.button("Apply Vec", key="btn_apply_vec"):
                for q in st.session_state.automated_queries:
                    try:
                        current_val = float(q.get('vector_min', 0.50))
                        new_val = max(0.0, min(6.0, current_val + vec_adjust))
                        q['vector_min'] = float(f"{new_val:.2f}")
                    except ValueError: pass 
                st.rerun()
        with gc_reset:
            if st.button("Reset Defaults", key="btn_reset_defaults"):
                for q in st.session_state.automated_queries:
                    q['vector_min'] = 0.50; q['start_rec'] = 0; q['stop_rec'] = 12; q['fail_fast_triggered'] = False
                st.rerun()

        gr1, gr2, gr3 = st.columns([1, 1, 2])
        with gr1: g_start = st.number_input("Set All Start #", min_value=0, step=10, key="glob_start_in_unique_v2")
        with gr2: g_stop = st.number_input("Set All Stop #", min_value=1, value=12, step=10, key="glob_stop_in_unique_v2")
        with gr3:
            if st.button("Apply Range to All", key="btn_apply_range"):
                for q in st.session_state.automated_queries:
                    q['start_rec'] = int(g_start); q['stop_rec'] = int(g_stop)
                st.rerun()
        
        df_list = st.session_state.automated_queries
        current_data = pd.DataFrame(df_list)
        if not current_data.empty:
            if 'vector_min' in current_data.columns: current_data['vector_min'] = current_data['vector_min'].astype(float)
            if 'start_rec' in current_data.columns: current_data['start_rec'] = current_data['start_rec'].astype(int)
            if 'stop_rec' in current_data.columns: current_data['stop_rec'] = current_data['stop_rec'].astype(int)
            if 'semantic_threshold' in current_data.columns: current_data['semantic_threshold'] = current_data['semantic_threshold'].astype(float)

        if "Select" not in current_data.columns: current_data.insert(0, "Select", False)
        if "semantic_sentence" not in current_data.columns: current_data.insert(len(current_data.columns), "semantic_sentence", "")
        if "fail_fast_triggered" not in current_data.columns: current_data["fail_fast_triggered"] = False
        current_data["Status"] = current_data["fail_fast_triggered"].apply(lambda x: "🛑 LOW YIELD" if x else "✅ Ready")

        st.caption("📋 **Query Execution Queue** (Editable)")
        cols = ['Select', 'Status', 'query', 'folder', 'semantic_sentence', 'semantic_threshold', 'vector_min', 'start_rec', 'stop_rec']
        cols = [c for c in cols if c in current_data.columns]
        current_data = current_data[cols]

        edited_df = st.data_editor(
            current_data, num_rows="dynamic", use_container_width=True, width='stretch', key="query_table_editor",
            column_config={
                "Select": st.column_config.CheckboxColumn(width="small"),
                "Status": st.column_config.TextColumn("Run Status", disabled=True, width="small"),
                "query": st.column_config.TextColumn("Query String", required=True, width="large"),
                "folder": st.column_config.TextColumn("Collection ID", width="small"),
                "semantic_sentence": st.column_config.TextColumn("Semantic Sentence (Per Query)", width="medium"), 
                # C.4 Table Max Range & Column Label
                "semantic_threshold": st.column_config.NumberColumn("Composite Min", min_value=0.0, max_value=6.0, step=0.01, format="%.2f"), 
                "vector_min": st.column_config.NumberColumn("Vec Min", min_value=0.0, max_value=6.0, step=0.01, format="%.2f"),
                "start_rec": st.column_config.NumberColumn("Start #", min_value=0, step=10),
                "stop_rec": st.column_config.NumberColumn("Stop #", min_value=1, step=10)
            }, hide_index=True
        )
        
        records = edited_df.to_dict('records')
        valid_records = []
        for r in records:
            if r.get('query') and str(r.get('query')).strip():
                clean_r = {k:v for k,v in r.items() if k not in ['Select', 'Status']}
                clean_r['fail_fast_triggered'] = (r.get('Status') == "🛑 LOW YIELD")
                valid_records.append(clean_r)
        
        def clean_rec(rec_list): return [{k:v for k,v in r.items() if k != 'Select'} for r in rec_list]
        if clean_rec(valid_records) != clean_rec(st.session_state.automated_queries):
             st.session_state.automated_queries = clean_rec(valid_records)

        d_col, _, _ = st.columns([1,2,1])
        if d_col.button("🗑️ Delete Selected"):
            if not edited_df[edited_df.Select == True].empty:
                st.session_state.automated_queries = edited_df[edited_df.Select == False].drop(columns=['Select', 'Status'], errors='ignore').to_dict('records')
                st.rerun()

        st.markdown("---")
        with st.expander("💾 Load / Save Full Configuration", expanded=False):
            c_name, c_save, c_load = st.columns([3, 1, 1])
            files = [f for f in os.listdir(CONFIG_DIR) if f.endswith(".json")]
            existing_file = c_name.selectbox("Select Config", [""] + files, key="config_file_select")
            new_file_name = c_name.text_input("Or create new file name", placeholder="my_research_config")
            target_save_name = new_file_name if new_file_name else existing_file
            if c_save.button("💾 Save") and target_save_name: save_full_state(target_save_name)
            c_load.button("📂 Load", on_click=load_full_state, args=(existing_file,))

elif search_mode == "Paste citation / page text":
    st.text_area("📋 Paste text", height=200, key="paste_text")
else:
    st.text_input("🔗 Paste URL/DOI", key="url_or_doi")

st.markdown("---")
with st.expander("🎯 Semantic Classification Sentences"):
    col_t, col_e, col_p, col_c, col_d = st.columns([5, 1, 1, 3, 1])
    col_t.caption("**Classification Sentence**"); col_e.caption("**Enabled**"); col_p.caption("**Pass**"); col_c.caption("**Custom Tag (Edit)**"); col_d.caption("**Del**")
    for i, (sent, en, pas, tag) in enumerate(st.session_state.semantic_sentences):
        c1, c2, c3, c4, c5 = st.columns([5, 1, 1, 3, 1])
        c1.text(sent)
        c2.button("✅" if en else "❌", key=f"tog_s_{i}", on_click=toggle_sentence, args=(i,))
        c3.button("➡️" if pas else "🚫", key=f"tog_p_{i}", on_click=toggle_pass_state, args=(i,))
        c4.text_input("Tag", value=tag, key=f"tag_edit_{i}", label_visibility="collapsed", on_change=edit_custom_tag, args=(i,))
        c5.button("🗑️", key=f"del_s_{i}", on_click=delete_sentence, args=(i,))
    st.text_input("New Sentence", key="new_sentence_input")
    st.button("Add Sentence", on_click=add_new_sentence)

st.markdown("---")
m_keys = list(MODEL_OPTIONS.keys())
saved_model_key = st.session_state.get("model_key_selector", m_keys[0])
try: default_idx = m_keys.index(saved_model_key)
except ValueError: default_idx = 0
selected_option = st.selectbox("🤖 Model", m_keys, index=default_idx, key="model_key_selector")
is_local_ollama = (MODEL_OPTIONS[selected_option]["model_id"] == "LOCAL_OLLAMA")
st.session_state.use_ollama = is_local_ollama
if is_local_ollama:
    local_models = get_ollama_models()
    if local_models:
        saved_ollama = st.session_state.get("ollama_local_model_selector", "")
        try: ollama_idx = local_models.index(saved_ollama)
        except ValueError: ollama_idx = 0
        local_model_name = st.selectbox("💻 Select Local Ollama Model", local_models, index=ollama_idx, key="ollama_local_model_selector")
    else: st.error("Could not fetch models from Ollama (localhost:11434). Is it running?")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🛠️ Tuning Tools")
    if st.button("🧪 Tune Vector Filter", use_container_width=True, help="Run a quick search to inspect vector scores and calibrate your filter."):
        run_vector_tuning_report()
    st.markdown("---")
    st.markdown("### 📂 History Viewer")
    if not os.path.exists("saved_results"): os.makedirs("saved_results")
    history_files = sorted([f for f in os.listdir("saved_results") if f.endswith(".json")], reverse=True)
    selected_history = st.selectbox("Select Past Cycle", [""] + history_files, key="history_file_selector")
    if st.button("📜 Load Selected History"):
        if selected_history:
            try:
                with open(os.path.join("saved_results", selected_history), "r", encoding='utf-8') as f:
                    st.session_state.results_history = json.load(f)
                st.toast(f"Loaded {len(st.session_state.results_history)} items.")
                st.rerun()
            except Exception as e: st.error(f"Error loading file: {e}")

    st.checkbox("📥 Add to Zotero", key="add_to_zotero_state", value=prefs.get("add_to_zotero_state_value", True))
    st.checkbox("⚠️ Allow Duplicates", key="allow_duplicates", value=prefs.get("allow_duplicates", False))
    st.markdown("---")
    st.text_input("Zotero Cloud API Key (for duplicate check)", type="password", key="zotero_api_key", value=prefs.get("zotero_api_key_value", ""))
    st.text_input("Zotero Library ID", key="user_zotero_id", value=prefs.get("library_id", ""))
    st.text_input("Zotero Collection ID", key="user_zotero_collection", value=prefs.get("collection_id", ""))
    st.text_area("Topics (comma-sep)", value=st.session_state.get("topics_txt", ""), key="topics_txt")
    st.text_area("Authors (comma-sep)", value=st.session_state.get("authors_txt", ""), key="authors_txt")
    if st.button("💾 Save Settings"): save_current_settings()

# ============================
# MAIN EXECUTION
# ============================
def run_cycle_logic(queries):
    prog = st.progress(0); status = st.empty()
    # B.5 Unbound Local Fix: Initialize counters
    total_ann = 0; total_save = 0
    sep_entry = {'type': 'separator'}
    st.session_state.results_history.append(sep_entry)
    _display_cycle_header(sep_entry)
    
    start_q_idx = st.session_state.cycle_state['query_idx']
    all_objs = st.session_state.automated_queries
    query_map = {obj['query']: obj for obj in all_objs}
    
    for q_idx in range(start_q_idx, len(queries)):
        raw_item = queries[q_idx]
        if isinstance(raw_item, dict):
            query_str = raw_item.get("query", "")
            val_vec = raw_item.get("vector_min")
            row_vec_min = float(val_vec) if val_vec is not None and str(val_vec).strip() != "" else st.session_state.vector_score_min_slider
            val_comp = raw_item.get("semantic_threshold")
            # B.4 Variable Mapping Fix (Maps 'semantic_threshold' col to Composite Min)
            row_composite_min = float(val_comp) if val_comp is not None and str(val_comp).strip() != "" else st.session_state.get("composite_score_min_slider", 0.0)
            row_start = int(raw_item.get("start_rec", 0))
            row_stop = int(raw_item.get("stop_rec", 12))
            folder_override = raw_item.get("folder", "").strip()
            row_semantic_sentence = raw_item.get("semantic_sentence", "").strip()
        else:
            query_str = str(raw_item)
            row_vec_min = st.session_state.vector_score_min_slider
            row_composite_min = st.session_state.get("composite_score_min_slider", 0.0)
            row_start = 0; row_stop = st.session_state.max_results_slider
            folder_override = ""; row_semantic_sentence = ""
        
        if not folder_override and query_str in query_map: folder_override = query_map[query_str].get('folder', "").strip()
        st.session_state.cycle_state['query_idx'] = q_idx
        status.info(f"Processing: {query_str}")
        if not query_str or not query_str.strip(): continue

        if st.session_state.get("use_ollama"): actual_model_id = st.session_state.get("ollama_local_model_selector", "llama3")
        else: actual_model_id = MODEL_OPTIONS[st.session_state.model_key_selector]["model_id"]

        header_entry = {
            'type': 'cycle_header', 'query': query_str,
            'settings': {
                'vec_min': f"{row_vec_min:.2f}", 'comp_min': f"{row_composite_min:.2f}",
                'start': row_start, 'stop': row_stop, 'source': st.session_state.search_source_selector,
                'model': actual_model_id.split("/")[-1][:15] + "...", 
                'speedup': st.session_state.enable_speedup_checkbox, 'failfast': st.session_state.get("skip_low_yield_checkbox", False),
                'collection': folder_override if folder_override else "Default", 'mode': search_mode,
                'semantic_sentence': row_semantic_sentence if row_semantic_sentence else "None"
            }
        }
        st.session_state.results_history.append(header_entry)
        _display_cycle_header(header_entry)

        query_stats = st.session_state.cycle_state.get('query_stats', {'processed': 0, 'saved': 0, 'bypass_ai': False, 'abort': False})
        total_limit = row_stop - row_start
        if total_limit <= 0: total_limit = 5
        resume_adder = st.session_state.cycle_state['paper_offset'] if q_idx == start_q_idx else 0
        current_api_offset = row_start + resume_adder
        
        if q_idx != start_q_idx: 
            st.session_state.cycle_state['paper_offset'] = 0
            st.session_state.cycle_state['query_stats'] = {'processed': 0, 'saved': 0, 'bypass_ai': False, 'abort': False}
            query_stats = st.session_state.cycle_state['query_stats']
            current_api_offset = row_start

        remaining_limit = total_limit - resume_adder
        if remaining_limit <= 0: continue
        final_query = query_str
        if search_mode == "Keyword Search" and st.session_state.get("use_boolean_checkbox"):
            res = gemini_boolean_query(query_str, actual_model_id)
            final_query = res.get("boolean_query", query_str)
            st.info(f"Boolean Query: {final_query}")

        if search_mode == "Keyword Search":
            source = st.session_state.search_source_selector
            if source in ["Semantic Scholar", "Both"]:
                for chunk in search_semantic_scholar(final_query, remaining_limit, current_api_offset):
                    if query_stats.get('abort'): break
                    a, s = process_chunk_and_save(
                        chunk, final_query, 100, 0, status, prog, query_stats, 
                        collection_override=folder_override, vec_min_override=row_vec_min, composite_min_override=row_composite_min,
                        extra_semantic_sentence=row_semantic_sentence, start_rec=row_start, stop_rec=row_stop
                    )
                    total_ann += a; total_save += s
                    if query_stats.get('abort'): 
                        if q_idx < len(st.session_state.automated_queries): st.session_state.automated_queries[q_idx]['fail_fast_triggered'] = True
                        st.session_state.cycle_state['query_idx'] = q_idx + 1; st.session_state.cycle_state['paper_offset'] = 0
                        st.session_state.cycle_state['query_stats'] = {'processed': 0, 'saved': 0, 'bypass_ai': False, 'abort': False}
                        st.rerun(); break
            
            if source in ["PubMed", "Both"] and not query_stats.get('abort'):
                 p_res = search_pubmed_paged(final_query, remaining_limit, current_api_offset)
                 total_pubmed = len(p_res); pubmed_processed_count = 0
                 for batch in _chunks(p_res, 50): 
                    if query_stats.get('abort'): break
                    a, s = process_chunk_and_save(
                        batch, final_query, total_pubmed, pubmed_processed_count, status, prog, query_stats, 
                        collection_override=folder_override, vec_min_override=row_vec_min, composite_min_override=row_composite_min,
                        extra_semantic_sentence=row_semantic_sentence, start_rec=row_start, stop_rec=row_stop
                    )
                    total_ann += a; total_save += s; pubmed_processed_count += len(batch)
                    if query_stats.get('abort'): 
                        if q_idx < len(st.session_state.automated_queries): st.session_state.automated_queries[q_idx]['fail_fast_triggered'] = True
                        st.session_state.cycle_state['query_idx'] = q_idx + 1; st.session_state.cycle_state['paper_offset'] = 0
                        st.session_state.cycle_state['query_stats'] = {'processed': 0, 'saved': 0, 'bypass_ai': False, 'abort': False}
                        st.rerun(); break

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
                a, s = process_chunk_and_save(
                    papers, final_query, len(papers), 0, status, prog, query_stats, 
                    collection_override=folder_override, vec_min_override=row_vec_min, composite_min_override=row_composite_min,
                    extra_semantic_sentence=row_semantic_sentence, start_rec=row_start, stop_rec=row_stop
                )
                total_ann += a; total_save += s

        elif search_mode == "Lookup by URL / PDF":
            papers = search_by_url_doi_pdf(st.session_state.url_or_doi)
            if papers:
                a, s = process_chunk_and_save(
                    papers, final_query, len(papers), 0, status, prog, query_stats, 
                    collection_override=folder_override, vec_min_override=row_vec_min, composite_min_override=row_composite_min,
                    extra_semantic_sentence=row_semantic_sentence, start_rec=row_start, stop_rec=row_stop
                )
                total_ann += a; total_save += s

    current_run_results = [item for item in st.session_state.results_history if isinstance(item, dict) and item.get('type') not in ['separator', 'cycle_header']]
    label = queries[start_q_idx].get('query', 'Batch_Run') if isinstance(queries[start_q_idx], dict) else str(queries[start_q_idx])
    if current_run_results: save_cycle_results_to_disk(label, st.session_state.results_history, status_tag="COMPLETE")
    st.success(f"Run Complete. Annotated: {total_ann}, Saved: {total_save}")
    st.session_state.cycle_state = {"active": False, "query_idx": 0, "paper_offset": 0, "query_stats": {'processed': 0, 'saved': 0, 'bypass_ai': False, 'abort': False}}

st.markdown("---")
st.subheader("📝 Results Log (Persists across runs)")
v1, v2, v3, v4 = st.columns(4)
v1.button("➕ Expand All", on_click=set_view_expand_all, use_container_width=True)
v2.button("➖ Collapse All", on_click=set_view_collapse_all, use_container_width=True)
v3.button("✅ Expand Saved", on_click=set_view_expand_saved, use_container_width=True)
v4.button("⚠️ Expand Rejected", on_click=set_view_expand_others, use_container_width=True)

if st.session_state.results_history:
    for idx, item in enumerate(st.session_state.results_history):
        if item.get('type') in ['separator', 'cycle_header']: _display_cycle_header(item)
        else: _display_paper_details(item, f"history_{idx}")
else: st.info("No results yet. Click Go to start.")

c_go, c_pause, c_reset, c_clear = st.columns([2, 1, 1, 2])
start_run = False
is_active = st.session_state.cycle_state['active']
with c_go:
    label = "▶️ Resume Cycle" if is_active else "🚀 Go"
    if st.button(label):
        valid_go = True
        qs = []
        if search_mode == "Keyword Search" and st.session_state.query_mode_selector == "Single Query":
             if not st.session_state.user_prompt_input.strip():
                 st.error("Please enter a topic.")
                 valid_go = False
             else:
                 qs = [{
                    'query': st.session_state.user_prompt_input,
                    'vector_min': st.session_state.vector_score_min_slider,       
                    'semantic_threshold': st.session_state.composite_score_min_slider, 
                    'start_rec': 0, 'stop_rec': st.session_state.get('max_results_slider', 20)
                 }]
        elif search_mode == "Keyword Search":
            if not st.session_state.automated_queries:
                st.error("No queries in table.")
                valid_go = False
            else: qs = st.session_state.automated_queries 

        if valid_go:
            if not is_active:
                st.session_state.cycle_state['paper_offset'] = 0 
                st.session_state.cycle_state['query_idx'] = 0
            st.session_state.cycle_state['active'] = True
            start_run = True

with c_pause:
    if st.button("⏸️ Pause"):
        st.session_state.cycle_state['active'] = False
        if st.session_state.results_history:
            label = "Batch_Run"
            for item in st.session_state.results_history:
                if isinstance(item, dict) and item.get('type') == 'cycle_header':
                    label = item.get('query', 'Batch_Run')
                    break
            save_cycle_results_to_disk(label, st.session_state.results_history, status_tag="INCOMPLETE_PAUSED")
            st.toast("💾 Partial results saved to History.")
        st.warning("Cycle Paused. Partial history saved.")

with c_reset:
    if is_active:
        if st.button("⏹️ Stop & Save"):
            if st.session_state.results_history:
                label = "Batch_Run"
                for item in st.session_state.results_history:
                    if isinstance(item, dict) and item.get('type') == 'cycle_header':
                        label = item.get('query', 'Batch_Run')
                        break
                save_cycle_results_to_disk(label, st.session_state.results_history, status_tag="INCOMPLETE_STOPPED")
                st.toast("💾 Partial results saved.")
            st.session_state.cycle_state = {"active": False, "query_idx": 0, "paper_offset": 0, "query_stats": {'processed': 0, 'saved': 0, 'bypass_ai': False}}
            st.rerun()

with c_clear:
    if st.button("🗑️ Clear Results"):
        st.session_state.results_history = []; st.session_state.search_cache = {}; st.rerun()

if start_run:
    if search_mode != "Keyword Search":
        qs = [{'query': 'Manual Input', 'vector_min': 0.5, 'start_rec':0, 'stop_rec': st.session_state.get('max_results_slider', 20)}] 
    if qs: run_cycle_logic(qs)
    else: st.error("No query selected.")
