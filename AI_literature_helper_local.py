# -*- coding: utf-8 -*- version 1.7 WORKING - valid data passed to zotera.
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
from urllib.parse import urljoin # <--- Added for relative link fixing

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
# Reordered: Gemini 2.0 Flash Lite is first
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
    ("The paper is a research paper on killing infective agents of humans using phytochemicals.", True, True, "phytochemicals"),
    ("The paper is primarily a review or meta-analysis", True, False, "The-paper"),
    ("The paper is primarily about the biology of an organism.", True, False, "The-paper"),
    ("The paper is primarily about the physical location of an organism", True, False, "The-paper"),
    ("The content focuses on a medical study testing the efficacy of a compound to treat an aliment.", True, True, "phytochemicals"),
    ("The content focuses on a study testing efficacy of phytochemicals against an organism.", True, True, "phytochemicals"),
    ("This content is related to things outside of health and medicine.", True, False, "This-content"),
    ("This content is an analysis of medical advise and medical guidelines for doctors.", True, False, "This-content"),
    ("This content analyzes the decision making process.", True, False, "This-content"),
    ("This content discussed the logic of medical diagnosis and unnecessary therapy.", True, False, "This-content"),
    ("This content is an analysis and overview for doctors.", True, False, "This-content"),
    ("This paper details a research investigation seeking a cure.", True, True, "phytochemicals"),
    ("This content contains herbal or herbal compounds being tested for the medicinal value.", True, True, "phytochemicals"),
    ("This content does not contains herbal or herbal compounds being tested for the medicinal value.", True, False, "This-content"),
    ("This content does not explore the medicinal value of herbal or herbal compounds.", True, False, "This-content"),
    ("This content does explore the medicinal value of herbal or herbal compounds.", True, True, "phytochemicals"),
    ("This contens discusses  Carotenoids OR Plant-Derived OR herbal extracts OR phytochemicals OR  Bioactive OR Phytonutrient OR Biologically Active OR Compounds OR ALKALOIDS OR TCM OR polyphenols OR plant extracts OR dose-dependent OR receptors OR synergistic OR phenolic acids OR  coumarins OR  stilbenes OR Terpenoids OR Terpenes OR Glucosinolates OR Organosulfur OR Phytosterols OR Saponins OR flavonoids", True, True, "phytochemicals"),
    ("The content discusses x-rays or radiation therapy or chemotherapy or radiation sickness", True, False, "The")
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
# CONFIGURATION MANAGER (SAVE/LOAD)
# ============================
def save_full_state(filename):
    """Saves every user-accessible setting and cycle state to a named JSON file."""
    if not filename.endswith(".json"): filename += ".json"
    path = os.path.join(CONFIG_DIR, filename)
    
    # Snapshot Session State
    state_dump = {
        "ver": "2.2",
        # Text Fields
        "topics_txt": st.session_state.get("topics_txt", ""),
        "authors_txt": st.session_state.get("authors_txt", ""),
        "user_zotero_id": st.session_state.get("user_zotero_id", ""),
        "user_zotero_collection": st.session_state.get("user_zotero_collection", ""),
        "zotero_api_key": st.session_state.get("zotero_api_key", ""),
        
        # Booleans
        "allow_duplicates": st.session_state.get("allow_duplicates", False),
        "add_to_zotero_state": st.session_state.get("add_to_zotero_state", True),
        "enable_speedup_checkbox": st.session_state.get("enable_speedup_checkbox", False),
        "use_boolean_checkbox": st.session_state.get("use_boolean_checkbox", False),
        
        # Sliders/Numbers (Globals)
        "abstract_length_slider": st.session_state.get("abstract_length_slider", 150),
        "min_score3_slider": st.session_state.get("min_score3_slider", 2),
        "speedup_threshold_slider": st.session_state.get("speedup_threshold_slider", 9),
        # Note: Vector min is now primarily per-query, but we save global for backup
        
        # Lists / Complex
        "semantic_sentences": st.session_state.get("semantic_sentences", DEFAULT_SEMANTIC_SENTENCES),
        "ai_tag_categories_list": st.session_state.get("ai_tag_categories_list", DEFAULT_AI_TAG_CATEGORIES),
        "ai_tag_post_filter_values": st.session_state.get("ai_tag_post_filter_values", []),
        
        # THE CORE: Detailed Queries Table (List of Dicts)
        "automated_queries": st.session_state.get("automated_queries", []),
        
        # Model Selections
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
            st.session_state[k] = v
        
        # Capture the file loaded for the Report Header
        st.session_state.current_config_file = filename
            
        st.toast(f"♻️ Loaded: {filename}")
        # Note: Callback handles refresh, no rerun needed here
    except Exception as e:
        st.error(f"Load failed: {e}")

# ============================
# PREFERENCES (LEGACY LOADER)
# ============================
PREFS_FILE = "prefs.json"

def load_prefs():
    # Only loads basics if no full config loaded
    default_prefs = {
        "topics": [
            "\"Carotenoids\"", "\"Plant-Derived\"", "\"herbal\"", "\"extracts\"", "\"phytochemicals\"", 
            "\" Bioactive\"", "\"Phytonutrient\"", "\"Biologically Active\"", "\"Compounds\"", "\"ALKALOIDS\"", 
            "\"TCM\"", "\"polyphenols\"", "\"plant extracts\"", "\"dose-dependent\"", "\"receptors\"", 
            "\"synergistic\"", "\"phenolic acids\"", "\" coumarins\"", "\" stilbenes\"", "\"Terpenoids\"", 
            "\"Terpenes\"", "\"Glucosinolates\"", "\"Organosulfur\"", "\"Phytosterols\"", "\"Saponins\"", "\"flavonoids\""
        ],
        "authors": [],
        "collection_id": "CPEYXPTI",
        "library_id": "18781930",
        "allow_duplicates": False,
        "automated_queries": [
            {"query": "Lyme ALKALOIDS", "folder": ""}, {"query": "Lyme coumarins", "folder": ""}, 
            {"query": "Lyme dose-dependent", "folder": ""}, {"query": "Lyme flavonoids", "folder": ""}, 
            {"query": "Lyme Glucosinolates", "folder": ""}, {"query": "Lyme \"Molecular docking\"", "folder": ""}, 
            {"query": "Lyme Organosulfur", "folder": ""}, {"query": "Lyme phenolic acids", "folder": ""}, 
            {"query": "Lyme phytochemicals", "folder": ""}, {"query": "Lyme Phytonutrient", "folder": ""}, 
            {"query": "Lyme Phytosterols", "folder": ""}, {"query": "Lyme \"plant extracts\"", "folder": ""}, 
            {"query": "Lyme Plant-Derived", "folder": ""}, {"query": "Lyme polyphenols", "folder": ""}, 
            {"query": "Lyme Saponins", "folder": ""}, {"query": "Lyme stilbenes", "folder": ""}, 
            {"query": "Lyme TCM", "folder": ""}, {"query": "Lyme Terpenes", "folder": ""}, 
            {"query": "Lyme Terpenoids", "folder": ""}, {"query": "Lyme Carotenoids", "folder": ""}
        ],
        "ai_tag_categories_list": DEFAULT_AI_TAG_CATEGORIES,
        "semantic_sentences": DEFAULT_SEMANTIC_SENTENCES,
        "max_results_value": 1000, 
        "min_score3_value": 2, 
        "min_abstract_length_chars": 68, 
        "vector_score_min_value": 0.35, 
        "model_key_selector": "models/gemini-2.0-flash-lite-preview-02-05",
        "search_source_selector": "Semantic Scholar",
        "query_mode_selector": "Automated Cycle",
        "add_to_zotero_state_value": True,
        "zotero_api_key_value": "",
        "ai_tag_post_filter_values": [],
        "enable_speedup_value": True,
        "speedup_threshold_value": 7
    }
    if not os.path.exists(PREFS_FILE):
        return default_prefs
    try:
        return json.load(open(PREFS_FILE))
    except:
        return default_prefs

def save_prefs(data):
    """Saves preferences data to the prefs.json file."""
    try:
        with open(PREFS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save preferences: {e}")

# --- FIX: Define save_current_settings to resolve NameError ---
def save_current_settings():
    """Collects all relevant session state variables and saves them via save_prefs."""
    # Get current Ollama selection if available
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
        "automated_queries": st.session_state.get("automated_queries", []),
        "max_results_value": st.session_state.get("max_results_slider", 20),
        "min_score3_value": st.session_state.get("min_score3_slider", 2),
        
        # --- SAVING MODEL SELECTIONS ---
        "selected_model_key": st.session_state.get("model_key_selector", "models/gemini-2.0-flash-lite-preview-02-05"), 
        "selected_ollama_model": ollama_sel,
        # -------------------------------
        
        "search_source_selector": st.session_state.get("search_source_selector", "Semantic Scholar"),
        "search_mode_value": st.session_state.get("search_mode_selector", "Keyword Search"),
        "vector_score_min_value": st.session_state.get("vector_score_min_slider", 0.50),
        "query_mode_value": st.session_state.get("query_mode_selector", "Single Query"),
        "add_to_zotero_state_value": st.session_state.get("add_to_zotero_state", True),
        "zotero_api_key_value": st.session_state.get("zotero_api_key", ""),
        "ai_tag_post_filter_values": st.session_state.get("ai_tag_post_filter_values", []),
        "enable_speedup_value": st.session_state.get("enable_speedup_checkbox", False),
        "speedup_threshold_value": st.session_state.get("speedup_threshold_slider", 9)
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

# --- RESTORE MODEL SELECTIONS FROM PREFS ---
if "model_key_selector" not in st.session_state:
    st.session_state.model_key_selector = prefs.get("selected_model_key", "models/gemini-2.0-flash-lite-preview-02-05")
if "ollama_local_model_selector" not in st.session_state:
    st.session_state.ollama_local_model_selector = prefs.get("selected_ollama_model", "")
# -------------------------------------------

if "automated_queries" not in st.session_state:
    st.session_state.automated_queries = []

# MIGRATION: Normalize Queries to include new fields (Defaults set here)
# If legacy query objects exist, update them with default start/stop/vector
migrated_queries = []
for q in st.session_state.automated_queries:
    if isinstance(q, dict):
        if "vector_min" not in q: q["vector_min"] = 0.50
        if "start_rec" not in q: q["start_rec"] = 0
        if "stop_rec" not in q: q["stop_rec"] = 12 # Default 12
        migrated_queries.append(q)
    elif isinstance(q, str):
        migrated_queries.append({
            "query": q, "folder": "", 
            "vector_min": 0.50, "start_rec": 0, "stop_rec": 12 # Default 12
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
    """
    Queries local Ollama chat API with STREAMING enabled.
    1. Prints output to Console in Real-Time.
    2. Implements a HARD TIMEOUT loop (abort if generation takes > 180s).
    """
    url = f"{OLLAMA_BASE_URL}/chat"
    
    # 1. Enable Streaming
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True, 
        "options": {"temperature": temperature}
    }
    
    print(f"\n{'='*20} [OLLAMA STREAM START] {'='*20}")
    
    # HARD TIMEOUT SETTING (Seconds)
    # If a single paper takes longer than this, we kill the connection.
    MAX_GEN_TIME = 180 
    start_time = time()
    full_response = ""

    try:
        # Connect to Ollama (10s connect timeout)
        with requests.post(url, json=payload, stream=True, timeout=10) as r:
            r.raise_for_status()
            
            # Iterate over the stream line by line
            for line in r.iter_lines():
                # CHECK FOR TIMEOUT EVERY TOKEN
                if (time() - start_time) > MAX_GEN_TIME:
                    print(f"\n\n🚨 TIMEOUT: Generation exceeded {MAX_GEN_TIME}s. Aborting this record to free GPU.")
                    return "ERROR: OLLAMA TIMEOUT"

                if line:
                    try:
                        # Parse the JSON chunk
                        body = json.loads(line)
                        content = body.get('message', {}).get('content', '')
                        
                        # Print to Console immediately (Flush ensures no buffer delay)
                        print(content, end="", flush=True)
                        
                        # Accumulate
                        full_response += content
                        
                        if body.get('done', False):
                            break
                    except json.JSONDecodeError:
                        pass
                        
    except Exception as e:
        print(f"\n\n❌ OLLAMA STREAM ERROR: {e}")
        return f"OLLAMA CONNECTION ERROR: {e}"
        
    print(f"\n{'='*20} [STREAM END] {'='*20}\n")
    return full_response

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
    
    # --- LOGGING: Full Prompt Sent ---
    provider_label = "OLLAMA" if use_ollama else "GEMINI"
    print(f"\n{'='*40}\n[{provider_label} BOOLEAN PROMPT SENT]\n{prompt}\n{'='*40}\n")
    
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
            
            # --- LOGGING: Raw Response Received ---
            print(f"\n{'='*40}\n[GEMINI BOOLEAN RESPONSE]\n{resp.text}\n{'='*40}\n")
            
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
    Handles PMC HTML pages by finding the actual PDF link.
    Includes robust retry logic (4 attempts, 7s delay) for stubborn servers (Springer, etc.).
    """
    if not url: return ""
    
    # --- USER FEEDBACK: EXTRACTION START ---
    logging.info(f"📄 Extracting text from PDF: {url}... one moment.")
    
    try:
        # Improved headers to look more like a browser during retries
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com/"
        }
        
        # 1. HEAD request to check file size first (Save bandwidth)
        try:
            head_resp = requests.head(url, headers=headers, timeout=5)
            file_size = int(head_resp.headers.get('Content-Length', 0))
            if file_size > 15 * 1024 * 1024: # Skip if > 15MB
                logging.warning(f"PDF too large ({file_size} bytes), skipping: {url}")
                return ""
        except:
            pass

        # 2. GET request with RETRY LOGIC
        # User requested: Reload 3-4 times, waiting 7 seconds each time.
        r = None
        max_retries = 4
        retry_delay = 7
        
        for attempt in range(1, max_retries + 1):
            try:
                # 20s timeout for the actual download
                r = requests.get(url, headers=headers, timeout=20) 
                
                if r.status_code == 200:
                    # Basic check to ensure we didn't just get a tiny error page
                    if len(r.content) > 500:
                        break # Success
                    else:
                        logging.warning(f"PDF Attempt {attempt}: Content too small ({len(r.content)} bytes). Retrying in {retry_delay}s...")
                else:
                    logging.warning(f"PDF Attempt {attempt}: Status {r.status_code}. Retrying in {retry_delay}s...")
            
            except Exception as e:
                logging.warning(f"PDF Attempt {attempt} Error: {e}. Retrying in {retry_delay}s...")
            
            # Wait before next retry, unless it's the last attempt
            if attempt < max_retries:
                sleep(retry_delay)

        if not r or r.status_code != 200:
            logging.error(f"Failed to download PDF after {max_retries} attempts.")
            return ""
        
        # 3. Check content type
        content_type = r.headers.get("Content-Type", "").lower()
        
        # --- PMC / HTML HANDLING ---
        if "application/pdf" not in content_type and "html" in content_type:
            # Check if it's a PMC article page
            if "ncbi.nlm.nih.gov/pmc/articles/PMC" in url or "pmc" in url.lower():
                logging.info("Detected PMC HTML page. Searching for PDF link...")
                soup = BeautifulSoup(r.content, 'html.parser')
                
                # Strategy: Look for links with .pdf extension or specific classes
                pdf_link = soup.find('a', href=re.compile(r'\.pdf$', re.I))
                
                if pdf_link:
                    # Construct absolute URL
                    new_pdf_url = urljoin(url, pdf_link['href'])
                    logging.info(f"🔗 Found actual PDF link: {new_pdf_url}")
                    
                    # Recursive call with the new PDF URL
                    return extract_pdf_text(new_pdf_url)
                else:
                    logging.warning("Could not find PDF link on PMC page.")
                    return ""

        # 4. Extract Text using PDFMiner (if it is a PDF)
        if "application/pdf" in content_type or r.content.startswith(b"%PDF"):
            with io.BytesIO(r.content) as pdf_stream:
                text = extract_text(pdf_stream, maxpages=24)
                return text
        else:
            logging.warning(f"Download successful but Content-Type was '{content_type}', not PDF.")
            return ""

    except Exception as e:
        logging.error(f"PDFMiner Extract Error: {e}")
        return ""

def extract_webpage_text(url):
    """
    Smart Harvester (Cluster/Density Update):
    1. Detects Semantic Scholar URLs and finds 'External Link' button.
    2. Resolves Publisher URLs.
    3. Checks for <meta name="citation_pdf_url"> (Hidden PDF).
    4. Falls back to "Cluster Density" Paragraph Hunting.
       - Identifies the specific HTML container (div/article) with the most valid text.
       - Ignores sidebars, footers, and reference lists outside that container.
    """
    if not url or url.endswith('.pdf'): return ""
    
    # --- USER FEEDBACK: EXTRACTION START ---
    logging.info(f"🕵️ Smart Harvester investigating: {url}...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        # STEP 1: S2 Button "Click" Logic (Keep existing logic)
        if "semanticscholar.org" in url:
            logging.info("Detected Semantic Scholar Page. Searching for External Link Button...")
            s2_resp = requests.get(url, headers=headers, timeout=10)
            if s2_resp.status_code == 200:
                s2_soup = BeautifulSoup(s2_resp.content, 'html.parser')
                ext_link = s2_soup.find("a", attrs={"data-heap-id": "paper-link"})
                if not ext_link:
                    ext_link = s2_soup.find("a", class_=lambda x: x and "icon-button" in x and "button--primary" in x)
                
                if ext_link and ext_link.get('href'):
                    new_target = ext_link['href']
                    logging.info(f"Found External Publisher Link on S2: {new_target}")
                    return extract_webpage_text(new_target) # RECURSE
        
        # STEP 2: Request the Publisher Landing Page
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        if resp.status_code != 200: return ""
        
        # STEP 3: Check for "Hidden" PDF via Metadata
        soup = BeautifulSoup(resp.content, 'html.parser')
        pdf_meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
        if pdf_meta and pdf_meta.get("content"):
            pdf_target = pdf_meta["content"]
            logging.info(f"🎯 Found hidden PDF meta-tag: {pdf_target}")
            extracted_pdf_text = extract_pdf_text(pdf_target)
            if extracted_pdf_text and len(extracted_pdf_text) > 200:
                return extracted_pdf_text

        # STEP 4: Cluster Density Hunting (New Strategy)
        logging.info("No direct PDF found. Running Cluster Density analysis...")
        
        # A. Remove structural clutter tags
        for element in soup(["script", "style", "nav", "footer", "header", "meta", "noscript", "aside", "form", "button", "input", "iframe"]):
            element.decompose()
        
        # B. Define Keywords for Garbage Filtering
        garbage_triggers = [
            "copyright", "all rights reserved", "download citation", "view author", 
            "search author", "log in", "sign up", "et al.", "vol.", "pp.", "doi:", 
            "google scholar", "pubmed", "crossref", "cited by", "similar articles"
        ]

        # C. Initialize Scoring
        # We will map {Parent_Node: Score}
        parent_scores = {}
        
        # Find ALL paragraphs in the document
        all_paragraphs = soup.find_all('p')
        
        for p in all_paragraphs:
            text = p.get_text(" ", strip=True)
            
            # --- FILTER LEVEL 1: Basic Hygiene ---
            # Must be > 30 words and start with Uppercase
            if len(text.split()) < 30: continue
            if not text[0].isupper(): continue
            if any(trigger in text.lower() for trigger in garbage_triggers): continue

            # --- FILTER LEVEL 2: The "Three Sentence" Rule ---
            # Split by sentence terminators followed by a Capital letter
            sentences = re.split(r'[.!?]\s+(?=[A-Z])', text)
            
            # Must have at least 3 sentences
            if len(sentences) < 3: continue
            
            # Count sentences with at least 4 words
            valid_sentences_count = 0
            for s in sentences:
                if len(s.split()) >= 4:
                    valid_sentences_count += 1
            
            # STRICT REQUIREMENT: At least 3 robust sentences
            if valid_sentences_count < 3: continue

            # --- SCORING ---
            # If it passes, add the length of the text to the parent's score.
            # This creates a "gravity well" around the main article div.
            parent = p.parent
            # Move up one level if the parent is just a generic wrapper (optional optimization)
            if parent.name in ['span', 'strong', 'em', 'a']:
                parent = parent.parent
                
            current_score = parent_scores.get(parent, 0)
            parent_scores[parent] = current_score + len(text)

        # D. Pick the Winner
        if not parent_scores:
            logging.warning("Cluster Analysis: No valid text clusters found.")
            return ""

        # Find the parent element with the highest score
        best_parent = max(parent_scores, key=parent_scores.get)
        logging.info(f"Cluster Analysis: Winner found (<{best_parent.name}>) with score {parent_scores[best_parent]}")

        # E. Extract Text ONLY from the Winner
        # We re-iterate through the winner's paragraphs to ensure order
        final_blocks = []
        for p in best_parent.find_all('p'):
            t = p.get_text(" ", strip=True)
            # Re-apply basic length filter to avoid caption snippets inside the main div
            if len(t.split()) > 15: 
                final_blocks.append(t)
        
        clean_text = "\n\n".join(final_blocks)
        return clean_text[:40000]
            
    except Exception as e:
        logging.warning(f"Smart Harvester Failed for {url}: {e}")
    
    return ""

def check_zotero_duplicate(doi: str, library_id: str, collection_id: str) -> bool:
    """Checks Zotero Cloud API for existing DOI."""
    api_key = st.session_state.get("zotero_api_key", "").strip()
    if not doi or not library_id or not api_key:
        return False
    
    url = f"https://api.zotero.org/users/{library_id}/items"
    headers = {"Zotero-API-Key": api_key}
    # Check globally or in specific collection? Usually global check is safer for "Duplicate"
    # But if user wants to know if it is in THIS collection, we should add collection to params.
    # For now, default behavior is checking library-wide to prevent multi-saving.
    params = {"q": doi, "itemType": "journalArticle", "limit": 1}
    
    if collection_id:
        params['collection'] = collection_id
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        if resp.status_code == 200:
            return len(resp.json()) > 0
    except Exception:
        pass
    return False

def upload_to_zotero(payload_dict: Dict[str, Any], target_coll_id: str = None) -> tuple[bool, str]:
    """
    Posts record to local Zotero Connector.
    Includes a 10-minute retry loop to handle Zotero syncing/busy states.
    """
    connector_endpoint = "http://127.0.0.1:23119/connector/saveItems"
    final_payload = {"items": [payload_dict]}
    
    if target_coll_id:
					 
        final_payload["collectionId"] = target_coll_id 

    # 10 Minute Budget (600 seconds)
    start_time = time()
    MAX_WAIT = 600
    
    while (time() - start_time) < MAX_WAIT:
        try:
            # Brief pause to let Zotero breathe between requests
            sleep(0.5)
            
            z_resp = requests.post(connector_endpoint, json=final_payload, timeout=60)
            z_resp.raise_for_status()
            return True, "Item sent to Zotero."
            
																	  
        except Exception as e:
            # If Zotero is busy (common during sync), wait and retry
            # Logging this would show up in console, but we keep it silent to UI until timeout
            sleep(5) # Wait 5 seconds before retrying
            last_error = str(e)

    return False, f"Zotero Timeout (10m): {last_error}"

def delete_zotero_item(doi):
    """Deletes an item from Zotero based on DOI search."""
    library_id = st.session_state.get("user_zotero_id")
    api_key = st.session_state.get("zotero_api_key")
    
    if not library_id or not api_key:
        return False, "Missing Library ID or API Key in settings."
    
    # We must use the API because the Local Connector (127.0.0.1) does not permit deletion.
    base_url = f"https://api.zotero.org/users/{library_id}/items"
    headers = {"Zotero-API-Key": api_key}
    
    try:
        # 1. Search for the Item Key
        search_params = {"q": doi, "itemType": "journalArticle", "limit": 1}
        search_resp = requests.get(base_url, headers=headers, params=search_params, timeout=10)
        search_data = search_resp.json()
        
        if not search_data:
            return False, "Item not found in Zotero."
            
        item_key = search_data[0]['key']
        version = search_data[0]['version']
        
        # 2. Send Delete Request
        headers["If-Match"] = str(version)
        del_resp = requests.delete(f"{base_url}/{item_key}", headers=headers, timeout=10)
        
        if del_resp.status_code == 204:
            return True, "Item deleted."
        else:
            return False, f"Delete failed: {del_resp.status_code} {del_resp.text}"
            
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
    
    # --- LOGGING: Full Prompt Sent ---
    provider_label = "OLLAMA" if use_ollama else "GEMINI"
    print(f"\n{'='*40}\n[{provider_label} EXTRACT PROMPT SENT]\n{prompt}\n{'='*40}\n")
    
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
            
            # --- LOGGING: Raw Response Received ---
            print(f"\n{'='*40}\n[GEMINI EXTRACT RESPONSE]\n{resp.text}\n{'='*40}\n")
            
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

def generate_semantic_tags(abstract_text: str, semantic_sentences: List[tuple], semantic_model, top_n: int = 1):
    """
    Generates tags based on semantic similarity.
    Returns a list of tuples: (TagString, OriginalSentenceString)
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
            match_sentence = enabled_entries[best_idx][0]
            
            clean_tag = re.sub(r'[^a-zA-Z0-9_\-]+', '', custom_tag)
            # RETURN TUPLE: (Tag, Sentence)
            return [(f"sTag-{clean_tag}", match_sentence)]
            
        return []
    except Exception as e:
        logging.error(f"Semantic Tag Error: {e}")
        return []

def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def remove_think_tags(text):
    """
    Removes content within <think>...</think> tags to avoid parsing the reasoning process.
    Handles multiple think blocks and case sensitivity.
    Also handles stray </think> tags if the opening tag is missing.
    """
    if not text: return ""
    
    # 1. Remove complete blocks
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 2. Handle stray closing tags (common in streaming/interrupted outputs)
    # If we see </think> but no <think>, remove everything before it.
    if '</think>' in clean_text:
        parts = clean_text.split('</think>')
        # CRITICAL FIX: Only take the last part if it's not empty.
        if parts[-1].strip():
            clean_text = parts[-1]
        else:
            if len(parts) > 1:
                clean_text = parts[0]
        
    return clean_text.strip()

                                                                 
def gemini_annotate_paper(title, authors_info, snippet, pdf_text, url, user_query, model, suggested_tags=None, priority_topics=None):
    use_ollama = st.session_state.get("use_ollama", False)
    
    # --- PROMPT ---
    tags_instruction = ""
    if suggested_tags and len(suggested_tags) > 0:
        tags_instruction = f"OFFICIAL TAG LIST: {', '.join(suggested_tags)}\nSelect relevant tags from this list."
    else:
        tags_instruction = "Generate 3-5 relevant tags."

    context_text = pdf_text[:40000] if pdf_text else 'N/A'

    prompt = f"""
    Analyze this scientific paper for a structured report.
    
    METADATA:
    Title: {title}
    Abstract: {snippet}
    Full Text Segment: {context_text}
    User Query: {user_query}
    Priority Interests (FOR EXTRACTION ONLY): {priority_topics}

    INSTRUCTIONS:
    1. ANALYZE RELEVANCE (0-3).
       - Score based strictly on the alignment between the 'Priority Interests' list and the paper content.
       - NEGATIVE CONSTRAINT: Do NOT consider the  'User Query' or 'Priority Interests' as verbiage to score. Only use the 'Title' and 'Abstract' to calculate the relevance score.
    2. EXTRACT TAGS (Comma separated). {tags_instruction}
    3. SUMMARIZE findings (Bullet points).
    4. EXTRACT SUBSTANCES (Chemicals and Plants):
       - Use 'Priority Interests' as a reference for what to look for.
       - CRITICAL: Only list substances EXPLICITLY mentioned in the paper text provided above.
       - DO NOT hallucinate plants or chemicals based on the 'Priority Interests' list.
       - If the paper does not mention specific plants or chemicals, strictly write "None".

    OUTPUT FORMAT (STRICTLY USE THESE HEADERS):
    ###SCORE###
    <0, 1, 2, or 3>
    ###TAGS###
    <tag1, tag2, tag3>
    ###SUMMARY###
    <Bullet points of key findings>
    ###CHEMICALS###
    <List of pure chemicals found in the text, or "None">
    ###PLANTS###
    <List of whole plants/herbs found in the text, or "None">
    """
    
    # --- LOGGING: Full Prompt Sent ---
    provider_label = "OLLAMA" if use_ollama else "GEMINI"
    print(f"\n{'='*40}\n[{provider_label} PROMPT SENT]\n{prompt}\n{'='*40}\n")
    
    raw_text = ""
    
    # --- API CALL ---
    try:
        if use_ollama:
            # DIRECT CALL - NO CHECKS - NO ROADBLOCKS
            raw_text = query_ollama_chat(model, prompt)
            print(f"\n{'='*40}\n[OLLAMA RESPONSE]\n{raw_text}\n{'='*40}\n")
        elif client:
            for attempt in range(3):
                try:
                    sleep(GEMINI_DELAY)
                    resp = client.models.generate_content(model=model, contents=prompt)
                    raw_text = resp.text
                    
                    # --- LOGGING: Raw Response Received ---
                    print(f"\n{'='*40}\n[GEMINI RESPONSE]\n{raw_text}\n{'='*40}\n")
                    
                    break
                except Exception as e:
                    if attempt < 2 and handle_gemini_backoff(str(e)): continue
                    # Error return structure updated to tuple(5)
                    return f"API FAILURE: {e}", [], 0, "None", "None"
        else:
             # Error return structure updated to tuple(5)
            return "API KEY MISSING", [], 0, "None", "None"

    except Exception as e:
         # Error return structure updated to tuple(5)
        return f"EXECUTION ERROR: {e}", [], 0, "None", "None"

    # --- PARSING ---
    text = remove_think_tags(raw_text)

    # Regex using the ### headers from the prompt
    score_matches = re.findall(r"(?:\*\*|)?###\s*SCORE\s*###(?:\*\*|)?\s*(\d+)", text, re.IGNORECASE)
    score = 0
    if score_matches:
        try:
            score = int(score_matches[-1])
            if score > 3: score = 3
        except: pass

    tags_matches = re.findall(r"(?:\*\*|)?###\s*TAGS\s*###(?:\*\*|)?\s*(.*?)(?=\n(?:\*\*|)?###|$)", text, re.DOTALL | re.IGNORECASE)
    tags = []
    if tags_matches:
        raw_tags = tags_matches[-1].strip()
        tags = [t.strip() for t in raw_tags.split(',') if t.strip()]

    summary_matches = re.findall(r"(?:\*\*|)?###\s*SUMMARY\s*###(?:\*\*|)?\s*(.*?)(?=\n(?:\*\*|)?###|$)", text, re.DOTALL | re.IGNORECASE)
    summary_txt = summary_matches[-1].strip() if summary_matches else "Analysis unavailable."

    chems_matches = re.findall(r"(?:\*\*|)?###\s*CHEMICALS\s*###(?:\*\*|)?\s*(.*?)(?=\n(?:\*\*|)?###|$)", text, re.DOTALL | re.IGNORECASE)
    chemicals = chems_matches[-1].strip() if chems_matches else "None listed"

    plants_matches = re.findall(r"(?:\*\*|)?###\s*PLANTS\s*###(?:\*\*|)?\s*(.*?)(?=\n(?:\*\*|)?###|$)", text, re.DOTALL | re.IGNORECASE)
    plants = plants_matches[-1].strip() if plants_matches else "None listed"

    # --- BUILD CLEAN REPORT ---
    # UPDATED: Uses Unicode escape \U0001F9EA for the test tube emoji to prevent SyntaxError
    substances_display = (
        "\n---\n"
        "**\U0001F9EA Substances & Plants**\n"
        f"*   **Chemicals:** {chemicals}\n"
        f"*   **Plants:** {plants}\n"
    )

    clean_report = f"""
**AI Abstract Summary**
{summary_txt}
{substances_display}
"""
    # RETURN TUPLE (5): report, tags, score, chemicals, plants                                                                          
    return clean_report, tags, score, chemicals, plants

def gemini_abstract_fallback(title, authors_info, current_snippet, model, full_text=None):
    """
    Generates an abstract.
    PRIORITY 1: Summarize the PDF Full Text (if available).
    PRIORITY 2: Hallucinate based on Title/Authors (only if no text available).
    """
    use_ollama = st.session_state.get("use_ollama", False)
    
    # LOGIC CHANGE: Use full text if available
    if full_text and len(full_text) > 500:
        prompt = f"""
        Summarize the following academic text into a concise abstract (approx 150 words).
        Focus on the objectives, methods, and results.
        
        TEXT SOURCE:
        {full_text[:15000]}
        """
    else:
        # Fallback to the "Guess" if we couldn't get the PDF
        prompt = f"""
        Generate a 3-sentence abstract for this paper based on metadata.
        Title: {title}
        Authors: {authors_info}
        Output ONLY text.
        """
    
    # --- LOGGING: Full Prompt Sent ---
    provider_label = "OLLAMA" if use_ollama else "GEMINI"
    # User requested display for troubleshooting:
    print(f"\n{'='*40}\n[ABSTRACT NOT FOUND - {provider_label} FALLBACK PROMPT SENT]\n{prompt}\n{'='*40}\n")
    
    # --- OLLAMA PATH ---
    if use_ollama:
        try:
            resp = query_ollama_chat(model, prompt)
            # --- LOGGING: Full Response Received ---
            print(f"\n{'='*40}\n[ABSTRACT NOT FOUND - {provider_label} FALLBACK RESPONSE]\n{resp}\n{'='*40}\n")
            return resp
        except Exception as e:
            print(f"OLLAMA FALLBACK ERROR: {e}")
            return current_snippet

    # --- GEMINI PATH ---
    if client:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                sleep(GEMINI_DELAY) # Rate limit
                resp = client.models.generate_content(model=model, contents=prompt)
                resp_text = resp.text.strip()
                
                # --- LOGGING: Raw Response Received ---
                print(f"\n{'='*40}\n[ABSTRACT NOT FOUND - GEMINI FALLBACK RESPONSE]\n{resp.text}\n{'='*40}\n")
                
                return resp_text
            except Exception as e:
                if attempt < max_retries - 1 and handle_gemini_backoff(str(e)):
                    logging.info(f"Retrying Abstract Fallback (Attempt {attempt+2}/{max_retries})...")
                    continue
                return current_snippet
       
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
# UI Helper functions 
# ============================
# (UI Helpers block - Updated for Custom Tags)
def add_new_sentence():
    ns = st.session_state.new_sentence_input.strip()
    if ns:
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
# SEARCH PROVIDERS (UPDATED FOR OFFSET)
# ============================

def search_semantic_scholar(query: str, limit: int, start_offset: int = 0) -> Iterator[List[Dict[str, Any]]]:
    if not query or not query.strip():
        logging.warning("Skipping Semantic Scholar search: Query is empty.")
        return 
    
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
    
    # OFFSET LOGIC: Stop when we reach start+limit
    offset = start_offset
    target_stop = start_offset + limit
    
    st.info(f"🔎 Starting Search for: '{query}' (Records {start_offset} to {target_stop})")
    
    # Initialize duplicate tracker
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
                if offset == start_offset: st.warning("Semantic Scholar returned 0 results.")
                break
            
            raw_papers = data.get("data", [])
            
            page_recs = []
            for paper in raw_papers:
                doi = paper.get("externalIds", {}).get("DOI")
                if doi in seen_dois and doi is not None: continue
                if doi: seen_dois.add(doi)
                
                # Title Fix: Ensure it is never None
                paper_title = paper.get("title")
                if not paper_title:
                    paper_title = "Untitled Paper"

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
                    "original_abstract": abstract_text # <-- SAVES ORIGINAL
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
            "original_abstract": abstract_text # <-- SAVES ORIGINAL
        }
    except Exception:
        return None

def search_pubmed_paged(query, limit=10, start_offset=0):
    """
    Unchained PubMed search - No artificial delays.
    Includes simple caching and batching for stability.
    """
    cache_key = f"PUBMED_{query}_{limit}_{start_offset}"
    if cache_key in st.session_state.search_cache:
        return st.session_state.search_cache[cache_key]

    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    term = (query or "").strip() 
    
    # 1. ESearch (Get all IDs first)
    all_pmids = []
    # Start at the user-specified offset
    retstart = start_offset 
    
    while len(all_pmids) < limit:
        retmax = min(1000, limit - len(all_pmids))
        params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax, "retstart": retstart, "email": NCBI_EMAIL, "api_key": NCBI_API_KEY}
        try:
            # POST is safer for very long queries
            r = requests.post(f"{base}/esearch.fcgi", data=params, timeout=30).json()
            ids = r.get("esearchresult", {}).get("idlist", [])
            if not ids: break
            all_pmids.extend(ids)
            retstart += 1000
        except Exception:
            break
            
    ids = all_pmids[:limit]
    if not ids: return []

    # 2. ESummary (Batch processing to avoid URI too long or timeouts)
    # We fetch summaries in chunks of 200 (safe limit)
    all_summaries = {}
    for chunk in _chunks(ids, 200):
        sum_params = {"db": "pubmed", "id": ",".join(chunk), "retmode": "json", "email": NCBI_EMAIL, "api_key": NCBI_API_KEY}
        try:
            # POST is required here because the ID list can be long
            resp = requests.post(f"{base}/esummary.fcgi", data=sum_params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                result_block = data.get("result", {})
                # Remove 'uids' list from result block to iterate papers
                if "uids" in result_block: del result_block["uids"]
                all_summaries.update(result_block)
        except:
            pass

    # 3. EFetch (XML for abstracts) - Also batched
    abstracts = {}
    for chunk in _chunks(ids, 200):
        try:
            ef_params = {"db": "pubmed", "retmode": "xml", "email": NCBI_EMAIL, "api_key": NCBI_API_KEY}
            ef = requests.post(f"{base}/efetch.fcgi", data={"id": ",".join(chunk)}, params=ef_params, timeout=60)
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
            "original_abstract": abstract_text # <-- SAVES ORIGINAL
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
            
            extracted_snippet = text[:500]
            return [{
                "title": "Extracted PDF",
                "url": url_or_doi,
                "snippet": extracted_snippet,
                "pdf_text": text,
                "snippet_source": "PDF_EXTRACT",
                "original_abstract": extracted_snippet # <-- ADDED FOR CONSISTENCY
            }]
    return []

# ============================
# DISPLAY & PROCESSING (UPDATED REPORT & AUDIT)
# ============================

def _display_paper_details(paper_data, idx_key):
    """Render a single paper details card matching the provided screenshot design."""
    paper = paper_data.get('paper', {})
    score = paper_data.get('score', 0)
    tags = paper_data.get('tags', [])
    ai_report_body = paper_data.get('abstract_ai', "")
    
    title = paper.get("title") or "Untitled Paper"
    url = paper.get("url", "#")
    doi = paper.get("doi", "N/A")
    
    status_msg = paper_data.get('status_msg', "") 
    reason = paper_data.get('reason', "")
    
    # Icon and Expand State
    icon = "✅" if status_msg == "SAVED" else ("🚫" if status_msg == "DUPLICATE" else "⚠️")
    
    # Construct Header Label (Status + Score + Title + Reason)
    header_label = f"{icon} [Score: {score}/3] {title}"
    if status_msg != "SAVED" and status_msg != "DUPLICATE":
        header_label += f" | {reason}"
    elif status_msg == "SAVED":
        header_label += " | SAVED"

    # INJECT CSS FOR EXPANDER HEADER (Single line string to prevent SyntaxError)
    st.markdown(
        "<style>.streamlit-expanderHeader {font-size: 1.2rem !important; font-weight: bold !important;}</style>",
        unsafe_allow_html=True
    )
    
    with st.expander(header_label, expanded=(status_msg == "SAVED")):
        
        # 1. CITATION DATA
        st.markdown("### 1. Citation Data")
        st.markdown(f"**Authors:** {paper.get('authors_info', 'N/A')}")
        st.markdown(f"**DOI:** [{doi}](https://doi.org/{doi})")
        if url: st.markdown(f"**Link:** [Open Source]({url})")
        
        # 2. ABSTRACTS (Split View: Source Left, AI Right)
        st.markdown("### 2. Abstracts")
        
        col_src, col_ai = st.columns(2)
        
        with col_src:
            # ORIGINAL ABSTRACT ONLY (Unmodified)
            raw_abstract = paper.get('original_abstract', paper.get('snippet', 'No abstract available.'))
            
            st.caption("Source Abstract")
            st.info(raw_abstract) 
        
        with col_ai:
            st.caption("AI Abstract Summary")
            if ai_report_body and "Duplicate" not in ai_report_body:
                st.markdown(ai_report_body)
            else:
                st.caption("Analysis unavailable.")
        
        # 3. CATEGORIZATION
        st.markdown("### 3. Categorization")
        
        # --- FIX: ROBUST TAG HANDLING ---
        # Semantic search adds tuples: (Tag, Sentence). We must extract just the Tag.
        safe_tags = []
        if tags:
            for t in tags:
                if isinstance(t, tuple) and len(t) > 0:
                    safe_tags.append(str(t[0])) # Extract tag name from tuple
                elif t:
                    safe_tags.append(str(t)) # Handle normal strings
        
        st.markdown(f"**Tags:** {', '.join(safe_tags) if safe_tags else 'None detected'}")
        # -------------------------------
        
        # ACTIONS
        st.markdown("---")
        if st.button("📤 Upload to Zotero", key=f"btn_save_{idx_key}"):
            # Sanitize tags: Ensure string type and truncate to limit
            clean_tags = []
            for t in tags:
                if isinstance(t, str):
                    clean_tags.append({"tag": t[:MAX_TAG_LENGTH]})
                elif isinstance(t, tuple) and len(t) > 0:
                    # Handle accidental tuple tags from semantic search
                    clean_tags.append({"tag": str(t[0])[:MAX_TAG_LENGTH]})
            
            item = {
                "itemType": "journalArticle",
                "title": paper.get("title"),
                "creators": parse_authors(paper.get("authors_info")),
                "abstractNote": (paper.get("original_abstract") or paper.get("snippet"))[:ZOTERO_MAX_ABSTRACT_CHARS],
                "date": str(paper.get("year") or ""),
                "tags": clean_tags,
                "url": with_ntu_proxy(doi or paper.get("url")),
                "DOI": doi
            }
            current_coll = st.session_state.user_zotero_collection
            ok, msg = upload_to_zotero(item, target_coll_id=current_coll)
            if ok: st.toast(f"✅ Saved to Zotero: {title}")
            else: st.error(f"Zotero Error: {msg}")

def process_chunk_and_save(chunk, query, total_papers, papers_processed, status_ph, prog_ph, query_stats, collection_override=None, vec_min_override=None):
    annotated = 0
    saved = 0
    
    # Determine which collection to use
    target_collection = collection_override if collection_override else st.session_state.user_zotero_collection

    # UI Settings
    min_len = st.session_state.abstract_length_slider
    z_thresh = st.session_state.min_score3_slider
    
    # --- LOGIC UPDATE: USE ROW SETTING IF AVAILABLE, ELSE GLOBAL SLIDER ---
    if vec_min_override is not None:
        vec_min = float(vec_min_override)
    else:
        vec_min = st.session_state.vector_score_min_slider
    # ----------------------------------------------------------------------
    
    # Filter Config
    req_vals = [v.lower() for v in st.session_state.ai_tag_post_filter_values]
    suggested_tags = st.session_state.selected_tag_categories
    
    # Speed Up Settings
    enable_speedup = st.session_state.enable_speedup_checkbox
    speedup_threshold = st.session_state.speedup_threshold_slider
    
    # AI Model Resolution
    if st.session_state.get("use_ollama"):
        actual_model_id = st.session_state.get("ollama_local_model_selector", "llama3")
    else:
        selected_key = st.session_state.model_key_selector
        actual_model_id = MODEL_OPTIONS[selected_key]["model_id"]
    
    # Context
    topics_context = st.session_state.get("topics_txt", "")

    # 1. Preranking & Vector Filter
    sem_model = get_embedding_model()
    if sem_model:
        chunk = prerank_papers(chunk, query, sem_model)
        # Use the variable 'vec_min' determined above
        if vec_min > 0.0:
            original_count = len(chunk)
            chunk = [p for p in chunk if p.get('vector_score', 0) >= vec_min]
            dropped = original_count - len(chunk)
            if dropped > 0:
                logging.info(f"Vector Filter: Dropped {dropped} papers (Score < {vec_min})")
    
    if not chunk: return 0, 0

    for i, paper in enumerate(chunk):
        start_time = time()
        idx = papers_processed + i
        bypass_ai = query_stats.get('bypass_ai', False)
        
        # --- LOGGING UPDATE: Display Raw Source Data if Abstract Missing ---
        initial_snippet = paper.get("snippet", "")
        print(f"\n{'='*40}\n[PROCESSING RECORD {idx+1}]\nTITLE: {paper.get('title')}")
        
        if not initial_snippet:
            # If missing, show the user exactly what the API gave us (excluding huge fields)
            debug_paper = {k: v for k, v in paper.items() if k != 'original_abstract'}
            print(f"SOURCE VERBIAGE: ((NO ABSTRACT FOUND))")
            print(f"RAW METADATA DUMP: {debug_paper}")
        else:
            print(f"SOURCE VERBIAGE (First 500 chars):\n{initial_snippet[:500]}...")
        print(f"{'='*40}\n")
        # ------------------------------------------------

        # 2. Duplicate Check
        doi = paper.get("doi")
        is_dup = False
        if doi and (not st.session_state.allow_duplicates) and check_zotero_duplicate(doi, st.session_state.user_zotero_id, target_collection):
             is_dup = True

        # ==========================================================
        # 3. TEXT EXTRACTION HIERARCHY (Updated with Smart Harvester)
        # ==========================================================
        full_text_context = ""
        
        # LOGIC INJECTION: Check if Abstract exists in metadata
        current_snippet = paper.get("snippet", "")
        
        if current_snippet and len(current_snippet) >= min_len:
             logging.info(f"✅ Abstract Sourced from Metadata (Skipping PDF/Web): {paper.get('title')[:30]}...")
             paper["snippet_source"] = "METADATA"
        
        elif not bypass_ai:
            logging.info(f"📉 Abstract Missing/Short: Initiating Smart Harvest for {paper.get('title')[:30]}...")
            
            # --- THE FIX: DETERMINE TARGET URL FOR "CLICK" ---
            target_url = ""
            # Priority 1: Use DOI to "click" through to publisher (Standard)
            if paper.get("doi"):
                 target_url = f"https://doi.org/{paper.get('doi')}"
            # Priority 2: Use the provided URL (S2 page or direct link)
            elif paper.get("url"):
                 target_url = paper.get("url")
            
            if target_url:
                # Trigger the Smart Harvester (extract_webpage_text now handles the detective work)
                full_text_context = extract_webpage_text(target_url)

                # --- UPDATE UI IF API FAILED BUT SCRAPER WORKED ---
                if not paper.get("original_abstract") and full_text_context:
                    # Limit to 3000 chars for display so it doesn't flood the UI
                    paper["original_abstract"] = full_text_context[:3000] + ("..." if len(full_text_context) > 3000 else "")

        # 4. Abstract Fallback
        # If the original snippet is too short, we ask AI to generate one using the full text we just found.
        snip = paper.get("snippet", "")
        if len(snip) < min_len:
            if not bypass_ai:
                # NOTE: logging of the "Complete Verb" happens inside this function
                snip = gemini_abstract_fallback(paper.get("title"), paper.get("authors_info"), snip, actual_model_id, full_text=full_text_context)
                paper["snippet"] = snip
                paper["snippet_source"] = "AI_FALLBACK"

        # 5. AI Annotation (Updated Unpacking for Chemicals/Plants)
        if is_dup:
            ai_abs, tags, score, chemicals, plants = "Duplicate - Skipped Analysis", [], 0, "None", "None"
        elif bypass_ai:
            ai_abs, tags, score, chemicals, plants = "Auto-Saved (Speed Mode)", ["Speed-Save"], 3, "Skipped", "Skipped"
        else:
            ai_abs, tags, score, chemicals, plants = gemini_annotate_paper(
                paper.get("title"), paper.get("authors_info"), snip, full_text_context, paper.get("url"), query, actual_model_id,
                suggested_tags=suggested_tags, priority_topics=topics_context
            )
            if "API FAILURE" not in ai_abs: annotated += 1

        # 6. Semantic Tags
        if sem_model and "API FAILURE" not in ai_abs and not is_dup:
            stags = generate_semantic_tags(ai_abs if not bypass_ai else snip, st.session_state.semantic_sentences, sem_model)
            tags.extend(stags)

        # 7. Save Logic & Filtering
        passes = True
        missing_filters = []
        if req_vals:
            search_text = (
                str(paper.get("title", "")) + " " + 
                str(paper.get("original_abstract", "")) + " " + 
                str(snip) + " " + 
                str(ai_abs) + " " + 
                " ".join([str(t) for t in tags]) # Ensure tags are strings
            ).lower()
            passes = any(v in search_text for v in req_vals)
            if not passes: missing_filters = [v for v in req_vals]

        status_msg = "SKIPPED"
        reason = ""
        
        # --- NEW LOGIC: SUBSTANCE FILTER ---
        substance_rejected = False
        if "none" in chemicals.lower() and "none" in plants.lower():
            substance_rejected = True
            
        if is_dup:
            status_msg = "DUPLICATE"
        elif st.session_state.add_to_zotero_state and score >= z_thresh and passes:
            if substance_rejected:
                status_msg = "REJECTED"
                reason = "No Substances/Plants Found (Filter)"
            else:
                # Sanitize tags: Ensure string type and truncate to limit
                clean_tags = []
                for t in tags:
                    if isinstance(t, str):
                        clean_tags.append({"tag": t[:MAX_TAG_LENGTH]})
                    elif isinstance(t, tuple) and len(t) > 0:
                        # Handle accidental tuple tags from semantic search
                        clean_tags.append({"tag": str(t[0])[:MAX_TAG_LENGTH]})
                
                item = {
                    "itemType": "journalArticle",
                    "title": paper.get("title"),
                    "creators": parse_authors(paper.get("authors_info")),
                    "abstractNote": (paper.get("original_abstract") or snip)[:ZOTERO_MAX_ABSTRACT_CHARS],
                    "date": str(paper.get("year") or ""),
                    "tags": clean_tags,
                    "url": with_ntu_proxy(doi or paper.get("url")),
                    "DOI": doi
                }
                ok, msg = upload_to_zotero(item, target_coll_id=target_collection)
                if ok: 
                    saved += 1
                    status_msg = "SAVED"
                else: 
                    st.error(f"Zotero Error: {msg}")
        else:
            if not passes: reason = f"Filters failed (Missing: {', '.join(missing_filters)})"
            elif score < z_thresh: reason = "Score too low"
        
        # Speedup Stats
        if not is_dup and not bypass_ai:
            query_stats['processed'] += 1
            if status_msg == "SAVED": query_stats['saved'] += 1
            if enable_speedup and query_stats['processed'] == 10:
                if query_stats['saved'] >= speedup_threshold:
                    query_stats['bypass_ai'] = True
                    st.success("⚡ Smart Speed-Up Triggered!")

        duration = time() - start_time
        
        # Result Object
        result_entry = {
            'paper': paper,
            'score': score,
            'tags': tags,
            'abstract_ai': ai_abs,
            'z_thresh': z_thresh,
            'passes': passes,
            'status_msg': status_msg,
            'process_time': duration,
            'reason': reason
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

# --- MAIN CONTROLS ---
with col_mode:
    search_mode = st.radio("🔍 Mode", ["Keyword Search", "Paste citation / page text", "Lookup by URL / PDF"], horizontal=True, key="search_mode_selector")
with col_max:
    # Deprecated for Automation mode, used for Manual Search
    max_res = st.slider("📄 Max articles (Manual Mode Only)", 5, 1000, value=prefs.get("max_results_value", 20), key="max_results_slider")

if search_mode == "Keyword Search":
    src = st.selectbox("📡 Source", ["Semantic Scholar", "PubMed", "Both"], index=0, key="search_source_selector")
    
    st.markdown("---")
    st.subheader("🤖 AI Filters")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.slider("📏 Min Abstract Len", 50, 500, value=prefs.get("min_abstract_length_chars", 150), key="abstract_length_slider")
        # Global Backup/Manual Mode Vector Slider
        st.slider("✅ Vector Score (Manual Mode)", 0.00, 1.00, value=prefs.get("vector_score_min_value", 0.50), step=0.01, format="%.2f", key="vector_score_min_slider")
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
        # --- ADVANCED QUERY TABLE MANAGER ---
        st.caption("🛠️ **Global Bulk Adjustments**")
        
        # Row 1: Vector & Reset
        gc1, gc2, gc_reset = st.columns([2, 1, 1])
        with gc1:
            vec_adjust = st.number_input("Global Vector Adjust (+/-)", value=0.00, step=0.01, format="%.2f", key="global_vec_adj")
        with gc2:
            if st.button("Apply Vec"):
                # Apply diff to all rows
                for q in st.session_state.automated_queries:
                    current_val = float(q.get('vector_min', 0.50))
                    new_val = max(0.0, min(1.0, current_val + vec_adjust))
                    q['vector_min'] = float(f"{new_val:.2f}")
                st.rerun()
        with gc_reset:
            if st.button("Reset Defaults"):
                for q in st.session_state.automated_queries:
                    q['vector_min'] = 0.50
                    q['start_rec'] = 0
                    q['stop_rec'] = 12 # RESET DEFAULT 12
                st.rerun()

        # Row 2: Global Range Settings (NEW)
        gr1, gr2, gr3 = st.columns([1, 1, 2])
        with gr1:
            g_start = st.number_input("Set All Start #", min_value=0, step=10, key="glob_start_in")
        with gr2:
            g_stop = st.number_input("Set All Stop #", min_value=1, value=12, step=10, key="glob_stop_in")
        with gr3:
            if st.button("Apply Range to All"):
                for q in st.session_state.automated_queries:
                    q['start_rec'] = int(g_start)
                    q['stop_rec'] = int(g_stop)
                st.rerun()
        
        # Data Editor Setup
        current_data = pd.DataFrame(st.session_state.automated_queries)
        
        if not current_data.empty:
            # Ensure proper types for display
            if 'vector_min' in current_data.columns:
                current_data['vector_min'] = current_data['vector_min'].astype(float)
            if 'start_rec' in current_data.columns:
                current_data['start_rec'] = current_data['start_rec'].astype(int)
            if 'stop_rec' in current_data.columns:
                current_data['stop_rec'] = current_data['stop_rec'].astype(int)
        
        if "Select" not in current_data.columns:
            current_data.insert(0, "Select", False)

        st.caption("📋 **Query Execution Queue** (Editable)")
        edited_df = st.data_editor(
            current_data,
            num_rows="dynamic",
            use_container_width=True,
            width='stretch', # Prepare for future deprecation
            key="query_table_editor",
            column_config={
                "Select": st.column_config.CheckboxColumn(width="small"),
                "query": st.column_config.TextColumn("Query String", required=True, width="large"),
                "folder": st.column_config.TextColumn("Collection ID", width="small", help="Optional Zotero Key"),
                "vector_min": st.column_config.NumberColumn("Vec Min", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
                "start_rec": st.column_config.NumberColumn("Start #", min_value=0, step=10),
                "stop_rec": st.column_config.NumberColumn("Stop #", min_value=1, step=10)
            },
            hide_index=True
        )
        
        # Sync Logic
        records = edited_df.to_dict('records')
        valid_records = [r for r in records if r.get('query') and str(r.get('query')).strip()]
        
        # Check if we need to update session state
        # Create comparable lists (removing Select key for comparison)
        def clean_rec(rec_list):
            return [{k:v for k,v in r.items() if k != 'Select'} for r in rec_list]
            
        if clean_rec(valid_records) != clean_rec(st.session_state.automated_queries):
             st.session_state.automated_queries = clean_rec(valid_records)

        # Delete Button Logic
        d_col, _, _ = st.columns([1,2,1])
        if d_col.button("🗑️ Delete Selected"):
            selected_rows = edited_df[edited_df.Select == True]
            if not selected_rows.empty:
                # Keep only unselected
                remaining = edited_df[edited_df.Select == False].drop(columns=['Select']).to_dict('records')
                st.session_state.automated_queries = remaining
                st.rerun()

        # --- CONFIG MANAGER UI (With On_Click Fix) ---
        st.markdown("---")
        with st.expander("💾 Load / Save Full Configuration", expanded=False):
            c_name, c_save, c_load = st.columns([3, 1, 1])
            
            # Refresh File List
            files = [f for f in os.listdir(CONFIG_DIR) if f.endswith(".json")]
            existing_file = c_name.selectbox("Select Config", [""] + files, key="config_file_select")
            new_file_name = c_name.text_input("Or create new file name", placeholder="my_research_config")
            
            target_save_name = new_file_name if new_file_name else existing_file
            
            if c_save.button("💾 Save") and target_save_name:
                save_full_state(target_save_name)
                
            # UPDATED: Use Callback to prevent Widget Instantiation Error
            c_load.button("📂 Load", on_click=load_full_state, args=(existing_file,))


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
# RESTORE SAVED SELECTION
saved_model_key = st.session_state.get("model_key_selector", m_keys[0])
try:
    default_idx = m_keys.index(saved_model_key)
except ValueError:
    default_idx = 0

selected_option = st.selectbox("🤖 Model", m_keys, index=default_idx, key="model_key_selector")

# 2. Check if user picked Ollama
is_local_ollama = (MODEL_OPTIONS[selected_option]["model_id"] == "LOCAL_OLLAMA")
st.session_state.use_ollama = is_local_ollama

# 3. If Local Ollama, show secondary dropdown for actual model
if is_local_ollama:
    local_models = get_ollama_models()
    if local_models:
        # RESTORE SAVED OLLAMA SELECTION
        saved_ollama = st.session_state.get("ollama_local_model_selector", "")
        try:
            ollama_idx = local_models.index(saved_ollama)
        except ValueError:
            ollama_idx = 0
            
        local_model_name = st.selectbox("💻 Select Local Ollama Model", local_models, index=ollama_idx, key="ollama_local_model_selector")
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
    
    # Map the selected query strings back to their objects to get folders
    all_objs = st.session_state.automated_queries
    # Create lookup (maps "query string" -> {object})
    query_map = {obj['query']: obj for obj in all_objs}
    
    for q_idx in range(start_q_idx, len(queries)):
        raw_item = queries[q_idx]
        
        # --- ROBUSTNESS FIX: Handle Dict vs String input ---
        if isinstance(raw_item, dict):
            query_str = raw_item.get("query", "")
            # --- LOGIC FIX: Extract Row-Specific Settings ---
            row_vec_min = raw_item.get("vector_min", 0.50)
            row_start = int(raw_item.get("start_rec", 0))
            row_stop = int(raw_item.get("stop_rec", 12))
            folder_override = raw_item.get("folder", "").strip()
        else:
            query_str = str(raw_item)
            # Default Fallbacks if passing raw strings
            row_vec_min = st.session_state.vector_score_min_slider
            row_start = 0
            row_stop = st.session_state.max_results_slider
            folder_override = ""
        # ---------------------------------------------------
        
        # Look up folder if not found in raw_item (for string-based flow)
        if not folder_override and query_str in query_map:
            folder_override = query_map[query_str].get('folder', "").strip()

        # Update state for resume (Query Level)
        st.session_state.cycle_state['query_idx'] = q_idx
        
        status.info(f"Processing: {query_str}")
        
        # --- SKIP EMPTY QUERIES ---
        if not query_str or not query_str.strip():
            st.warning(f"Skipping empty query at index {q_idx}")
            continue

        # Initialize stats for this query cycle
        query_stats = st.session_state.cycle_state.get('query_stats', {
            'processed': 0, 'saved': 0, 'bypass_ai': False
        })
        
        if folder_override:
            st.caption(f"📂 Target Collection: {folder_override}")
        
        # Calculate Limits based on Table Row
        total_limit = row_stop - row_start
        if total_limit <= 0: total_limit = 5 # Safety fallback
        
        # Calculate Offset for API based on Resume State
        # If we are resuming the SAME query (q_idx == start_q_idx), we add the paper_offset to the row's start.
        resume_adder = st.session_state.cycle_state['paper_offset'] if q_idx == start_q_idx else 0
        current_api_offset = row_start + resume_adder
        
        # If we are starting a NEW query, reset the paper_offset in state to 0
        if q_idx != start_q_idx: 
            st.session_state.cycle_state['paper_offset'] = 0
            st.session_state.cycle_state['query_stats'] = {'processed': 0, 'saved': 0, 'bypass_ai': False}
            query_stats = st.session_state.cycle_state['query_stats']
            current_api_offset = row_start # No resume adder

        # Calculate remaining limit (Total limit - what we already processed)
        remaining_limit = total_limit - resume_adder
        if remaining_limit <= 0:
            st.info(f"Skipping {query_str}: Range {row_start}-{row_stop} already processed.")
            continue

        # 1. Prepare Query
        final_query = query_str
        if search_mode == "Keyword Search" and st.session_state.get("use_boolean_checkbox"):
            res = gemini_boolean_query(query_str, actual_model_id)
            final_query = res.get("boolean_query", query_str)
            st.info(f"Boolean Query: {final_query}")

        # 2. Acquire
        if search_mode == "Keyword Search":
            source = st.session_state.search_source_selector
            if source in ["Semantic Scholar", "Both"]:
                # PASS ROW-SPECIFIC OFFSET and LIMIT
                for chunk in search_semantic_scholar(final_query, remaining_limit, current_api_offset):
                    # PASS vec_min_override
                    a, s = process_chunk_and_save(
                        chunk, 
                        final_query, 
                        100, 
                        0, 
                        status, 
                        prog, 
                        query_stats, 
                        collection_override=folder_override,
                        vec_min_override=row_vec_min 
                    )
                    total_ann += a; total_save += s
                    
            if source in ["PubMed", "Both"]:
                 p_res = search_pubmed_paged(final_query, remaining_limit, current_api_offset)
                 
                 # Chunk the results into batches of 50 for AI processing
                 total_pubmed = len(p_res)
                 pubmed_processed_count = 0

                 for batch in _chunks(p_res, 50): 
                    # PASS vec_min_override
                    a, s = process_chunk_and_save(
                        batch,
                        final_query,
                        total_pubmed, 
                        pubmed_processed_count, 
                        status,
                        prog,
                        query_stats,
                        collection_override=folder_override,
                        vec_min_override=row_vec_min 
                    )
                    total_ann += a
                    total_save += s
                    pubmed_processed_count += len(batch)

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
                    papers, 
                    final_query, 
                    len(papers), 
                    0, 
                    status, 
                    prog, 
                    query_stats, 
                    collection_override=folder_override,
                    vec_min_override=row_vec_min
                )
                total_ann += a; total_save += s

        elif search_mode == "Lookup by URL / PDF":
            papers = search_by_url_doi_pdf(st.session_state.url_or_doi)
            if papers:
                a, s = process_chunk_and_save(
                    papers, 
                    final_query, 
                    len(papers), 
                    0, 
                    status, 
                    prog, 
                    query_stats, 
                    collection_override=folder_override,
                    vec_min_override=row_vec_min
                )
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
    # Use unique index key for correct display in history
    for idx, item in enumerate(st.session_state.results_history):
        _display_paper_details(item, f"history_{idx}")
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
        valid_go = True
        qs = []
        if search_mode == "Keyword Search" and st.session_state.query_mode_selector == "Single Query":
             if not st.session_state.user_prompt_input.strip():
                 st.error("Please enter a topic.")
                 valid_go = False
             else:
                 # Single Query Mode: Fake a list for the generic runner
                 qs = [{
                    'query': st.session_state.user_prompt_input,
                    'vector_min': st.session_state.vector_score_min_slider, # Use fallback
                    'start_rec': 0, # Default Start
                    'stop_rec': st.session_state.get('max_results_slider', 20) # Default Stop
                 }]
        elif search_mode == "Keyword Search":
            # Check Table
            if not st.session_state.automated_queries:
                st.error("No queries in table.")
                valid_go = False
            else:
                qs = st.session_state.automated_queries # List of Dicts

        if valid_go:
            if not is_active:
                st.session_state.cycle_state['paper_offset'] = 0 # Default start is handled per-row now
                st.session_state.cycle_state['query_idx'] = 0
            
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
    # qs is already populated in the button block above
    # Check manual/single modes vs table mode
    if search_mode != "Keyword Search":
        # Create a Dummy object for Manual modes to satisfy the Dict requirement
        qs = [{'query': 'Manual Input', 'vector_min': 0.5, 'start_rec':0, 'stop_rec': st.session_state.get('max_results_slider', 20)}] 
        
    if qs:
        run_cycle_logic(qs)
    else:
        st.error("No query selected.")
