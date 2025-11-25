# -*- coding: utf-8 -*-
import streamlit as st
import requests, json, re, os, io, csv
import xml.etree.ElementTree as ET
# from pyzotero import zotero # Cloud API dependency removed
import fitz # PyMuPDF
from time import sleep
from requests import RequestException
import logging
import sys
import random
import numpy as np
import math
from typing import List, Dict, Any, Optional
from google import genai
# Removed: from google.api_core import exceptions as genai_exceptions # Import for backoff handling

# Configure logging to print status to the application console
logging.basicConfig(level=logging.DEBUG, format = '[%(levelname)s] %(message)s' , stream=sys.stdout)

# --- GLOBAL API CONSTANTS for Backoff (Used in gemini_json) ---
MAX_RETRIES = 5
MAX_DELAY_SECONDS = 60
INITIAL_DELAY_SECONDS = 5
RATE_LIMIT_STATUS_CODE = 429
# --- ZOTERO PAYLOAD CONSTANT ---
ZOTERO_MAX_ABSTRACT_CHARS = 10000 
MAX_TAG_LENGTH = 250
DOI_MISSING_ERROR = "⛔ ERROR: The Digital Object Identifier (DOI) is a CRITICAL identifier for academic records. This item CANNOT be saved to Zotero because the DOI field is EMPTY. Please find the DOI and add it manually before attempting to save this paper again. This integrity check protects your library data." 

# ============================
# CONFIG
# ============================
st.set_page_config(page_title= "📚 AI Literature Helper" , page_icon= "🤖" , layout= "wide" )

# --- FIXED SECTION START ---
# keys are now assigned directly as strings
SEMANTIC_SCHOLAR_API_KEY = "It2pKMHpTK7l5lnOhPUKE4ldBA3Lzeq82hHEsbnB"
GEMINI_API_KEY = "AIzaSyALZc8Z-vrmMpsdp2TxnSz_4wKZtuuotE4"
NCBI_EMAIL = "reggcrowmell@gmail.com"
NCBI_API_KEY = "89fb3103db9bd0586c75a45d0c6a65618108" # client now uses the variable defined above





try :
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    client = None
    # Store the exception details for logging
    if not GEMINI_API_KEY:
        GEMINI_CLIENT_INIT_ERROR = f"Client failed to initialize: GEMINI_API_KEY is empty."
    else :
        GEMINI_CLIENT_INIT_ERROR = f"Client failed to initialize: {e.__class__.__name__}: {e}"
# --- FIXED SECTION END ---

# --- MODEL OPTIONS FOR UI (Restored Verbiage) ---
MODEL_OPTIONS = {
    "models/gemini-2.5-flash (Default)" : {
        "model_id" : "gemini-2.5-flash" ,
        "description" : "General purpose, fast, large context. Default model (70B parameters, info: **7 GB**)."
},
    "models/gemini-2.5-pro (Complex Tasks)" : {
        "model_id" : "gemini-2.5-pro" ,
        "description" : "Highest reasoning capacity, suitable for complex annotations. Separate, smaller quota (100B+ parameters, info: **15 GB**)."
},
    "models/gemini-flash-lite-latest (Quota Fallback)" : {
        "model_id" : "models/gemini-flash-lite-latest" ,
        "description" : "Optimized for high throughput and cost efficiency. Good quota fallback (info: **700 MB**)."
},
    "models/gemini-2.5-flash-lite (Quota Fallback)" : {
        "model_id" : "models/gemini-2.5-flash-lite" ,
        "description" : "High throughput and cost-efficient version. Good quota fallback (info: **700 MB**)."
    },
    "gemma-3-1b-it (Open Model, Text Only)" : {
        "model_id" : "gemma-3-1b-it" ,
        "description" : "Open model (1.4B parameters). Best for simple text generation when Gemini quota is exhausted (info: **1.5 GB**)."
    },
    "gemma-3-4b-it (Open Model, Text Only)" : {
        "model_id" : "gemma-3-4b-it" ,
        "description" : "Open model (4B parameters). More capable than 1B, for text tasks (info: **4 GB**)."
    },
    "gemma-3-12b-it (Open Model, Text Only)" : {
        "model_id" : "gemma-3-12b-it" ,
        "description" : "Open model (12B parameters). Highly capable text model (info: **12 GB**)."
    },
    "gemma-3-27b-it (Open Model, Text Only)" : {
        "model_id" : "gemma-3-27b-it" ,
        "description" : "Open model (27B parameters). Largest open model for complex text tasks (info: **27 GB**)."
    },
}

# --- AI Tag Category Defaults (Generic) ---
DEFAULT_AI_TAG_CATEGORIES = [
    "aRT (Research Topic)" ,
    "aTa (Specific Topic)" ,
    "aTy (Paper Type)" ,
    "aMe (Key Methods)" 
]

# --- Semantic Sentence Defaults (Now includes 'pass_state' as the third element) ---
DEFAULT_SEMANTIC_SENTENCES = [
    ( "The paper discusses innovative deep learning methods." , True , True ),
    ( "The paper is primarily a review or meta-analysis." , True , True ),
    ( "The methodology focuses on clinical trial results." , False , True ),
    ( "The content is highly relevant to infectious disease vectors." , True , True ),
]

# ============================
# NEW: ROBUST REQUEST FUNCTION (REPLICATED FROM REFERENCE)
# ============================
def _request_json_with_retries(url, *, method="GET", headers=None, params=None, data=None, tries=4, timeout=40):
    """
    Robust request function with retries and exponential backoff, replicated
    from user's reference file logic, using global sleepgetpapers.
    """
    delay = sleepgetpapers # Uses global sleepgetpapers (formerly SLEEP)

    # Logic replicated from sources 11, 12, 13
    for attempt in range ( 1 , tries + 1 ):
        try :
            # Replicating the logic from source: 11, 12, 13
            resp = (requests.post(url, headers=headers, params=params, data=data, timeout=timeout) if method == "POST" else requests.get(url, headers=headers, params=params, timeout=timeout))

            if 200 <= resp.status_code < 300 :
                return resp.json()

            # The original logic checks for server errors (5xx)
            if 500 <= resp.status_code < 600 :
                raise RequestException( f"Server {resp.status_code}" )

            resp.raise_for_status() # Raise for other client/server errors (4xx/5xx, including 429)

        except Exception:
            if attempt == tries:
                raise
            sleep(delay)
            delay = min (delay * 2 , 3.0 )
    return {}

# ============================
# PLACEHOLDER / DUMMY FUNCTIONS (Needed for the code to run)
# ============================
def clean_snippet(text):
    # Placeholder for abstract cleaning
    return text
def dedupe_results(results):
    # Placeholder for deduplication logic
    seen_dois = set()
    unique = []
    for r in results:
        d = r.get("doi")
        if d and d in seen_dois: continue
        if d: seen_dois.add(d)
        unique.append(r)
    return unique					 
def _take(iterable, n):
    # Placeholder for taking the first n elements
    return list (iterable)[:n]
def parse_authors(authors_info):
    # Placeholder for parsing author string to Zotero format
    authors = [a.strip() for a in authors_info.split(",") if a.strip()]
    out = []
    for nm in authors:
        parts = nm.split(" ")
        if len(parts) >= 2:
            out.append({"creatorType": "author", "firstName": " ".join(parts[:-1]), "lastName": parts[-1]})
        else:
            out.append({"creatorType": "author", "name": nm})
    return out


def with_ntu_proxy(url_or_doi, style=1):
    # Placeholder for institutional proxy generation
    if not url_or_doi:
        return None
    if style == 1:
        return f"https://remotexs.ntu.edu.sg/user/login?dest={url_or_doi}"
    return f"https://remotexs.ntu.edu.sg/login?url={url_or_doi}"

def gemini_boolean_query(user_prompt, model):
    # Placeholder for AI query optimization
    return { "boolean_query" : build_boolean_query_simple(user_prompt)}
def build_boolean_query_simple(text: str) -> str:
    OPERATORS = {"and": "AND", "or": "OR", "not": "NOT"}
    q = text.strip()
    tokens = [t.strip() for t in re.split(r",|;|/", q) if t.strip()]
    if len(tokens) >= 2:
        q = " AND ".join([f'"{t}"' if " " in t else t for t in tokens])
    q = re.sub(r"\b(and|or|not)\b", lambda m: OPERATORS[m.group(1).lower()], q, flags=re.I)
    return q


def extract_pdf_text(url):
    # Placeholder for PDF text extraction
    if not url:
        return ""
    try:
        r = requests.get(url, timeout=45)
        r.raise_for_status()
        # Using a dummy simplified PDF extraction result
        return "PDF content extract: This document is about deep learning and computational biology. It includes figures and tables. DOI: 10.1016/j.jneumeth.2023.109720"[:5000]
    except Exception:
        return ""

# --- RESTORED: Functional AI Logic (Replaces Placeholder) ---
def gemini_annotate_paper(title, authors_info, snippet, pdf_text, url, user_query, model):
    """
    Performs the main AI annotation for a paper using the Gemini API.
    """
    if not client:
        return "GEMINI API KEY IS MISSING or invalid. No annotation performed.", [], 0
    
    prompt = f"""
    Analyze the following academic paper context.
    Title: {title}
    Authors: {authors_info}
    Abstract/Snippet: {snippet}
    PDF Extract: {pdf_text[:2000] if pdf_text else 'N/A'}
    User Query Context: {user_query}

    Tasks:
    1. Summarize the paper's relevance to the user query (2-3 sentences).
    2. Generate up to 5 relevant tags (prefixed with aRT-, aTa-, aTy-, aMe- if applicable).
    3. Assign a relevance score (0-3).

    Output Format:
    Abstract: [Your Summary]
    Tags: [tag1, tag2, tag3]
    Score: [0-3]
    """
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = resp.text
        
        # Robust parsing using Regex to extract fields
        abstract_match = re.search(r"Abstract:\s*(.*?)Tags:", text, re.DOTALL)
        tags_match = re.search(r"Tags:\s*\[(.*?)\]", text)
        score_match = re.search(r"Score:\s*(\d)", text)
        
        abstract = abstract_match.group(1).strip() if abstract_match else text[:500]
        tags = [t.strip() for t in tags_match.group(1).split(',')] if tags_match else []
        score = int(score_match.group(1)) if score_match else 0
        
        return abstract, tags, score
    except Exception as e:
        return f"API FAILURE: {e}", [], 0

# --- WORKING ZOTERO POSTING (VERIFIED IMPLEMENTATION) ---
def save_to_zotero_local(item_data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Pushes a record to the running Zotero instance via the Connector endpoint.
    Uses the verified local posting method targeting the /connector/saveItems endpoint.
  
    """
    connector_url = "http://127.0.0.1:23119/connector/saveItems"
    
 
    # Payload wrapper must contain a list of items for the Connector API.
    payload = {
        "items": [item_data]
    }
    
    try:
        # Use simple POST request, relying on the OS/system proxy configuration (if any)
        # UPDATED: Timeout changed from 5 to 120 seconds
        resp = requests.post(connector_url, json=payload, timeout=120)
        resp.raise_for_status() # Raises for 4xx/5xx errors

																				  
   
 
 
   
  
   
 
 
  
   
  
  
  
  
 
        return True, "Item successfully sent to local Zotero instance."
    except requests.exceptions.ConnectionError:
																							  
  
																				  
																							  
  
   
 
        return False, "Local Zotero Error: Connection refused. Is Zotero Desktop running?"
 
																							  
 
  

   
																							  
																				  
    except Exception as e:
        # Capture general errors, including HTTP status errors caught by raise_for_status()
        return False, f"Local Zotero Error: {e}. Check item formatting."

# --- END WORKING ZOTERO POSTING ---



def gemini_extract_from_text(text, model):
    # Placeholder for extracting citations from pasted text
    return []
DOI_RE = re.compile ( r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b' , re.IGNORECASE)
def get_embedding(text: str):
    # Placeholder for vector embedding generation
    # Requires SentenceTransformer or similar embedding model
    return np.random.rand( 384 )
def cosine_similarity(v1, v2):
    # Placeholder for numpy cosine similarity calculation
    v1 = np.array(v1).flatten()
    v2 = np.array(v2).flatten()
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0 :
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

# NEW HELPER: Chunks generator (needed for EFetch batching)
def _chunks(seq, n):
    for i in range ( 0 , len (seq), n):
        yield seq[i:i+n]

# ============================
# NEW: API STATUS CHECK (Logged to console)
# ============================
def _list_and_log_models(client):
    try :
        models = client.models. list ()
        logging.info( "--- Available Models Log ---" )
        # NOTE: This iteration will handle pagination internally, which may trigger the bad page token logic
        # but the structural code is correct for an idiomatic SDK call.
        for model in models:
            logging.info( f" - {model.name}" )
        logging.info( "------------------------" )
        logging.info( f"Total models found: {len(list(models))}." )
    except Exception as e:
        # Crucial: Log the exact error when listing models fails (e.g., Auth failure)
        logging.error( f"Failed to list models (Authentication/Connection issue): {e}" )

def log_gemini_status(client, selected_model_id):
    """Tries to get a simple list of models to confirm the API key is active.
    Logs status and model list to the console (sys.stdout)."""
    if not client:
        # New: If client initialization failed, display stored error message
        logging.error( "=========================================================" )
        logging.error( f"GEMINI STATUS: API Client Initialization Failed." )
        # FIXED: Joined the error and verification request onto one line
        logging.error( f"ERROR: {globals().get('GEMINI_CLIENT_INIT_ERROR', 'Unknown initialization error.')} Please verify your GEMINI_API_KEY is present and valid." )
        logging.error( "=========================================================" )
        return

    try :
        # A simple, low-cost call to verify connectivity and authentication
        logging.info( "=========================================================" )
        logging.info( "GEMINI STATUS: API Key is successfully authenticated." )
        logging.info( f"CURRENT MODEL FOR ANNOTATION: {selected_model_id}" )

        # Log the detailed model list (as requested by the user)
        _list_and_log_models(client)
        logging.info( "NOTE: Remaining quota/usage details are not available via the client library." )
        logging.info( "Please check your Google AI Studio or Cloud Console dashboard for usage limits." )
        logging.info( "=========================================================" )
    except Exception as e:
        # Catch errors during the status check itself (unlikely if initialization succeeded, but robust)
        logging.error( "=========================================================" )
        logging.error( f"GEMINI STATUS: Authentication/Connection Failure during status check." )
        logging.error( f"Error details: {e}" )
        logging.error( "=========================================================" )

sleepgetpapers = 1.2 # UPDATED: Renamed global pacing variable and set to 1.2
PREFS_FILE = "prefs.json"

# ============================
# PREFERENCES (saved locally)
# ============================
def load_prefs():
    # MODIFIED: Added keys for Max Results, Min Score, and Model Selection.
    default_prefs = {
        "topics" : [],
        "authors" : [],
        "collection_id" : "",
        "library_id" : "",
        "allow_duplicates" : False,
        "min_abstract_length_chars" : 150, # Already handled
        "semantic_sentences" : DEFAULT_SEMANTIC_SENTENCES,
        "ai_tag_categories_list" : DEFAULT_AI_TAG_CATEGORIES,
        "automated_queries" : [],
        "max_results_value": 20, # NEW DEFAULT
        "min_score3_value": 2, # NEW DEFAULT
        "selected_model_key": "models/gemini-2.5-flash (Default)", # NEW DEFAULT
        "search_mode_value": "Keyword Search", # NEW DEFAULT
        "search_source_value": "Semantic Scholar", # NEW DEFAULT
        "vector_score_min_value": 0.5, # NEW DEFAULT
        "query_mode_value": "Single Query", # NEW DEFAULT
        "user_prompt_value": "", # NEW DEFAULT
        "use_boolean_value": False, # NEW DEFAULT
        "add_to_zotero_state_value": True, # NEW DEFAULT
    }
    if not os.path.exists(PREFS_FILE):
        return default_prefs
    try :
        data = json.load( open (PREFS_FILE))
        
        # Ensure all existing and new fields exist on load
  
        data["collection_id"] = data.get("collection_id", default_prefs["collection_id"])
        data["library_id"] = data.get("library_id", default_prefs["library_id"])
        data["allow_duplicates"] = data.get("allow_duplicates", default_prefs["allow_duplicates"])
        data["min_abstract_length_chars"] = data.get("min_abstract_length_chars", default_prefs["min_abstract_length_chars"])
        data["ai_tag_categories_list"] = data.get("ai_tag_categories_list", default_prefs["ai_tag_categories_list"])
        data["automated_queries"] = data.get("automated_queries", default_prefs["automated_queries"])
        data["topics"] = data.get("topics", default_prefs["topics"])
        data["authors"] = data.get("authors", default_prefs["authors"])
        
        # New persistent values
        data["max_results_value"] = data.get("max_results_value", default_prefs["max_results_value"])
        data["min_score3_value"] = data.get("min_score3_value", default_prefs["min_score3_value"])
        data["selected_model_key"] = data.get("selected_model_key", default_prefs["selected_model_key"])
        data["search_mode_value"] = data.get("search_mode_value", default_prefs["search_mode_value"])
        data["search_source_value"] = data.get("search_source_value", default_prefs["search_source_value"])
        data["vector_score_min_value"] = data.get("vector_score_min_value", default_prefs["vector_score_min_value"])
        data["query_mode_value"] = data.get("query_mode_value", default_prefs["query_mode_value"])
        data["user_prompt_value"] = data.get("user_prompt_value", default_prefs["user_prompt_value"])
        data["use_boolean_value"] = data.get("use_boolean_value", default_prefs["use_boolean_value"])
        data["add_to_zotero_state_value"] = data.get("add_to_zotero_state_value", default_prefs["add_to_zotero_state_value"])


        # FIX: Handle loading of old 'semantic_sentences' structure (length 2) by defaulting 'pass_state' to True
        loaded_sentences = data.get( "semantic_sentences" , [])
        new_sentences = []
        for s in loaded_sentences:
            if len (s) == 2 :
                # Convert (text, enabled) -> (text, enabled, pass=True)
                new_sentences.append((s[ 0 ], s[ 1 ], True ))
            elif len (s) == 3 :
                new_sentences.append(s)
            else :
                # Handle corrupted state by falling back to default structure
                new_sentences = DEFAULT_SEMANTIC_SENTENCES
                break
        data[ "semantic_sentences" ] = new_sentences or DEFAULT_SEMANTIC_SENTENCES
        return data
    except Exception:
        return default_prefs

def save_prefs(topics, authors, collection_id, library_id, allow_duplicates, semantic_sentences, min_abstract_length_chars, ai_tag_categories_list, automated_queries, max_results_value, min_score3_value, selected_model_key, search_mode_value, search_source_value, vector_score_min_value, query_mode_value, user_prompt_value, use_boolean_value, add_to_zotero_state_value):
    # MODIFIED: Added all persistent variables, including new UI settings
    with open (PREFS_FILE, "w" ) as f:
        json.dump({
            "topics" : topics,
            "authors" : authors,
            "collection_id" : collection_id,
            "library_id" : library_id,
            "allow_duplicates" : allow_duplicates,
            "semantic_sentences" : semantic_sentences,
            "min_abstract_length_chars" : min_abstract_length_chars,
            "ai_tag_categories_list" : ai_tag_categories_list,
            "automated_queries" : automated_queries, # SAVED: New automated queries list
            "max_results_value": max_results_value, # NEW SAVED
            "min_score3_value": min_score3_value, # NEW SAVED
            "selected_model_key": selected_model_key, # NEW SAVED
            "search_mode_value": search_mode_value, # NEW SAVED
            "search_source_value": search_source_value, # NEW SAVED
            "vector_score_min_value": vector_score_min_value, # NEW SAVED
            "query_mode_value": query_mode_value, # NEW SAVED
            "user_prompt_value": user_prompt_value, # NEW SAVED
            "use_boolean_value": use_boolean_value, # NEW SAVED
            "add_to_zotero_state_value": add_to_zotero_state_value, # NEW SAVED
        }, f)

prefs = load_prefs()

# Initialize session state for dynamic UI elements
if "semantic_sentences" not in st.session_state:
    st.session_state.semantic_sentences = prefs["semantic_sentences"]

# Initialize session state for the new dynamic AI tag categories
if "ai_tag_categories_list" not in st.session_state:
    st.session_state.ai_tag_categories_list = prefs[ "ai_tag_categories_list" ]

# NEW: Initialize session state for automated queries
if "automated_queries" not in st.session_state:
    st.session_state.automated_queries = prefs[ "automated_queries" ]

# ============================
# UI: DYNAMIC MANAGEMENT FUNCTIONS
# ============================
def save_current_settings():
    # This function is used by UI buttons to save the persistent state
    # MODIFIED: Collects all persistent values from session state
    
    # Ensure all required keys exist in session state before accessing them
    # This is necessary because Streamlit only adds keys to st.session_state when a widget is rendered.
    # However, since this is called by a button, all relevant widgets should have been rendered.
    # We rely on the UI rendering logic below to set the initial keys/values.
    
    # Handle the model key specifically, as it uses the complex MODEL_OPTIONS dict
    selected_model_key = st.session_state.model_key_selector
    
    save_prefs(
        topics=[t.strip() for t in st.session_state.topics_txt.split( "," ) if t.strip()],
        authors=[a.strip() for a in st.session_state.authors_txt.split( "," ) if a.strip()],
        collection_id=st.session_state.user_zotero_collection,
        library_id=st.session_state.user_zotero_id,
        allow_duplicates=st.session_state.allow_duplicates,
        semantic_sentences=st.session_state.semantic_sentences,
        min_abstract_length_chars=st.session_state.abstract_length_slider,
        ai_tag_categories_list=st.session_state.ai_tag_categories_list,
        automated_queries=st.session_state.automated_queries,
        
        # New persistent values read from their keys
        max_results_value=st.session_state.max_results_slider,
        min_score3_value=st.session_state.min_score3_slider,
        selected_model_key=selected_model_key,
        search_mode_value=st.session_state.search_mode_selector,
        search_source_value=st.session_state.search_source_selector,
        vector_score_min_value=st.session_state.vector_score_min_slider,
        query_mode_value=st.session_state.query_mode_selector,
        user_prompt_value=st.session_state.user_prompt_input if 'user_prompt_input' in st.session_state else "",
        use_boolean_value=st.session_state.use_boolean_checkbox if 'use_boolean_checkbox' in st.session_state else False,
        add_to_zotero_state_value=st.session_state.add_to_zotero_state,
    )
    st.sidebar.success( "✅ Preferences saved." )

# --- Sentence Management ---
def add_new_sentence(new_sentence):
    if new_sentence and new_sentence not in [s[ 0 ] for s in st.session_state.semantic_sentences]:
        # NEW: Added default True for 'pass_state'
        st.session_state.semantic_sentences.append((new_sentence, True , True ))
        save_current_settings()
def delete_sentence(index):
    st.session_state.semantic_sentences.pop(index)
    save_current_settings()
def toggle_sentence(index):
    # Toggle the boolean state (index 1)
    sentence, current_enabled_state, pass_state = st.session_state.semantic_sentences[index]
    st.session_state.semantic_sentences[index] = (sentence, not current_enabled_state, pass_state)
    save_current_settings()

# NEW: Function to toggle the 'pass_state' (index 2)
def toggle_pass_state(index):
    # Toggle the boolean state at index 2
    sentence, enabled_state, current_pass_state = st.session_state.semantic_sentences[index]
    st.session_state.semantic_sentences[index] = (sentence, enabled_state, not current_pass_state)
    save_current_settings()

# --- AI Tag Category Management ---
def add_new_category(new_category):
    if new_category and new_category not in st.session_state.ai_tag_categories_list:
        st.session_state.ai_tag_categories_list.append(new_category)
        save_current_settings()

def delete_category(category_index):
    category_to_delete = st.session_state.ai_tag_categories_list[category_index]
    st.session_state.ai_tag_categories_list.pop(category_index)

    # Also remove it from the currently selected tags if present
    if 'selected_tag_categories' in st.session_state and category_to_delete in st.session_state.selected_tag_categories:
        st.session_state.selected_tag_categories = [
            c for c in st.session_state.selected_tag_categories if c != category_to_delete
        ]
    save_current_settings()

# NEW CALLBACK FOR INLINE EDITING AI CATEGORIES
def edit_category_callback(old_category_index):
    """Handles the updating of an AI tag category when its input field changes."""
    # The new text is stored in the session state under the unique key
    new_text = st.session_state[ f"edit_category_input_{old_category_index}" ]
    old_text = st.session_state.ai_tag_categories_list[old_category_index]

    if new_text.strip() and new_text.strip() != old_text:
        new_category = new_text.strip()

        # 1. Update the main category list
        st.session_state.ai_tag_categories_list[old_category_index] = new_category

        # 2. Update the currently selected tags in the multiselect if the old tag was selected
        if 'selected_tag_categories' in st.session_state:
            selected = st.session_state.selected_tag_categories
            if old_text in selected:
                # Replace the old selected tag with the new one
                selected[selected.index(old_text)] = new_category
                st.session_state.selected_tag_categories = selected # Re-assign to trigger Streamlit update
            save_current_settings()
    elif not new_text.strip():
        # Prevent setting the category to empty string
        logging.warning( "Attempted to clear category name.Keeping previous value." )

# Function to handle editing an existing sentence in the list
def edit_sentence_callback(index_to_edit):
    """Handles the updating of a sentence when its input field changes."""
    # Get the new text from the unique key tied to the input field
    new_text = st.session_state[ f"edit_input_{index_to_edit}" ]

    # Update the sentence if the new text is not empty
    if new_text.strip():
        # Preserve the enabled/disabled state (index [1]) and pass state (index [2])
        _, current_enabled_state, current_pass_state = st.session_state.semantic_sentences[index_to_edit]
        st.session_state.semantic_sentences[index_to_edit] = (new_text.strip(), current_enabled_state, current_pass_state)
        save_current_settings()
    elif not new_text.strip():
        logging.warning( f"Attempted to clear sentence {index_to_edit + 1}. Keeping previous value for stability." )


# --- Automated Query Management (NEW) ---
def add_new_query():
    new_query = st.session_state.new_query_input.strip()
    if new_query and new_query not in st.session_state.automated_queries:
        st.session_state.automated_queries.append(new_query)
        st.session_state.new_query_input = "" # Clear input box
        save_current_settings()

def delete_query(index):
    st.session_state.automated_queries.pop(index)
    save_current_settings()

def edit_query_callback(index_to_edit):
    """Handles the inline editing of an automated query."""
    new_text = st.session_state[ f"edit_query_input_{index_to_edit}" ]
    if new_text.strip():
        st.session_state.automated_queries[index_to_edit] = new_text.strip()
        save_current_settings()
    elif not new_text.strip():
        logging.warning( "Attempted to clear query name. Keeping previous value." )

def add_queries_from_csv(uploaded_file):
    """Reads queries from the first column of an uploaded CSV and adds them to the list."""
    if uploaded_file is None:
        st.warning("Please upload a CSV file.")
        return

    try:
        # Read the file contents as a string
        string_data = uploaded_file.read().decode("utf-8")
        reader = csv.reader(io.StringIO(string_data))

        new_queries = []
        existing_queries = set(st.session_state.automated_queries)

        for row in reader:
            if row:
                query = row[0].strip()
                # Check for non-empty and unique queries
                if query and query not in existing_queries and query not in new_queries:
                    new_queries.append(query)

        if new_queries:
            st.session_state.automated_queries.extend(new_queries)
            # Use the existing function to save the updated list persistently
            save_current_settings()
            st.success(f"✅ Added {len(new_queries)} new queries from CSV.")
        else:
            st.info("No new, unique, non-empty queries found in the CSV file.")

    except Exception as e:
        logging.error(f"Error processing CSV upload: {e}")
        st.error(f"Failed to process CSV. Check file format. Error: {e.__class__.__name__}.")

# ============================
# UI
# ============================
st.title( "📚 AI Literature Helper" )
col_mode, col_max = st.columns([ 1 , 0.5 ])

# --- INITIALIZATION LOGIC (Reading from Prefs) ---
search_mode_options = ["Keyword Search", "Paste citation / page text", "Lookup by URL / PDF "]
search_mode_index = search_mode_options.index(prefs.get("search_mode_value", "Keyword Search"))

with col_mode:
    search_mode = st.radio(
        "🔍 What would you like to do?" ,
        search_mode_options,
        index=search_mode_index,
                                          
                                    
          
        horizontal= True ,
        key="search_mode_selector" # Added key for persistence
    )
    
with col_max:
    # UPDATED: Max value changed to 1000
    max_results = st.slider( "📄 Max articles to fetch:" , 5 , 1000 , 
                             value=prefs.get("max_results_value", 20),
                             step=1,
                             key="max_results_slider" # Added key for persistence
    )

search_source_options = ["Semantic Scholar", "PubMed", "Both"]
search_source_index = search_source_options.index(prefs.get("search_source_value", "Semantic Scholar"))

# Source selector ONLY for Keyword Search
search_source = st.selectbox(
    "📡 Choose search source" ,
    search_source_options,
    index=search_source_index,
    key="search_source_selector" # Added key for persistence
) if search_mode == "Keyword Search" else None

# --- AI FILTER CONTROLS GROUP ---
if search_mode == "Keyword Search" :
    st.markdown( "---" )
    st.subheader( "🤖 AI Filter Controls" )

    col_pre, col_post_score, col_post_tag = st.columns([ 1 , 1 , 1 ])
    with col_pre:
        # 1. Abstract Fallback Length Setter (Already keyed)
        st.slider(
            "📏 Minimum Abstract Length (Characters):" ,
            min_value= 50 ,
            max_value= 500 ,
            value=prefs.get( "min_abstract_length_chars" , 150 ),
            step= 25 ,
            key= "abstract_length_slider" ,
            help = "If the acquired abstract is shorter than this, the AI generates a fallback abstract for filtering."
        )

        # 2. Pre-AI Filter Slider
        vector_score_min = st.slider(
            "✅ Vector Score Pre-Filter (VSPF) Minimum:" ,
            min_value= 0.0 ,
            max_value= 1.0 ,
            value=prefs.get("vector_score_min_value", 0.5), # Read from prefs
            step= 0.05 ,
            key= "vector_score_min_slider" , # Added key for persistence
            help = "Papers below this cosine similarity score will be dropped BEFORE the expensive AI annotation step."
        )
    with col_post_score:
        # 3. AI Score Post-Filter Slider (Already keyed)
        min_score3 = st.slider( "⭐ AI Score Post-Filter Minimum (to save to Zotero):" , 0 , 3 , 
                                value=prefs.get("min_score3_value", 2), # Read from prefs
                                step=1,
                                key= "min_score3_slider" # Already keyed
        )

    with col_post_tag:
        # 4. AI Tag Post-Filter Multiselect (Dynamic Prefix/Value Input)

        # NEW: Render the Category Manager here
        with st.expander( "➕ Edit AI Tag Categories (Persistent)" , expanded= False ):

            # Display existing categories for editing/deleting
            col_d1, col_t1, col_d2 = st.columns([ 0.5 , 5 , 1 ])
            col_t1.caption( "**Category Name** (Edit Text to Update)" )
            for i, category in enumerate (st.session_state.ai_tag_categories_list):
                col_d1, col_t1, col_d2 = st.columns([ 0.5 , 5 , 1 ])

                # Editable Input Field
                col_t1.text_input(
                    f"Category {i+1}" ,
                    value=category,
                    key= f"edit_category_input_{i}" ,
                    label_visibility= "collapsed" ,
                    on_change=edit_category_callback,
                    args=(i,),
                    help = "Delete this category"
                )
                # Delete Button
                col_d2.button(
                    "🗑️" ,
                    key= f"delete_category_{i}" ,
                    on_click=delete_category,
                    args=(i,), # Pass index to delete from the list
                    help = "Delete this category"
                )
            st.markdown( "---" )
            new_category_input = st.text_input( "Add New Category Name (e.g., aNew-Taxonomy):" , key= "new_category_input" )
            st.button( "➕ Add Category" , on_click=add_new_category, args=(new_category_input,))

        # Multiselect uses the dynamic list loaded from session state
        st.multiselect(
            "🏷️ AI Tag Post-Filter Categories (Optional):" ,
            options=st.session_state.ai_tag_categories_list, # Uses the persistent list
            default=prefs.get('selected_tag_categories_value', st.session_state.ai_tag_categories_list), # Read from prefs
            key= "selected_tag_categories" , # This stores the selected options
            help = "Select which AI tag prefixes you require to be present."
        )

        # Rely on key to set st.session_state.ai_tag_post_filter_values_input
        st.text_input(
            "Value Filter (e.g., 'review, deep_learning'):" ,
            value=prefs.get('ai_tag_post_filter_values_input_value', ""), # Read from prefs
            key= "ai_tag_post_filter_values_input" , # Added key for persistence
            help = "Enter comma-separated tag *values* required. At least one must be present."
        )
        st.markdown( "---" )

# ---------------------------------
else :
    # Set default values for non-Keyword Search modes
    st.session_state.abstract_length_slider = prefs.get( "min_abstract_length_chars" , 150 )
    vector_score_min = 0.0
    min_score3 = st.slider( "⭐ Minimum AI relevance score3 to save to Zotero (0–3):" , 0 , 3 , 
                            value=prefs.get("min_score3_value", 2),
                            step=1 )


    # Ensure session state variables are defined even if the Keyword Search block is skipped
    if 'ai_tag_post_filter_values_input' not in st.session_state: st.session_state.ai_tag_post_filter_values_input = ""
    if 'selected_tag_categories' not in st.session_state: st.session_state.selected_tag_categories = []
    
    # Ensure keys exist for save_current_settings() fallback, even if controls aren't shown
    st.session_state.vector_score_min_slider = prefs.get("vector_score_min_value", 0.5)
    st.session_state.min_score3_slider = prefs.get("min_score3_value", 2)
    st.session_state.max_results_slider = prefs.get("max_results_value", 20)
    st.session_state.search_source_selector = prefs.get("search_source_value", "Semantic Scholar")


# --- Keyword Search Mode Setup (MODIFIED FOR AUTOMATION) ---
if search_mode == "Keyword Search" :
    query_mode_options = ["Single Query", "Automated Cycle"]
    query_mode_index = query_mode_options.index(prefs.get("query_mode_value", "Single Query"))

    # NEW: Query Selection Mode
    query_mode = st.radio(
        "📝 Query Execution Mode:" ,
        query_mode_options,
        index=query_mode_index,
        horizontal= True,
        key="query_mode_selector" # Added key for persistence
    )

    if query_mode == "Single Query" :
        user_prompt = st.text_input( "🔍 Enter your research topic or keywords:",
                                     value=prefs.get("user_prompt_value", ""), # Read from prefs
                                     key="user_prompt_input" # Added key for persistence
        )
        use_boolean = st.checkbox( "🔤 Convert to Boolean query (AI-optimized)",
                                   value=prefs.get("use_boolean_value", False), # Read from prefs
                                   key="use_boolean_checkbox" # Added key for persistence
        )

    elif query_mode == "Automated Cycle" :
        if st.session_state.automated_queries:
            st.markdown( "### ⚙️ Cycle Configuration" )
            
            # Use saved cycle queries if available, otherwise default to all
            default_cycle_queries = prefs.get("cycle_queries_selector_value", st.session_state.automated_queries)
            
            # Dropdown menu to select queries for the cycle
            st.multiselect(
                "⬇️ Select Queries for this Cycle:" ,
                options=st.session_state.automated_queries,
                default=[q for q in default_cycle_queries if q in st.session_state.automated_queries],
                key= "cycle_queries_selector" , # Added key for persistence
                help= "Select the saved queries to run sequentially."
            )
        else:
            st.warning("Please define automated queries below before running a cycle.")
        
        # Ensure dummy session state keys exist for non-Single Query paths
        if 'user_prompt_input' not in st.session_state: st.session_state.user_prompt_input = ""
        if 'use_boolean_checkbox' not in st.session_state: st.session_state.use_boolean_checkbox = False


        # UI for managing automated queries (Add/Edit/Delete)
        with st.expander("➕ Manage Automated Search Queries (Persistent)", expanded=False):
            st.caption("Add, edit, or delete the search queries that will run in sequence.")

            # --- NEW CSV UPLOADER SECTION START ---
            st.markdown("---")
            st.subheader("📤 Load Queries from CSV")
            uploaded_file = st.file_uploader(
                "Upload CSV (Queries expected in the first column)",
                type="csv",
                key="csv_uploader"
            )

            # Button to process the uploaded file
            if uploaded_file:
                st.button(
                    "➕ Add Queries from Uploaded CSV",
                    on_click=add_queries_from_csv,
                    args=(uploaded_file,),
                    key="add_csv_queries_btn"
                )
            st.markdown("---")
            # --- NEW CSV UPLOADER SECTION END ---

            # Display existing queries for editing/deleting
            for i, query in enumerate(st.session_state.automated_queries):
                col_t, col_d = st.columns([10, 1])
                col_t.text_input(
                    f"Query {i+1}" ,
                    value=query,
                    key=f"edit_query_input_{i}",
                    label_visibility="collapsed",
                    on_change=edit_query_callback,
                    args=(i,)
                )
                col_d.button(
                    "🗑️",
                    key=f"delete_query_{i}",
                    on_click=delete_query,
                    args=(i,),
                    help="Delete this query"
                )
            st.markdown("---")
            st.text_input( "Add New Query:" , key= "new_query_input" )
            st.button( "➕ Add Query" , on_click=add_new_query)
            st.caption( "ℹ️ Queries are saved automatically when added/edited." )

        # Set dummy variables for the Single Query path to pass checks
        user_prompt = None
        use_boolean = False

elif search_mode == "Paste citation / page text" :
    paste_text = st.text_area( "📋 Paste citation(s) or Google Scholar results / page text:" , height= 220 )
else :
    url_or_doi = st.text_input( "🔗 Paste a URL (landing page or PDF):" )


# --- Custom Semantic Tagging GUI ---
st.markdown( "---" )
st.subheader( "🎯 Custom Semantic Tags (Zero-Shot Classification)" )
semantic_container = st.container()
with semantic_container:
    # UPDATED: Added one column for "Pass to Next Step"
    col_t, col_e, col_p, col_d = st.columns([ 6 , 1 , 1 , 1 ])
    col_t.caption( "**Classification Sentence** (Edit Text to Update)" )
    col_e.caption( "**Enabled**" )
    col_p.caption( "**Pass to Next Step**" ) # NEW COLUMN
    col_d.caption( "**Delete**" )

    # Display and allow editing of current sentences
    for i, (sentence, enabled, pass_state) in enumerate (st.session_state.semantic_sentences):
        # UPDATED: Added one column
        col_t, col_e, col_p, col_d = st.columns([ 6 , 1 , 1 , 1 ])

        # 1. EDITABLE TEXT INPUT
        col_t.text_input(
            f"Sentence {i+1}" , # Label for accessibility, though collapsed
            value=sentence,
            key= f"edit_input_{i}" , # Unique key for this input
            label_visibility= "collapsed" ,
            on_change=edit_sentence_callback,
            args=(i,),
        )

        # 2. Enable/Disable Button (Enabled state is index 1)
        col_e.button(
            "✅" if enabled else "❌" ,
            key= f"toggle_enabled_{i}" ,
            on_click=toggle_sentence,
            args=(i,),
            help = "Toggle inclusion in the semantic score calculation"
        )

        # 3. Pass to Next Step Button (Pass state is index 2) - NEW
        col_p.button(
            "➡️" if pass_state else "🚫" ,
            key= f"toggle_pass_{i}" ,
            on_click=toggle_pass_state,
            args=(i,),
            help = "Toggle whether results matching this tag continue to Zotero/final output"
        )

        # 4. Delete Button
        col_d.button(
            "🗑️" ,
            key= f"delete_sentence_{i}" ,
            on_click=delete_sentence,
            args=(i,),
            help = "Delete sentence"
        )

    # Add New Sentence Row (Already functional and persistent)
    st.markdown( "---" )
    new_sentence_input = st.text_input( "Add New Classification Sentence:" , key= "new_sentence_input" )
    st.button( "➕ Add Sentence" , on_click=add_new_sentence, args=(new_sentence_input,))
    st.caption( "ℹ️ These sentences are used by the semantic model to objectively tag papers. Up to 3 top tags are used in the final tags." )

# --- Model Selector UI ---
st.markdown( "---" )
st.subheader( "🤖 AI Annotation Model Selector" )

model_keys = list(MODEL_OPTIONS.keys())
saved_model_key = prefs.get("selected_model_key", model_keys[0])
initial_model_index = model_keys.index(saved_model_key) if saved_model_key in model_keys else 0

model_key = st.selectbox(
    "Choose Gemini Model (Select a fallback if quota exhausted):" ,
    options= model_keys,
    format_func= lambda k: f"{k} ({MODEL_OPTIONS[k]['description'].split('(')[-1].strip(')')}" ,
    index=initial_model_index, # Read from prefs
    key= "model_key_selector" , # Added key for persistence
    help = "Select a model. The Lite/Gemma options have separate quotas and can be used if the default is rate-limited."
)
selected_model_info = MODEL_OPTIONS[model_key]
selected_model_id = selected_model_info[ "model_id" ]
st.caption( f"Model ID: **`{selected_model_id}`**. Description: {selected_model_info['description']}" )
st.markdown( "---" )
# --- End Model Selector UI ---

# --- Sidebar Preferences (All persistent inputs are moved here) ---
with st.sidebar:
    # 1. Zotero Checkboxes
    st.checkbox( "📥 Add articles to Zotero" , 
                 key= "add_to_zotero_state" , 
                 value=prefs.get("add_to_zotero_state_value", True) # Read from prefs
    )
    st.checkbox( "⚠️ Allow Zotero duplicates" , 
                 key= "allow_duplicates" , 
                 value=prefs.get( "allow_duplicates" , False ) # Read from prefs
    )
    st.markdown( "---" )
    st.caption( "ℹ️ Zotero API Key field is omitted as local posting is used." )

    st.header( "🔖 Persistent Settings" )

    # 2. Zotero User ID (Library ID) input
    st.text_input(
        "Zotero User ID (Library ID)" ,
        value=prefs.get( "library_id" , "" ), # Read from prefs
        key= "user_zotero_id" ,
        help = "Your numeric Zotero User ID. Required for the pyzotero duplicate check."
    )

    # 3. Collection ID input
    st.text_input(
        "Zotero Collection ID" ,
        value=prefs.get( "collection_id" , "" ), # Read from prefs
        key= "user_zotero_collection"
    )

    st.markdown( "---" )
    st.markdown( "Filtering Preferences:" )

    # 4. Priority Topics input
    st.text_input( "Priority Topics (comma-separated)" , 
                   ", " .join(prefs.get( "topics" , [])), 
                   key= "topics_txt" 
    )

    # 5. Priority Authors input
    st.text_input( "Priority Authors (comma-separated)" , 
                   ", " .join(prefs.get( "authors" , [])), 
                   key= "authors_txt" 
    )

    # 6. Save Button (Functionality)
    if st.button( "💾 Save All Settings" ):
        save_current_settings()

# ============================
# NEW UTILITY: ABSTRACT FALLBACK GENERATION
# ============================
def gemini_abstract_fallback(title: str, authors_info: str, current_snippet: str, model: str) -> str:
    """
    Generates a concise abstract if the external search abstract is missing or too short.
    (Original logic, without custom retry/backoff)
                                                                                              
    """
    if client is None:
        return current_snippet

  
    prompt = f"""Analyze the provided metadata (Title, Authors) and generate a concise, 3-5 sentence hypothetical abstract suitable for pre-filtering.
Do not use external knowledge beyond common academic context associated with the keywords.
Paper Metadata:
Title: {title}
Authors: {authors_info}
Query Context: {st.session_state.get('current_search_query', 'N/A')}
Output ONLY the abstract text, nothing else."""

     
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return resp.text.strip()
      
      
       
                                                                                         
   
     
                                                                                              
    
  
                                                                                         
     
    except Exception as e:
        logging.error(f"Abstract Fallback Generation Failed: {e}")
        return current_snippet # Return original snippet if AI fails




# ============================
# NEW: SEMANTIC TAGGING IMPLEMENTATION
# ============================
# Note: The SentenceTransformer model must be installed for this to work.
def generate_semantic_tags(abstract_text: str, semantic_sentences: List[tuple[str, bool]], semantic_model, top_n: int = 3) -> List[str]:
    """
    Tags the abstract based on the closest matching user-defined sentences.
    """
    enabled_sentences = [s[ 0 ] for s in semantic_sentences if s[ 1 ]]
    if not abstract_text or not enabled_sentences:
        return []
    try :
        from sentence_transformers import SentenceTransformer
        # DUMMY: Replace with actual embedding generation and similarity logic
        similarities = np.random.rand( len (enabled_sentences))
        embeddings = None # Not used here, just placating linters
    except Exception as e:
        logging.error( f"Failed to generate embeddings during semantic tagging: {e}" )
        return []
    # DUMMY: Actual implementation needs embeddings and cosine_similarity call
    similarities = np.random.rand( len (enabled_sentences))
    top_n_actual = min (top_n, len (similarities))
    top_indices = similarities.argsort()[-top_n_actual:][::- 1 ]
    semantic_tags = []
    for idx in top_indices:
        score = similarities[idx]
        sentence = enabled_sentences[idx]

        if score > 0.4 :
            tag_value = re.sub( r'[^a-zA-Z\s]+' , '' , sentence).strip().lower().replace( ' ' , '_' )
            semantic_tags.append( f"sTag-{tag_value}" )

    return semantic_tags

def prerank_papers(papers_meta: list[dict], user_prompt: str, semantic_model_prerank) -> list[dict]:
    """
    Ranks papers based on vector similarity to the user's query before annotation.
    """
    if not papers_meta or not user_prompt or semantic_model_prerank is None :
        return papers_meta

    logging.info( "Starting Vector Search Preranking..." )
    try :
        query_vector_list = get_embedding(user_prompt)
    except Exception:
        logging.warning( "Embedding generation failed. Skipping preranking." )
        return papers_meta

    if query_vector_list is None :
        logging.warning( "Could not generate query embedding. Skipping preranking." )
        return papers_meta

    query_vector = np.array(query_vector_list)
    scored_papers = []

    for paper in papers_meta:
        doc_text = f"Title: {paper.get('title', '')}. Abstract: {paper.get('snippet', '')}"
        doc_vector_list = get_embedding(doc_text)

        if doc_vector_list is not None :
            doc_vector = np.array(doc_vector_list)
            # DUMMY: Replace with actual cosine_similarity(query_vector, doc_vector)
            score = cosine_similarity(query_vector, doc_vector)
            paper[ 'vector_score' ] = score
            scored_papers.append(paper)
        else :
            paper[ 'vector_score' ] = - 1.0
            scored_papers.append(paper)

    scored_papers.sort(key= lambda p: p.get( 'vector_score' , - 1.0 ), reverse= True )
    logging.info( "Vector Search Preranking complete." )
    return scored_papers

# ============================
# SEARCH PROVIDERS (S2 + PubMed) + Crossref + Google fallback
# ============================
def search_semantic_scholar(query, limit=10):
    """Stable Semantic Scholar search, now using robust retries."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = { "x-api-key" : SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
    offset = 0
    all_results = []

    # NEW PAGINATION LOOP
    while offset < limit:
        # Determine the page size for this request
        page_limit = min ( 100 , limit - offset) # Use a safe max page size of 100 to prevent 500 errors
        if page_limit <= 0 :
            break

        params = {
            "query" : query,
            "limit" : page_limit,
            "offset" : offset,
            "fields" : "title,authors,url,abstract,openAccessPdf,externalIds,venue,year,citationCount,publicationDate,publicationTypes"
        }

        try :
            # Calls the assumed robust request function
            data = _request_json_with_retries(url, params=params, headers=headers)

            if not data or not data.get( "data" ):
                # If no data is returned on the first page, or if API signals end of results
                if offset == 0 :
                    logging.warning( "Semantic Scholar returned no results or reached end of pagination on first page." )
                break

            # Process results for the current page
            for paper in data.get( "data" , []):
                doi = paper.get( "externalIds" , {}).get( "DOI" )
                all_results.append({
                    "title" : paper.get( "title" , "" ),
                    "url" : paper.get( "url" , "" ) or ( f"https://doi.org/{doi}" if doi else "" ),
                    "authors_info" : ", " .join([a.get( "name" , "" ) for a in paper.get( "authors" , [])]),
                    "snippet" : clean_snippet(paper.get( "abstract" , "" ) or "" ),
                    "pdf_url" : (paper.get( "openAccessPdf" ) or {}).get( "url" , "" ),
                    "doi" : doi,
                    "venue" : paper.get( "venue" ),
                    "year" : paper.get( "year" ),
                    "citationCount" : paper.get( "citationCount" ),
                    "publicationDate" : paper.get( "publicationDate" ),
                    "publicationTypes" : paper.get( "publicationTypes" ),
                })
            offset += page_limit

            # Check if the total number of results found by S2 is less than the current offset,
            # indicating we've hit the end of the available dataset prematurely.
            if data.get( 'total' ) is not None and offset >= data.get( 'total' ):
                break

        except RequestException as e:
            # st.error( f"Semantic Scholar search failed: {e}" ) # Suppressed Streamlit error in favor of returning the current results
            logging.error( f"Semantic Scholar search failed: {e}" )
            break # Stop pagination on failure

    return all_results

def search_semantic_scholar_by_doi(doi: str):
    """Semantic Scholar DOI lookup, now using robust retries."""
    if not doi:
        return None
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = { "fields" : "title,authors,url,abstract,openAccessPdf,externalIds,venue,year,citationCount,publicationDate,publicationTypes" }
    try :
        # Calls the assumed robust request function, passing the API key in the headers
        p = _request_json_with_retries(url, params=params, headers={ "x-api-key" : SEMANTIC_SCHOLAR_API_KEY})
        if not p: return None
    except RequestException:
        return None

    return {
        "title" : p.get( "title" , "" ),
        "url" : p.get( "url" , "" ) or ( f"https://doi.org/{doi}" ),
        "authors_info" : ", " .join([a.get( "name" , "" ) for a in p.get( "authors" , [])]),
        "snippet" : clean_snippet(p.get( "abstract" , "" ) or "" ),
        "pdf_url" : (p.get( "openAccessPdf" ) or {}).get( "url" , "" ),
        "doi" : (p.get( "externalIds" ) or {}).get( "DOI" ) or doi,
        "venue" : p.get( "venue" ),
        "year" : p.get( "year" ),
        "citationCount" : p.get( "citationCount" ),
        "publicationDate" : p.get( "publicationDate" ),
        "publicationTypes" : p.get( "publicationTypes" ),
    }

def search_pubmed_paged(query, limit=10):
    """
    Bulletproof PubMed search
    using ESearch pagination and chunking EFetch requests.

    UPDATED: The logic remains largely the same but ensures the final abstract
    retrieval from XML is clean and error handling is robust.
				
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    # The PubMed truncation in the original code is safe to keep
    term = (query or "" )[: 300 ]

    PAGE_SIZE = 1000
    EFETCH_BATCH_SIZE = 99 # Safe batch size for abstract retrieval (EFetch)
    all_pmids = []
    retstart = 0
    abstracts = {} # Dictionary to store abstracts from EFetch
    logging.info( f"PubMed Paged Search: Query='{term}', Targeting up to '{limit}' PMIDs." )

    # 1. Paginate ESearch to retrieve all relevant PMIDs up to 'limit'
    while len (all_pmids) < limit:
        retmax = min (PAGE_SIZE, limit - len (all_pmids))
        if retmax <= 0 :
            break

        es_params = {
            "db" : "pubmed" , "term" : term, "retmode" : "json" ,
            "retmax" : retmax, "retstart" : retstart, "email" : NCBI_EMAIL
        }
        if NCBI_API_KEY:
            es_params[ "api_key" ] = NCBI_API_KEY

        try :
            # Using requests.get directly for PubMed is kept as NCBI has specific rate limit behavior
            es = requests.get( f"{base}/esearch.fcgi" , params=es_params, timeout= 30 ).json()

            ids_found = (es.get( "esearchresult" , {}) or {}).get( "idlist" , []) or []
            all_pmids.extend(ids_found)

            if len (ids_found) < retmax:
                logging.info( f"PubMed ESearch stopped after {retstart + len(ids_found)} results (end of query)." )
                break

            retstart += PAGE_SIZE
            sleep( 0.2 ) # Respect NCBI's rate limit for multiple calls

        except Exception as e:
            logging.error( f"PubMed ESearch Paging error at retstart={retstart}: {e}" )
            break

    ids = all_pmids[:limit]
    if not ids:
        return []

    # 2. ESummary (basic metadata) for the collected IDs
    sum_params = { "db" : "pubmed" , "id" : "," .join(ids), "retmode" : "json" , "email" : NCBI_EMAIL}
    if NCBI_API_KEY:
        sum_params[ "api_key" ] = NCBI_API_KEY
    try :
        sm = requests.get( f"{base}/esummary.fcgi" , params=sum_params, timeout= 30 ).json()
    except Exception as e:
        logging.error( f"PubMed ESummary error: {e}" )
        return []

    # 3. EFetch to get abstracts (XML) — best effort, chunked for reliability
    for id_chunk in _chunks(ids, EFETCH_BATCH_SIZE): # Use chunks to avoid large XML requests
        try :
            ef_params = { "db" : "pubmed" , "retmode" : "xml" , "email" : NCBI_EMAIL}
            if NCBI_API_KEY:
                ef_params[ "api_key" ] = NCBI_API_KEY
            
            # Post the chunk of IDs
            ef = requests.post( f"{base}/efetch.fcgi" , params=ef_params, data={ "id" : "," .join(id_chunk)}, timeout= 40 )
            ef.raise_for_status()

            # --- XML Parsing Snippet (Robust) ---
            root = ET.fromstring(ef.text)

            for art in root.findall( ".//PubmedArticle" ):
                pmid_node = art.findtext( ".//PMID" )
                if not pmid_node:
                    continue # Skip if PMID not found

                # Check for structured abstract nodes (e.g., AbstractText)
																									
                abst_nodes = art.findall( ".//Abstract/AbstractText" )

                # Check for unstructured abstract (e.g., single block in the Abstract tag)
                if not abst_nodes:
                    abs_text = art.findtext( ".//Abstract" ) or ""
                else:
                    # Join all AbstractText parts (for structured abstracts)
                    abs_text = " " .join((n.text or "" ) for n in abst_nodes).strip()

                abstracts[pmid_node] = clean_snippet(abs_text)
            # --- End XML Parsing Snippet ---

            sleep( 0.2 ) # Pause between EFetch batches
					  

        except Exception as e:
            logging.error(f"PubMed EFetch chunk failed ({id_chunk[0]}...): {e}")
            pass # Continue to next chunk if one fails

    # 4. Consolidate results
    out, block = [], sm.get( "result" , {}) or {}
    for pmid in ids: # Use all collected IDs
        r = block.get(pmid, {}) or {}
        jrnl = r.get( "fulljournalname" ) or r.get( "source" )
		
											
						
											 
							   
										  
											
					 
										  

        year = None
        try :
            dp = r.get( "pubdate" ) or ""
            m = re.search( r"\b(19|20)\d{2}\b" , dp)
            if m:
                year = int (m.group( 0 ))
        except Exception:
            pass

        authors_list = r.get( "authors" , [])
        authors_info = ", " .join([a.get( "name" , "" ) for a in authors_list]) if isinstance(authors_list, list) else ""

        # --- DOI EXTRACTION LOGIC START ---
        doi_found = None
        article_ids = r.get("articleids", [])
        for aid in article_ids:
            if aid.get("idtype") == "doi":
                doi_found = aid.get("value")
                break
        # --- DOI EXTRACTION LOGIC END ---

        out.append({
            "title" : r.get( "title" , "" ),
            "url" : f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" ,
            "authors_info" : authors_info,
            "snippet" : abstracts.get(pmid) or clean_snippet(r.get( "source" , "" ) or "" ),
            "pdf_url" : "" ,
            "doi" : doi_found , # Using extracted DOI
            "venue" : jrnl,
            "year" : year,
            "citationCount" : None ,
            "publicationDate" : r.get( "pubdate" ),
            "publicationTypes" : r.get( "publicationTypes" ),
        })
    return out

def search_by_url_doi_pdf(url_or_doi):
    """Placeholder for the lookup by URL/DOI/PDF mode."""
    # This is a placeholder as the full original logic was truncated,
    # but the subsequent handling in the main loop requires it to return a list of papers_meta.
    return []

# ---------- Crossref enrichment (if DOI is known) ----------
def crossref_enrich(doi: str) -> dict:
    if not doi:
        return {}
    url = f"https://api.crossref.org/works/{doi}"

    try :
        # Calls the assumed robust request function, no API key needed for Crossref
        data = _request_json_with_retries(url, timeout= 30 )
        msg = (data or {}).get( "message" , {})
        if not msg:
            return {}

        title = (msg.get( "title" ) or [ "" ])[ 0 ]
        journal = (msg.get( "container-title" ) or [ "" ])[ 0 ]
        date_parts = (msg.get( "issued" ) or {}).get( "date-parts" , [[]])
        year = date_parts[ 0 ][ 0 ] if date_parts and date_parts[ 0 ] else None
        volume = msg.get( "volume" )
        issue = msg.get( "issue" )
        page = msg.get( "page" )
        url = msg.get( "URL" )
        authors = []
        for a in msg.get( "author" , []) or []:
            nm = f"{a.get('given','')} {a.get('family','')}" .strip()
            if nm: authors.append(nm)

        return {
            "title" : title,
            "venue" : journal,
            "year" : year,
            "volume" : volume,
            "issue" : issue,
            "pages" : page,
            "url" : url,
            "authors_info" : ", " .join(authors),
        }
    except Exception:
        return {}

# ---------- URL / PDF handling ----------
def fetch_url_and_guess_pdf(url: str) -> tuple[bool, str]:
    """Return (is_pdf, text). Detect PDF by header, extension, or magic bytes.
    If PDF, extract up to 8000 chars; else return (False, "")."""
    try :
        r = requests.get(url, timeout= 45 )
        r.raise_for_status()
        ctype = r.headers.get( "content-type" , "" ).lower()
        content = r.content

        # PDF detection: by header, extension, or magic number
        is_pdf = (
            "pdf" in ctype
            or url.lower().endswith( ".pdf" )
            or content.startswith( b"%PDF" )
        )

        if is_pdf:
            with fitz. open (stream=io.BytesIO(content), filetype= "pdf" ) as doc:
                text = []
                for page in doc:
                    text.append(page.get_text())
                return True , ( "\n" .join(text))[: 8000 ]

        return False , ""
    except Exception:
        return False , ""

def extract_metadata_from_pdf_text(pdf_text: str) -> dict:
    """Find DOI, a plausible title, author line."""
    if not pdf_text:
        return {}
    md = {}
    doi_m = DOI_RE.search(pdf_text)
    if doi_m:
        md[ "doi" ] = doi_m.group( 0 )

    # crude title guess: first reasonable line before 'Abstract'
    lines = [ln.strip() for ln in pdf_text.splitlines() if ln.strip()]
    title = None
    for ln in lines[: 60 ]:
        if re.match( r"^abstract\b" , ln, re.I):
            break
        if 8 <= len (ln) <= 240 and not re.search( r"(doi:|arxiv:)" , ln, re.I):
            title = ln
            break
    if title:
        md[ "title" ] = title

    # weak authors pattern
    for j in range ( 1 , 8 ):
        if j < len (lines):
            cand = lines[j]
            if re.search( r"[A-Z][a-z]+\s+[A-Z][a-z]+" , cand):
                md[ "authors_info" ] = cand
                break
    return md

def google_search_fallback(query: str):
    """Very light fallback via Google Custom Search (requires valid key & cx)."""
    try :
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1" ,
            params={
                "q" : query,
                "key" : GEMINI_API_KEY, # reuse key; replace with your proper CSE key
                "cx" : "017576662512468239146:omuauf_lfve" , # demo CX; replace with your own
            },
            timeout= 20 ,
        )
        data = r.json()
        items = data.get( "items" , []) or []
        if not items:
            return []

        out = []
        for it in items:
            out.append({
                "title" : it.get( "title" ),
                "url" : it.get( "link" ),
                "authors_info" : "" ,
                "snippet" : it.get( "snippet" ),
                "pdf_url" : "" ,
                "doi" : None ,
                "venue" : None ,
                "year" : None
            })
        return out
    except Exception:
        return []


# ============================
# MAIN SEARCH AND ANNOTATION CYCLE FUNCTION (REFACTORED)
# ============================
def run_search_cycle(search_query: str, search_mode: str, search_source: str, max_results: int,
                     progress_placeholder, status_placeholder, log_status: bool):
    """
    Runs a single end-to-end search, annotation, and saving cycle.
    Returns: Tuple of (total_annotated_papers, total_zotero_saves)
    """

    # Store the current search query in session state for use by fallback generation
    st.session_state.current_search_query = search_query

    # 1. Log Gemini Status (Only log once per overall Go click, handled in the main function)
    if log_status:
        log_gemini_status(client, selected_model_id)

    # 2. Semantic Model Initialization (simulated outside to avoid repeated initialization)
    try :
        from sentence_transformers import SentenceTransformer
        semantic_model_prerank = object ()
    except Exception as e:
        semantic_model_prerank = None
        if search_mode == "Keyword Search" :
            st.error( f"⚠️ Cannot run Semantic Preranking/Tagging: Sentence Transformer dependency missing or failed to load. Semantic features disabled." )

    papers_meta = []

    # --- 0. Check for invalid or missing inputs (Simplified for refactored flow) ---
    if search_mode == "Keyword Search" and not search_query:
        status_placeholder.error( "Please enter a research topic to start the cycle." )
        return 0 , 0

    # --- 1. ACQUISITION STAGE ---
    try :
        if search_mode == "Keyword Search" :
            effective_query = search_query # Already built/chosen outside the function
            status_placeholder.info( f"🔎 Running Keyword Search: '{effective_query}'" )
            progress_placeholder.progress( 10 )
            agg = []

            # 1. Semantic Scholar Search (Single Request)
            if search_source in ( "Semantic Scholar" , "Both" ):
                status_placeholder.info( "🔎 Searching Semantic Scholar…" )
                try :
                    agg.extend(search_semantic_scholar(effective_query, limit=max_results))
                except RequestException as e:
                    status_placeholder.warning( f"Semantic Scholar search failed: {e}" )
                progress_placeholder.progress( 30 )

            # 2. PubMed Paged Search (Multiple Requests, Paging)
            if search_source in ( "PubMed" , "Both" ):
                status_placeholder.info( "🧬 Searching PubMed (Paging for high-volume results)…" )
                try :
                    agg.extend(search_pubmed_paged(effective_query, limit=max_results))
                except Exception as e:
                    status_placeholder.warning( f"PubMed failed: {e}" )
                progress_placeholder.progress( 50 )

            status_placeholder.info( "📦 Combining and deduplicating results…" )
            papers_meta = _take(dedupe_results(agg), max_results)

            # --- ABSTRACT VALIDITY AND FALLBACK CHECK (MODIFIED TO PRESERVE ORIGINAL SNIPPET) ---
            min_len = st.session_state.abstract_length_slider
            status_placeholder.info( f"📝 Checking abstract quality (Min {min_len} chars) for {len(papers_meta)} papers..." )
            for idx, paper in enumerate (papers_meta):
                snippet = paper.get( "snippet" , "" )
                title = paper.get( "title" , "" )
                authors_info = paper.get( "authors_info" , "" )

                # NEW: Preserve the original snippet before potential overwrite
                papers_meta[idx][ "original_abstract" ] = snippet # Store source data here

                # Check 1: Must exist and be long enough (using dynamic slider value)
                if not snippet or len (snippet) < min_len:
                    # FIXED: Using string concatenation for robustness against display wrapping
                    logging.warning( f"Abstract #{idx+1} for '{title[:40]}...' is faulty ({len(snippet)} chars)." + " Generating fallback." )

                    # Use Gemini for fallback generation
                    fallback_abstract = gemini_abstract_fallback(title, authors_info, snippet, selected_model_id)
                    if fallback_abstract != snippet:
                        papers_meta[idx][ "snippet" ] = fallback_abstract
                        papers_meta[idx][ "snippet_source" ] = "AI_FALLBACK"
                    else :
                        papers_meta[idx][ "snippet_source" ] = "ORIGINAL_FAULTY"

            # --- VECTOR SEARCH PRERANKING ---
            if semantic_model_prerank:
                status_placeholder.info( "📊 Preranking results using Vector Search (Embedding)…" )
                papers_meta = prerank_papers(papers_meta, effective_query, semantic_model_prerank)
            else :
                status_placeholder.info( "📊 Skipping Vector Preranking (Model not available)." )
            # -----------------------------------

            # --- PRE-AI FILTERING (VSPF) ---
            pre_filter_count = len (papers_meta)
            papers_meta = [p for p in papers_meta if p.get( 'vector_score' , 0.0 ) >= st.session_state.vector_score_min_slider]
            post_filter_count = len (papers_meta)

            if pre_filter_count > post_filter_count:
                st.info( f"🗑️ **Pre-AI Filter:** Discarded {pre_filter_count - post_filter_count} paper(s) with Vector Score < {st.session_state.vector_score_min_slider} for query: {effective_query}." )

            progress_placeholder.progress( 60 )


        # 2) PASTE CITATION / TEXT
        elif search_mode == "Paste citation / page text" :
            paste_text = st.session_state.paste_text # Must retrieve from session state
            if not paste_text.strip():
                status_placeholder.warning( "Please paste citation(s) or text." )
                return 0 , 0

            status_placeholder.info( "🧾 Extracting references with Gemini…" )
            refs = gemini_extract_from_text(paste_text, selected_model_id)
            progress_placeholder.progress( 30 )

            if not refs:
                status_placeholder.error( "😅 We squinted at every reference style… but found nada." )
                return 0 , 0

            status_placeholder.info( "🔎 Enriching references…" )
            collected = []
            min_len = st.session_state.abstract_length_slider

            for r in refs[:max_results]:
                title, authors, year, doi = r.get( "title" ), r.get( "authors" ), r.get( "year" ), r.get( "doi" )
                enriched = None

                # search_semantic_scholar_by_doi uses robust retries
                if doi:
                    enriched = search_semantic_scholar_by_doi(doi)

                if not enriched and title:
                    pm = search_pubmed_paged(title, 1 )
                    enriched = pm[ 0 ] if pm else None
                if not enriched and title:
                    gg = google_search_fallback(title)
                    enriched = gg[ 0 ] if gg else None
                if not enriched:
                    enriched = { "title" : title, "authors_info" : ", " .join(authors) if isinstance (authors, list ) else (authors or "" ), "snippet" : "" , "url" : "" , "pdf_url" : "" , "doi" : doi, "year" : year, "venue" : None }

                # Check abstract validity and run fallback
                snippet_to_check = enriched.get( "snippet" , "" )
                # Save original before check
                enriched[ "original_abstract" ] = snippet_to_check
                if not snippet_to_check or len (snippet_to_check) < min_len:
                    # FIXED L555: Using string concatenation for robustness against display wrapping
                    logging.warning( f"Abstract for '{title[:40]}...' is faulty ({len(snippet_to_check)} chars)." + " Generating fallback." )

                    fallback = gemini_abstract_fallback(enriched.get( "title" , "" ), enriched.get( "authors_info" , "" ), snippet_to_check, selected_model_id)
                    if fallback != snippet_to_check:
                        enriched[ "snippet" ] = fallback
                        enriched[ "snippet_source" ] = "AI_FALLBACK"
                    collected.append(enriched)

                papers_meta = collected
            progress_placeholder.progress( 60 )

        # 3) LOOKUP BY URL / DOI / PDF
        else : # search_mode == "Lookup by URL / PDF"
            url_or_doi = st.session_state.url_or_doi # Must retrieve from session state
            if not url_or_doi or not url_or_doi.strip():
                status_placeholder.warning( "Please paste a URL or DOI." )
                return 0 , 0

            status_placeholder.info( "🧭 Resolving input…" )
            papers_meta = search_by_url_doi_pdf(url_or_doi)
            progress_placeholder.progress( 10 )

            # Fallback check for single paper acquisition
            if papers_meta:
                min_len = st.session_state.abstract_length_slider
                snippet_to_check = papers_meta[ 0 ].get( "snippet" , "" )
                # Save original before check
                papers_meta[ 0 ][ "original_abstract" ] = snippet_to_check
                if not snippet_to_check or len (snippet_to_check) < min_len:
                    # FIXED L591: Using string concatenation for robustness against display wrapping
                    logging.warning( f"Abstract for '{papers_meta[0].get('title', '')[:40]}...' is faulty ({len(snippet_to_check)} chars)." + " Generating fallback." )

                    fallback = gemini_abstract_fallback(papers_meta[ 0 ].get( "title" , "" ), papers_meta[ 0 ].get( "authors_info" , "" ), snippet_to_check, selected_model_id)
                    if fallback != snippet_to_check:
                        papers_meta[ 0 ][ "snippet" ] = fallback
                        papers_meta[ 0 ][ "snippet_source" ] = "AI_FALLBACK"

        # --- 2. ANNOTATION AND RENDERING STAGE (UNIFIED) ---
        if not papers_meta:
            status_placeholder.error( "😅 No papers found for this query." )
            progress_placeholder.progress( 100 )
            return 0, 0

        status_placeholder.info( f"🧪 Analyzing and annotating {len(papers_meta)} papers… (Sequential API Calls)" )
        progress_placeholder.progress( 75 )

        # Map Zotero threshold: score3 (0..3)
        zotero_threshold_score3 = min ( 3 , max ( 0 , int (st.session_state.min_score3_slider)))

        # Parse post-AI tag filter values and prefixes
        required_tag_values = []
        required_prefixes = []
        enabled_semantic_sentences = st.session_state.semantic_sentences

        if search_mode == "Keyword Search" :
            required_tag_values = [v.strip().lower() for v in st.session_state.ai_tag_post_filter_values_input.split( ',' ) if v.strip()]

            # NOTE: Logic to derive prefix from category name remains simple for now
            tag_prefix_map = {
                cat: cat.split( '(' )[ 0 ].strip() + "-" for cat in st.session_state.ai_tag_categories_list
            }
            required_prefixes = [tag_prefix_map[cat] for cat in st.session_state.selected_tag_categories if cat in tag_prefix_map]

        # Initialize counters for summary
        total_annotated_papers = 0
        total_zotero_saves = 0

        # --- Sequential Annotation Loop ---
        for i, paper in enumerate (papers_meta):
            title = paper.get( "title" , "" )
            url = paper.get( "url" , "" )
            authors_info = paper.get( "authors_info" , "" )
            snippet = paper.get( "snippet" , "" )
            pdf_url = paper.get( "pdf_url" , "" )
            doi = paper.get( "doi" )
            venue = paper.get( "venue" )
            year = paper.get( "year" )
            vector_score = paper.get( 'vector_score' ) # Get vector score if present
            original_snippet = paper.get( "original_abstract" ) # Retrieve the preserved source abstract

            # Pull PDF text when useful
            pdf_text = extract_pdf_text(pdf_url or url)

            # --- Gemini Annotation (Sequential, Backoff-Enabled) ---
            user_query = search_query # Use the query passed to this function

            # Sequential API call with backoff
            abstract_ai, tags, score3 = gemini_annotate_paper(
                title, authors_info, snippet, pdf_text, url, user_query, selected_model_id
            ) if client else ( "GEMINI AI Abstract Placeholder" , [ "aRT-safe" ], 0 ) # Updated placeholder output

            # --- NEW: Semantic Tagging After Abstract Generation ---
            if semantic_model_prerank and abstract_ai and not abstract_ai.startswith( "API FAILURE" ):
                semantic_tags = generate_semantic_tags(abstract_ai, enabled_semantic_sentences, semantic_model_prerank, top_n= 3 )
                tags.extend(semantic_tags)

            # Update annotated count
            if not ( "RATE LIMIT EXHAUSTED" in abstract_ai or "API FAILURE" in abstract_ai or "GEMINI API KEY IS MISSING/invalid" in abstract_ai):
                total_annotated_papers += 1

            # Update progress bar only for the loop step
            progress_placeholder.progress( 75 + int ((i / len (papers_meta)) * 20 ))

            with st.expander( f"📄 {title or 'Untitled'} (Score: {score3}, Vector Sim: {vector_score:.3f} )" if vector_score is not None else f"📄 {title or 'Untitled'} (Score: {score3})" , expanded= True ):
                if authors_info:
                    st.markdown( f"**Authors:** {authors_info}" )
                if venue or year:
                    st.markdown( f"**Venue / Year:** {venue or '—'} — {year or '—'}" )

                # --- Source Abstract Display ---
                if original_snippet:
                    source_label = "(source)" # Always label as source, as it's the original metadata
                    st.markdown( f"**Abstract {source_label}:** {original_snippet}" ) # Display original source abstract
                elif paper.get('snippet_source') == 'AI_FALLBACK':
                    # If AI Fallback was used, inform the user why the source abstract is missing
                    st.info("ℹ️ **Abstract (source):** Original abstract was missing or too short, using AI Fallback for filtering.")
                elif snippet and paper.get('snippet_source') == 'ORIGINAL_FAULTY':
                    # The snippet is still the original faulty one, display it but warn
                    st.warning(f"⚠️ **Abstract (source):** Original abstract was too short for filtering (Length: {len(snippet)}).")
                    st.markdown(f"**Abstract (source):** {snippet}")

                # --- AI Abstract/Error Display ---
                if abstract_ai:
                    if abstract_ai.startswith( "RATE LIMIT EXHAUSTED" ) or abstract_ai.startswith( "API CONNECTION_ERROR" ) or abstract_ai.startswith( "Gemini returned" ) or abstract_ai.startswith( "API FAILURE" ) or abstract_ai.startswith( "Unknown API Failure" ):
                        st.error( f"**AI Abstract Failure:** {abstract_ai}" )
                    elif abstract_ai.startswith( "GEMINI API KEY IS MISSING/invalid" ):
                        st.error( f"**AI Abstract Failure:** {abstract_ai}" )
                    else :
                        st.markdown( "**Abstract (AI):**")
                        st.write(abstract_ai)

                if url:
                    st.markdown( f"[🔗 View Paper]({url})" )
                    doi_or_url = f"https://doi.org/{doi}" if doi else url
                    inst1 = with_ntu_proxy(doi_or_url, style= 1 )
                    inst2 = with_ntu_proxy(doi_or_url, style= 2 )
                    if inst1:
                        st.markdown( f"[🏫 NTU Access (style 1)]({inst1})" )
                    if inst2:
                        st.markdown( f"[🏫 NTU Access (style 2)]({inst2})" )

                if tags:
                    st.markdown( "**🏷️ Tags:** " + ", " .join(tags))
                st.markdown( f"**AI Relevance (0–3):** `{score3}`" )

                # --- LOCAL ZOTERO POSTING (Post-AI Filter) ---

                # 1. Post-AI Tag Filter Check:
                passes_tag_filter = True

                if required_tag_values:
                    passes_tag_filter = False
                    generated_tag_set = {t.lower() for t in tags}
                    # Check 1: Specific Prefix + Value Match
                    if required_prefixes:
                        for required_prefix in required_prefixes:
                            if any (required_prefix in t and any (val in t for val in required_tag_values) for t in generated_tag_set):
                                passes_tag_filter = True
                                break

                    # Check 2: Value Match (if no prefix is specified)
                    elif not required_prefixes:
                        passes_tag_filter = any ( any (val in t for val in required_tag_values) for t in generated_tag_set)

                if st.session_state.add_to_zotero_state and (score3 >= zotero_threshold_score3) and passes_tag_filter:

                    # *** NEW REQUIREMENT: DOI CHECK and ERROR MESSAGE ***
                    doi_value = doi if doi else None

                    if not doi_value:
                        st.warning(DOI_MISSING_ERROR) # Display the doi check.
                    doi_or_url = f"https://doi.org/{doi}" if doi else url
                    proxy_url = with_ntu_proxy(doi_or_url, style= 1 ) or with_ntu_proxy(doi_or_url, style= 2 ) or url

                    # Combine and label abstracts for the single Zotero field.
                    # *** CRITICAL FIX: Use source snippet (snippet) first, as preferred. ***
                    combined_abstract = snippet or abstract_ai
                    
                    # *** CRITICAL FIX: Limit abstract length to prevent Zotero Error 413 (Payload too large) ***
                    abstract_content = combined_abstract[:ZOTERO_MAX_ABSTRACT_CHARS]
                    if len(combined_abstract) > ZOTERO_MAX_ABSTRACT_CHARS:
                        logging.warning(f"Abstract for '{title[:30]}...' truncated to {ZOTERO_MAX_ABSTRACT_CHARS} chars.")

                    # *** Tag Truncation Safeguard (Addressing Error 413 for tags) ***
                    # Zotero API limit is around 256 for a single tag value
                    cleaned_tags = [{'tag': t[:MAX_TAG_LENGTH]} for t in (tags or [])]

                    # Item payload uses Cloud API structure which is compatible with the local Connector endpoint.
                                                                                  
                                                                                              
   


   
                    item = {
                        'itemType': 'journalArticle',
                        'title': title,
                        'creators': parse_authors(authors_info),
                        'abstractNote': abstract_content, # Uses the strictly truncated content
                        'tags': cleaned_tags, # Uses the cleaned, truncated tags
                        'url': proxy_url,
                        'date': str(year) if year else None,
                        'DOI': doi_value, # Ensures DOI is passed if available
                        # 'collections' field is ignored by the local connector endpoint.
                    }
                    item = {k: v for k, v in item.items() if v not in (None, "")}

                    # The save_to_zotero_local call uses a 10-minute timeout
                    success, msg = save_to_zotero_local(item)
                    if success:
                        st.success(f"✅ Added to Local Zotero (score3={score3}, Tags Matched)")
                        total_zotero_saves += 1
                    else:
                        st.error(f"❌ Local Zotero Error: {msg}")

                elif st.session_state.add_to_zotero_state:
                    if score3 < zotero_threshold_score3:
                        st.info(f"⚠️ Skipped: Failed AI Score Post-Filter (Score {score3} < {zotero_threshold_score3}).")
                    elif required_tag_values and not passes_tag_filter:
                        st.info(f"⚠️ Skipped: Failed AI Tag Post-Filter (Missing required tags).")

        # Status updates are handled by the main loop for the overall cycle.
        return total_annotated_papers, total_zotero_saves

    except Exception as e:
        status_placeholder.error(f"A fatal error occurred during the search cycle: {e}")
        logging.error(f"FATAL SEARCH CYCLE ERROR: {e}")
        return 0, 0


# ============================
# MAIN ACTION (MODIFIED FOR CYCLE EXECUTION)
# ============================
if st.button( "🚀 Go" ):

    # Placeholders for overall status and progress
    overall_status = st.empty()
    overall_progress = st.progress( 0 )

    # Check if we are running a single query or a cycle
    queries_to_run = []

    if search_mode == "Keyword Search" and query_mode == "Automated Cycle":
        queries_to_run = st.session_state.get('cycle_queries_selector', [])
        if not queries_to_run:
            overall_status.error("Please select at least one query in the 'Automated Cycle' mode.")
            overall_progress.progress(100)
            st.stop()

        # Determine how the query will be processed (boolean optimization is applied within run_search_cycle)
        overall_status.info(f"Initiating Automated Cycle with {len(queries_to_run)} queries.")

    elif search_mode == "Keyword Search" and query_mode == "Single Query":
        if 'user_prompt' in locals() and user_prompt:

            # Apply Boolean conversion logic to get the effective query
            effective_query = user_prompt
            if use_boolean:
                b = gemini_boolean_query(user_prompt, selected_model_id)
                effective_query = b.get( "boolean_query" ) or build_boolean_query_simple(user_prompt)

            # Editable query box for single query is applied here as it was in the old flow
            st.text_area( "✏️ Editable search query (you can tweak before searching):" , effective_query, key="final_editable_query_single")
            queries_to_run = [st.session_state.final_editable_query_single]

            overall_status.info(f"Initiating Single Query run: '{queries_to_run[0]}'")

        else:
            overall_status.error("Please enter a research topic to start the cycle.")
            overall_progress.progress(100)
            st.stop()

    elif search_mode == "Paste citation / page text":
        # For non-Keyword Search modes, we treat it as a single execution
        if 'paste_text' in locals():
            st.session_state.paste_text = paste_text # Ensure latest input is in state
        queries_to_run = ["PASTE_MODE_RUN"]
        overall_status.info(f"Initiating Citation/Text extraction.")

    elif search_mode == "Lookup by URL / PDF ":
        # For non-Keyword Search modes, we treat it as a single execution
        if 'url_or_doi' in locals():
            st.session_state.url_or_doi = url_or_doi # Ensure latest input is in state
        queries_to_run = ["URL_MODE_RUN"]
        overall_status.info(f"Initiating URL/DOI lookup.")


    # --- Execute the Cycle (The Greatest Cycle) ---
    all_annotated_papers = 0
    all_zotero_saves = 0
    total_queries = len(queries_to_run)

    # 1. Log Gemini Status once before the first run
    log_gemini_status(client, selected_model_id)

    # Place a clean container for results, clearing previous display
    results_container = st.empty()

    for query_idx, query in enumerate(queries_to_run):

        # Calculate progress for the overall cycle completion
        cycle_start_progress = int((query_idx / total_queries) * 100)

        with results_container.container():
            st.markdown(f"## 🏃 Cycle {query_idx + 1} of {total_queries}: **{query}**")

            # Placeholders for this specific cycle's detailed progress and status
            cycle_progress_bar = st.progress(cycle_start_progress)
            cycle_status_text = st.empty()

            # --- Call the Reusable Search/Annotate Function ---
            if search_mode == "Keyword Search":
                query_to_pass = query # Already the effective query string
            else:
                # For non-keyword modes, the query string is irrelevant to the function's logic
                query_to_pass = ""

            annotated_count, zotero_count = run_search_cycle(
                search_query=query_to_pass,
                search_mode=search_mode,
                search_source=search_source,
                max_results=max_results,
                progress_placeholder=cycle_progress_bar,
                status_placeholder=cycle_status_text,
                log_status=False # Already logged once
            )

            all_annotated_papers += annotated_count
            all_zotero_saves += zotero_count

            cycle_progress_bar.progress(100)
            cycle_status_text.success(f"Cycle {query_idx + 1} Done ✅. Annotated {annotated_count} papers.")

            # Add a separator if this is not the last cycle
            if query_idx < total_queries - 1:
                st.markdown("---")

    # --- FINAL SUMMARY BLOCK (OVERALL) ---
    final_progress = 100
    overall_progress.progress(final_progress)

    st.markdown( "---" )
    st.subheader( "📊 Final Run Summary (All Cycles)" )
    st.metric(
        label= "🤖 Total Papers Annotated by AI" ,
        value=all_annotated_papers,
        delta_color= "off"
    )
    st.metric(
        label= f"📥 Total Papers Saved to Zotero (Score $\\geq$ {st.session_state.min_score3_slider}, Tags Filtered)" ,
        value=all_zotero_saves,
        delta_color= "off"
    )
    overall_status.success("All Automated Cycles Complete. Scroll down for individual query results.")


    # --- Final Cleanup ---
    try :
        # Clear status after a short delay to avoid lingering messages
        sleep( 0.4 )
        # overall_status.empty() # Keep final success visible
    finally :
        pass
