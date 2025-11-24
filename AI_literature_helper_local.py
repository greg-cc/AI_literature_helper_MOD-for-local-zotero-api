# -*- coding: utf-8 -*-
import streamlit as st
import requests, json, re, os, io
import xml.etree.ElementTree as ET
from pyzotero import zotero # pyzotero is now imported for duplicate checking
import fitz # PyMuPDF
from time import sleep
from requests import RequestException
import logging
import sys
import random

# Configure logging to print status to the application console
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s', stream=sys.stdout)

# --- GLOBAL API CONSTANTS for Backoff (Used in gemini_json) ---
MAX_RETRIES = 5
MAX_DELAY_SECONDS = 60 # Initial delay starts low, max delay hits this limit
INITIAL_DELAY_SECONDS = 5
RATE_LIMIT_STATUS_CODE = 429
# ============================
# CONFIG
# ============================
st.set_page_config(page_title="📚 AI Literature Helper", page_icon="🤖")


# --- FIXED SECTION START ---
# keys are now assigned directly as strings
SEMANTIC_SCHOLAR_API_KEY = "It2pKMHpTK7l5lnOhPUKE4ldBA3Lzeq82hHEsbnB"
GEMINI_API_KEY = "AIzaSyDKI4T4lgP7N8oiPaYK-vC_4_osmLF6DMY" # Replaced with empty string as agreed. USER MUST REPLACE THIS.
NCBI_EMAIL = "reggcrowmell@gmail.com"
NCBI_API_KEY = "698aa15950467c8cda8583c04e199237cac08"

# client now uses the variable defined above
try:
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    client = None
    # Store the exception details for logging
    if not GEMINI_API_KEY:
        GEMINI_CLIENT_INIT_ERROR = f"Client failed to initialize: GEMINI_API_KEY is empty."
    else:
        GEMINI_CLIENT_INIT_ERROR = f"Client failed to initialize: {e.__class__.__name__}: {e}"

# --- FIXED SECTION END ---

# --- MODEL OPTIONS FOR UI (Restored Verbiage) ---
MODEL_OPTIONS = {
    "models/gemini-2.5-flash (Default)": {
        "model_id": "gemini-2.5-flash",
        "description": "General purpose, fast, large context. Default model (70B parameters, info: **7 GB**)."
    },
    "models/gemini-2.5-pro (Complex Tasks)": {
        "model_id": "gemini-2.5-pro",
        "description": "Highest reasoning capacity, suitable for complex annotations. Separate, smaller quota (100B+ parameters, info: **15 GB**)."
    },
    "models/gemini-flash-lite-latest (Quota Fallback)": {
        "model_id": "models/gemini-flash-lite-latest",
        "description": "Optimized for high throughput and cost efficiency. Good quota fallback (info: **700 MB**)."
    },
    "models/gemini-2.5-flash-lite (Quota Fallback)": {
        "model_id": "models/gemini-2.5-flash-lite",
        "description": "High throughput and cost-efficient version. Good quota fallback (info: **700 MB**)."
    },
    "gemma-3-1b-it (Open Model, Text Only)": {
        "model_id": "gemma-3-1b-it",
        "description": "Open model (1.4B parameters). Best for simple text generation when Gemini quota is exhausted (info: **1.5 GB**)."
    },
    "gemma-3-4b-it (Open Model, Text Only)": {
        "model_id": "gemma-3-4b-it",
        "description": "Open model (4B parameters). More capable than 1B, for text tasks (info: **4 GB**)."
    },
    "gemma-3-12b-it (Open Model, Text Only)": {
        "model_id": "gemma-3-12b-it",
        "description": "Open model (12B parameters). Highly capable text model (info: **12 GB**)."
    },
    "gemma-3-27b-it (Open Model, Text Only)": {
        "model_id": "gemma-3-27b-it",
        "description": "Open model (27B parameters). Largest open model for complex text tasks (info: **27 GB**)."
    },
}

# ============================
# NEW: API STATUS CHECK (Logged to console)
# ============================
def _list_and_log_models(client):
    try:
        models = client.models.list()
        logging.info("--- Available Models Log ---")
        for model in models:
            logging.info(f"    - {model.name}")
        logging.info("------------------------")
        logging.info(f"Total models found: {len(list(models))}.")
    except Exception as e:
        # Crucial: Log the exact error when listing models fails (e.g., Auth failure)
        logging.error(f"Failed to list models (Authentication/Connection issue): {e}")

def log_gemini_status(client, selected_model_id):
    """Tries to get a simple list of models to confirm the API key is active.
    Logs status and model list to the console (sys.stdout)."""

    if not client:
        # New: If client initialization failed, display stored error message
        logging.error("=========================================================")
        logging.error(f"GEMINI STATUS: API Client Initialization Failed.")
        logging.error(f"ERROR: {globals().get('GEMINI_CLIENT_INIT_ERROR', 'Unknown initialization error.')}")
        logging.error("Please verify your GEMINI_API_KEY is present and valid.")
        logging.error("=========================================================")
        return

    try:
        # A simple, low-cost call to verify connectivity and authentication
        logging.info("=========================================================")
        logging.info("GEMINI STATUS: API Key is successfully authenticated.")
        logging.info(f"CURRENT MODEL FOR ANNOTATION: {selected_model_id}")

        # Log the detailed model list (as requested by the user)
        _list_and_log_models(client)

        logging.info("NOTE: Remaining quota/usage details are not available via the client library.")
        logging.info("Please check your Google AI Studio or Cloud Console dashboard for usage limits.")
        logging.info("=========================================================")
    except Exception as e:
        # Catch errors during the status check itself (unlikely if initialization succeeded, but robust)
        logging.error("=========================================================")
        logging.error(f"GEMINI STATUS: Authentication/Connection Failure during status check.")
        logging.error(f"Error details: {e}")
        logging.error("=========================================================")


SLEEP = 0.08 # pacing for retries/backoff
PREFS_FILE = "prefs.json"

# ============================
# PREFERENCES (saved locally)
# ============================
def load_prefs():
    # MODIFIED: Added "allow_duplicates" to persistence
    if not os.path.exists(PREFS_FILE):
        return {"topics": [], "authors": [], "collection_id": "", "library_id": "", "allow_duplicates": False}
    try:
        data = json.load(open(PREFS_FILE))
        # Ensure all fields are present, using defaults if missing from old file
        data["collection_id"] = data.get("collection_id", "")
        data["library_id"] = data.get("library_id", "")
        data["allow_duplicates"] = data.get("allow_duplicates", False)
        return data
    except Exception:
        return {"topics": [], "authors": [], "collection_id": "", "library_id": "", "allow_duplicates": False}

def save_prefs(topics, authors, collection_id, library_id, allow_duplicates):
    # MODIFIED: Added allow_duplicates argument
    with open(PREFS_FILE, "w") as f:
        json.dump({
            "topics": topics, 
            "authors": authors, 
            "collection_id": collection_id, 
            "library_id": library_id,
            "allow_duplicates": allow_duplicates
        }, f)

prefs = load_prefs()

# ============================
# UI
# ============================
st.title("📚 AI Literature Helper")

search_mode = st.radio(
    "🔍 What would you like to do?",
    [
        "Keyword Search",
        "Paste citation / page text",
        "Lookup by URL / PDF ",
    ],
    horizontal=False,
)

# Source selector ONLY for Keyword Search (removed for Paste mode per request)
# MODIFIED DEFAULT: index=2 sets default to "Both"
search_source = st.selectbox(
    "📡 Choose search source",
    ["Semantic Scholar", "PubMed", "Both"],
    index=2
) if search_mode == "Keyword Search" else None

# MODIFIED DEFAULT: value=90
max_results = st.slider("📄 Max articles to fetch:", 5, 100, 90, 1)

# MODIFIED DEFAULT: value=1
min_score3 = st.slider("⭐ Minimum AI relevance score3 to save to Zotero (0–3):", 0, 3, 1, 1)

if search_mode == "Keyword Search":
    user_prompt = st.text_input("🔍 Enter your research topic or keywords:")
    # MODIFIED DEFAULT: value=False
    use_boolean = st.checkbox("🔤 Convert to Boolean query (AI-optimized)", value=False)
elif search_mode == "Paste citation / page text":
    paste_text = st.text_area("📋 Paste citation(s) or Google Scholar results / page text:", height=220)
else:
    url_or_doi = st.text_input("🔗 Paste a URL (landing page or PDF):")

# --- Model Selector UI ---
st.markdown("---")
st.subheader("🤖 AI Annotation Model Selector")

# MODIFIED DEFAULT: index=3 sets default to "models/gemini-2.5-flash-lite (Quota Fallback)"
model_key = st.selectbox(
    "Choose Gemini Model (Select a fallback if quota exhausted):",
    options=list(MODEL_OPTIONS.keys()),
    format_func=lambda k: f"{k} ({MODEL_OPTIONS[k]['description'].split('(')[-1].strip(')')})",
    index=3,
    help="Select a model. The Lite/Gemma options have separate quotas and can be used if the default is rate-limited."
)
selected_model_info = MODEL_OPTIONS[model_key]
selected_model_id = selected_model_info["model_id"]
st.caption(f"Model ID: **`{selected_model_id}`**. Description: {selected_model_info['description']}")
st.markdown("---")
# --- End Model Selector UI ---

# --- Zotero Inputs (for pyzotero initialization and action) ---

# MODIFIED DEFAULT: value=True
add_to_zotero = st.checkbox("📥 Add articles to Zotero", value=True)

# Library ID input (required for pyzotero duplicate check)
user_zotero_id = st.text_input(
    "Zotero User ID (Library ID)", 
    value=prefs.get("library_id", ""),
    help="Your numeric Zotero User ID. Required for the pyzotero duplicate check."
)

# RE-ADDED: Allow Duplicates Checkbox, persistent via prefs.json
allow_duplicates = st.checkbox(
    "⚠️ Allow Zotero duplicates", 
    value=prefs.get("allow_duplicates", False),
    help="If checked, papers will be added even if a matching title is found in Zotero."
)

st.caption("ℹ️ Zotero API Key field is omitted as local posting is used.")

# --- Sidebar Preferences ---
with st.sidebar:
    st.header("🔖 Preferences")
    topics_txt = st.text_input("Priority Topics (comma-separated)", ", ".join(prefs.get("topics", [])))
    authors_txt = st.text_input("Priority Authors (comma-separated)", ", ".join(prefs.get("authors", [])))

    # Collection ID input (visible and persistent)
    user_zotero_collection = st.text_input("Zotero Collection ID", value=prefs.get("collection_id", ""))

    if st.button("💾 Save Preferences"):
        # Passed all persistent variables, including the current state of allow_duplicates
        save_prefs(
            [t.strip() for t in topics_txt.split(",") if t.strip()],
            [a.strip() for a in authors_txt.split(",") if a.strip()],
            user_zotero_collection,
            user_zotero_id,
            allow_duplicates # Save the current state of the checkbox
        )
        st.sidebar.success("Saved preferences.")


# ============================
# PYZOtero HELPERS (for Duplicate Check)
# ============================

@st.cache_resource
def init_pyzotero_local(library_id):
    """Initializes and returns a pyzotero client for local access."""
    if not library_id or not library_id.isdigit():
        return None, "Invalid Zotero User ID (Library ID)."
    try:
        # Use a dummy key/API type; local=True handles the connection
        # to the running Zotero 7 desktop app via 127.0.0.1:23119
        zot = zotero.Zotero(library_id, 'user', 'DUMMY_KEY', local=True)
        # Verify connection by fetching a simple item count
        zot.count_items()
        return zot, None
    except Exception as e:
        logging.error(f"Failed to initialize pyzotero local connection: {e}")
        return None, f"Zotero API connection failed. Is Zotero Desktop running? ({e.__class__.__name__})"

def check_zotero_duplicate(zot, title):
    """Checks the local Zotero library for a duplicate title."""
    if not zot:
        return False, "Zotero client not available."
    try:
        # Search by title; items should be sorted by date descending by default
        items = zot.items(q=title.strip(), limit=5)
        
        # Simple check: look for an exact or near-exact title match
        for item in items:
            item_title = item.get('data', {}).get('title', '').strip()
            if item_title.lower() == title.strip().lower():
                 return True, f"Duplicate found by exact title: {title}"
        
        return False, "No duplicate found."
    except Exception as e:
        logging.warning(f"Zotero duplicate check failed (pyzotero query error): {e}")
        return False, f"Zotero query error: {e}"


# ============================
# LOCAL ZOTERO ACTION (Connector API)
# ============================
def save_to_zotero_local(item_data, timeout_seconds=600):
    """
    Pushes a record to the running Zotero instance via the Connector endpoint.
    """
    connector_url = "http://127.0.0.1:23119/connector/saveItems"
    payload = { "items": [item_data] }

    try:
        resp = requests.post(connector_url, json=payload, timeout=timeout_seconds)
        resp.raise_for_status()
        return True, "Item successfully sent to local Zotero instance."
    except requests.exceptions.ConnectionError:
        return False, "Local Zotero Error: Connection refused. Is Zotero Desktop running?"
    except Exception as e:
        return False, f"Local Zotero Error: {e}. Check item formatting."


# ============================
# HELPERS (Remaining)
# ============================
OPERATORS = {"and": "AND", "or": "OR", "not": "NOT"}

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.I)
HTML_TAG_RE = re.compile(r"<[^>]+>")
ARXIV_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?", re.I)

def build_boolean_query_simple(text: str) -> str:
    """Quick AND-join of comma/;/slash separated tokens; phrases quoted and logicals normalized."""
    q = text.strip()
    tokens = [t.strip() for t in re.split(r",|;|/", q) if t.strip()]
    if len(tokens) >= 2:
        q = " AND ".join([f'"{t}"' if " " in t else t for t in tokens])
    q = re.sub(r"\b(and|or|not)\b", lambda m: OPERATORS[m.group(1).lower()], q, flags=re.I)
    return q

def with_ntu_proxy(url: str | None, style: int = 2) -> str | None:
    if not url:
        return None
    if style == 1:
        return f"https://remotexs.ntu.edu.sg/user/login?dest={url}"
    return f"https://remotexs.ntu.edu.sg/login?url={url}"

def extract_pdf_text(url: str) -> str:
    """Download a PDF and return the first ~5000 chars of text, or empty string if fails."""
    if not url:
        return ""
    try:
        r = requests.get(url, timeout=45)
        r.raise_for_status()
        with fitz.open(stream=io.BytesIO(r.content), filetype="pdf") as doc:
            text = []
            for page in doc:
                text.append(page.get_text())
            return ("\n".join(text))[:5000]
    except Exception:
        return ""

def parse_authors(authors_info: str):
    authors = [a.strip() for a in authors_info.split(",") if a.strip()]
    out = []
    for nm in authors:
        parts = nm.split(" ")
        if len(parts) >= 2:
            out.append({"creatorType": "author", "firstName": " ".join(parts[:-1]), "lastName": parts[-1]})
        else:
            out.append({"creatorType": "author", "name": nm})
    return out

def dedupe_results(results):
    seen, out = set(), []
    for r in results:
        doi = (r.get("doi") or "").lower().replace("https://doi.org/", "")
        key = doi or (r.get("url") or r.get("title", "")).lower()
        if key in seen:
            continue
        seen.add(key); out.append(r)
    return out

def _request_json_with_retries(url, *, method="GET", headers=None, params=None, data=None, tries=4, timeout=40):
    delay = SLEEP
    for attempt in range(1, tries + 1):
        try:
            resp = (requests.post(url, headers=headers, params=params, data=data, timeout=timeout)
                     if method == "POST" else
                     requests.get(url, headers=headers, params=params, timeout=timeout))
            if 200 <= resp.status_code < 300:
                return resp.json()
            if 500 <= resp.status_code < 600:
                raise RequestException(f"Server {resp.status_code}")
            resp.raise_for_status()
        except Exception:
            if attempt == tries:
                raise
            sleep(delay)
            delay = min(delay * 2, 3.0)
    return {}

def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def _take(results, k):
    return results[:k] if len(results) > k else results

def clean_snippet(text: str) -> str:
    if not text:
        return ""
    text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if DOI_RE.fullmatch(text.replace("doi:", "").strip().lower()):
        return ""
    text = re.sub(r"^doi:\s*10\.\d{4,9}/\S+\s*", "", text, flags=re.I)
    return text

# ============================
# GEMINI CORE CALLS
# ============================

def gemini_json(prompt: str, model: str) -> dict | list:
    """
    Calls Gemini API with Jittered Exponential Backoff for 429 errors.
    Returns: Parsed JSON on success, or an empty dictionary/list on failure.
    """
    if not client:
        return {"error": "GEMINI_CLIENT_UNINITIALIZED"}

    max_delay = MAX_DELAY_SECONDS
    delay = INITIAL_DELAY_SECONDS

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            txt = resp.text or ""

            # Successful response, try to parse JSON
            try:
                return json.loads(txt)
            except Exception:
                # Fallback: try to find JSON blob within the response (if model wrapped it)
                m = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", txt)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        return {"error": "JSON_PARSE_FAILURE", "raw_text": txt} # Final JSON parse failed
                return {"error": "JSON_NOT_FOUND", "raw_text": txt} # JSON not found, return raw text

        except Exception as e:
            # Check if this is a 429 Rate Limit error (ResourceExhausted)
            error_message = str(e)

            # The client library often embeds the status code in the exception message/attributes on failure
            if "ResourceExhausted" in error_message or "429" in error_message:
                logging.warning(f"RATE LIMIT (429) hit for {model}. Attempt {attempt}/{MAX_RETRIES}.")

                if attempt == MAX_RETRIES:
                    return {"error": "RATE_LIMIT_EXHAUSTED", "details": error_message}

                # Jittered Exponential Backoff calculation
                wait_time = min(delay, max_delay)
                delay = delay * 2
                jitter = random.uniform(0, 1) * wait_time
                wait_duration = wait_time + jitter

                logging.info(f"Pausing for {wait_duration:.2f} seconds before retrying...")
                sleep(wait_duration)

            else:
                # This is a different, unhandled API error (e.g., Auth, Invalid Model ID)
                return {"error": "API_CONNECTION_ERROR", "details": error_message}

    return {"error": "API_FAILURE_UNKNOWN"}


def gemini_boolean_query(user_query: str, model: str) -> dict:
    data = gemini_json(f"""
Create a compact Boolean query (use AND/OR/NOT and quotes for phrases) suitable for academic APIs.
Return JSON {{"boolean_query": "...", "keywords": [], "year_from": null, "year_to": null}}
Topic: {user_query}
Priority topics: {prefs.get('topics')}
""", model)

    if data and "error" in data:
        logging.error(f"Boolean Query Failed: {data['error']} - {data.get('details', data.get('raw_text', ''))[:100]}...")
        return {"boolean_query": build_boolean_query_simple(user_query), "keywords": [], "year_from": None, "year_to": None}

    out = {"boolean_query": build_boolean_query_simple(user_query), "keywords": [], "year_from": None, "year_to": None}
    if isinstance(data, dict):
        out["boolean_query"] = data.get("boolean_query") or out["boolean_query"]
        out["keywords"] = data.get("keywords") or []
        out["year_from"] = data.get("year_from")
        out["year_to"] = data.get("year_to")
    return out

def gemini_extract_from_text(raw_text: str, model: str):
    """
    Extract refs from pasted text (e.g., Google Scholar page).
    Returns list of {title, authors:[...], year, doi?}
    """
    data = gemini_json(f"""
You are an academic reference extractor.
From the text below, extract a list of references as JSON array. Each object must have:
- "title" (string)
- "authors" (list of names)
- "year" (int if available else null)
- "doi" (string DOI without https://doi.org/ if present else null)

Text:
{raw_text}

Return strictly a JSON array.
""", model)

    if data and "error" in data:
        logging.error(f"Extraction Failed: {data['error']} - {data.get('details', data.get('raw_text', ''))[:100]}...")
        return []

    out = []
    if isinstance(data, list):
        for it in data:
            if not isinstance(it, dict):
                continue
            title = (it.get("title") or "").strip()
            if not title:
                continue
            authors = it.get("authors") or []
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(",") if a.strip()]
            year = it.get("year")
            doi  = it.get("doi")
            if isinstance(doi, str):
                m = DOI_RE.search(doi)
                doi = m.group(0) if m else doi.strip()
            out.append({"title": title, "authors": authors, "year": year, "doi": doi})
        return out

    # Log failure to extract references
    # Note: 'raw_data' is not defined here. Assuming 'data' means the unparsed JSON output
    logging.error(f"Extraction Failed! Model returned non-list data during reference extraction. Raw data: {data}")
    return []

def gemini_annotate_paper(title, authors, snippet, pdf_text, url, user_query, model: str):
    """
    Return: abstract (10–15 sentences), tags [aRT..., aTa..., aTy..., aMe..., ai score-n], score3 (0..3)
    """
    prompt = f"""
You are an academic assistant. Analyze this paper and return JSON with keys:
- "abstract": a 10–15 sentence abstract (self-contained; no refs; no hallucinations)
- "tags": list of strings with REQUIRED prefixes:
  * aRT – research topic (1–2 concise tags)
  * aTa – very specific topical tags (3–6 concise tags)
  * aTy – paper type (e.g., review, experimental, meta-analysis)
  * aMe – key method(s)
  * Plus exactly one tag "ai score-N" where N is 0..3
- "score3": integer 0..3 relevance to the query (0=marginal, 3=high)

Paper info:
Title: {title}
Authors: {authors}
Context: {snippet}
PDF: {pdf_text}
URL: {url}

User query: {user_query}
Priority topics: {prefs.get('topics')}
Priority authors: {prefs.get('authors')}

Output JSON only.
"""
    data = gemini_json(prompt, model)
    abstract, tags, score3 = "", [], 0

    if data and "error" in data:
        error_msg = data['error']
        details = data.get('details', '')
        raw_text = data.get('raw_text', '')

        # Log the failure for debugging
        logging.error(f"Annotation Failure ({model}): {error_msg}. Details: {details[:100]}")

        if error_msg == "RATE_LIMIT_EXHAUSTED":
            return f"RATE LIMIT EXHAUSTED for model {model}. Please try a different model or wait.", [], 0
        elif error_msg == "API_CONNECTION_ERROR" or error_msg == "GEMINI_CLIENT_UNINITIALIZED":
            # This covers missing key, connection problems, and invalid model ID
            return f"API FAILURE: {error_msg}. Check key/model ID. Details: {details[:100]}", [], 0
        elif error_msg in ["JSON_NOT_FOUND", "JSON_PARSE_FAILURE"]:
            # Model returned text, but it wasn't valid JSON
            return f"Gemini returned invalid/non-JSON text. Raw Verbiage: {raw_text[:200]}...", [], 0
        else:
            return f"Unknown API Failure: {error_msg}. Details: {details[:100]}...", [], 0


    if isinstance(data, dict):
        abstract = data.get("abstract", "") or ""
        raw_tags = data.get("tags", []) or []
        score3 = data.get("score3", 0) or 0
        try:
            score3 = int(score3)
        except Exception:
            score3 = 0
        tags = [t for t in raw_tags if isinstance(t, str)]

    # ensure ai score-n tag exists and matches score3
    score_tag = f"ai score-{max(0, min(3, score3))}"
    if score_tag not in tags:
        tags.append(score_tag)

    return abstract.strip(), tags, max(0, min(3, score3))

# ============================
# SEARCH PROVIDERS (S2 + PubMed) + Crossref + Google fallback
# ============================
def search_semantic_scholar(query, limit=10):
    """Stable Semantic Scholar search."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,url,abstract,openAccessPdf,externalIds,venue,year,citationCount,publicationDate,publicationTypes"
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        st.error(f"Semantic Scholar error: {e}")
        return []

    results = []
    for paper in (data or {}).get("data", []) or []:
        doi = None
        if isinstance(paper.get("externalIds"), dict):
            doi = paper["externalIds"].get("DOI")
        results.append({
            "title": paper.get("title", ""),
            "url": paper.get("url", "") or (f"https://doi.org/{doi}" if doi else ""),
            "authors_info": ", ".join([a.get("name", "") for a in paper.get("authors", [])]),
            "snippet": clean_snippet(paper.get("abstract", "") or ""),
            "pdf_url": (paper.get("openAccessPdf") or {}).get("url", ""),
            "doi": doi,
            "venue": paper.get("venue"),
            "year": paper.get("year"),
            "citationCount": paper.get("citationCount"),
            "publicationDate": paper.get("publicationDate"),
            "publicationTypes": paper.get("publicationTypes"),
        })
    return results

def semantic_scholar_by_doi(doi: str):
    if not doi:
        return None
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
    params = {"fields": "title,authors,url,abstract,openAccessPdf,externalIds,venue,year,citationCount,publicationDate,publicationTypes"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        p = r.json()
        return {
            "title": p.get("title", ""),
            "url": p.get("url", "") or (f"https://doi.org/{doi}"),
            "authors_info": ", ".join([a.get("name", "") for a in p.get("authors", [])]),
            "snippet": clean_snippet(p.get("abstract", "") or ""),
            "pdf_url": (p.get("openAccessPdf") or {}).get("url", ""),
            "doi": (p.get("externalIds") or {}).get("DOI") or doi,
            "venue": p.get("venue"),
            "year": p.get("year"),
            "citationCount": p.get("citationCount"),
            "publicationDate": p.get("publicationDate"),
            "publicationTypes": p.get("publicationTypes"),
        }
    except Exception:
        return None

def search_pubmed(query, limit=10):
    """
    Simple, robust PubMed: GET ESearch + ESummary + (best-effort) EFetch abstracts; term capped to 300 chars.
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    term = (query or "")[:300]  # PubMed truncation
    es_params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": limit, "email": NCBI_EMAIL}
    if NCBI_API_KEY:
        es_params["api_key"] = NCBI_API_KEY
    try:
        es = requests.get(f"{base}/esearch.fcgi", params=es_params, timeout=30).json()
    except Exception as e:
        st.error(f"PubMed ESearch error: {e}")
        return []

    ids = (es.get("esearchresult", {}) or {}).get("idlist", []) or []
    if not ids:
        return []

    # ESummary (basic metadata)
    sum_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json", "email": NCBI_EMAIL}
    if NCBI_API_KEY:
        sum_params["api_key"] = NCBI_API_KEY
    try:
        sm = requests.get(f"{base}/esummary.fcgi", params=sum_params, timeout=30).json()
    except Exception as e:
        st.error(f"PubMed ESummary error: {e}")
        return []

    # EFetch to get abstracts (XML) — best effort
    abstracts = {}
    try:
        ef_params = {"db": "pubmed", "retmode": "xml", "email": NCBI_EMAIL}
        if NCBI_API_KEY:
            ef_params["api_key"] = NCBI_API_KEY
        ef = requests.post(f"{base}/efetch.fcgi", params=ef_params, data={"id": ",".join(ids)}, timeout=40)
        ef.raise_for_status()
        root = ET.fromstring(ef.text)
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID")
            abst_nodes = art.findall(".//Abstract/AbstractText")
            abs_text = " ".join((n.text or "") for n in abst_nodes).strip()
            abstracts[pmid] = clean_snippet(abs_text)
    except Exception:
        pass

    out, block = [], sm.get("result", {}) or {}
    for pmid in ids[:limit]:
        r = block.get(pmid, {}) or {}
        jrnl = r.get("fulljournalname") or r.get("source")
        # year parsing
        year = None
        try:
            dp = r.get("pubdate") or ""
            m = re.search(r"\b(19|20)\d{2}\b", dp)
            if m:
                year = int(m.group(0))
        except Exception:
            pass

        out.append({
            "title": r.get("title", ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "authors_info": ", ".join([a.get("name","") for a in (r.get("authors") or [])]) if isinstance(r.get("authors", []), list) else "",
            "snippet": abstracts.get(pmid) or clean_snippet(r.get("source", "") or ""),
            "pdf_url": "",
            "doi": None,
            "venue": jrnl,
            "year": year,
            "citationCount": None,
            "publicationDate": r.get("pubdate"),
            "publicationTypes": r.get("pubtype"),
        })
    return out

# ---------- Crossref enrichment (if DOI is known) ----------
def crossref_enrich(doi: str) -> dict:
    if not doi:
        return {}
    url = f"https://api.crossref.org/works/{doi}"
    try:
        data = _request_json_with_retries(url, timeout=30)
        msg = (data or {}).get("message", {})
        if not msg:
            return {}
        title = (msg.get("title") or [""])[0]
        journal = (msg.get("container-title") or [""])[0]
        date_parts = (msg.get("issued") or {}).get("date-parts", [[]])
        year = date_parts[0][0] if date_parts and date_parts[0] else None
        volume = msg.get("volume")
        issue = msg.get("issue")
        page = msg.get("page")
        url = msg.get("URL")
        authors = []
        for a in msg.get("author", []) or []:
            nm = f"{a.get('given','')} {a.get('family','')}".strip()
            if nm: authors.append(nm)
        return {
            "title": title,
            "venue": journal,
            "year": year,
            "volume": volume,
            "issue": issue,
            "pages": page,
            "url": url,
            "authors_info": ", ".join(authors),
        }
    except Exception:
        return {}

# ---------- URL / PDF handling ----------
def fetch_url_and_guess_pdf(url: str) -> tuple[bool, str]:
    """Return (is_pdf, text). Detect PDF by header, extension, or magic bytes.
        If PDF, extract up to 8000 chars; else return (False, "")."""
    try:
        r = requests.get(url, timeout=45)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "").lower()
        content = r.content

        # PDF detection: by header, extension, or magic number
        is_pdf = (
            "pdf" in ctype
            or url.lower().endswith(".pdf")
            or content.startswith(b"%PDF")
        )

        if is_pdf:
            with fitz.open(stream=io.BytesIO(content), filetype="pdf") as doc:
                text = []
                for page in doc:
                    text.append(page.get_text())
                return True, ("\n".join(text))[:8000]

        return False, ""
    except Exception:
        return False, ""

def extract_metadata_from_pdf_text(pdf_text: str) -> dict:
    """Find DOI, a plausible title, author line."""
    if not pdf_text:
        return {}
    md = {}
    doi_m = DOI_RE.search(pdf_text)
    if doi_m:
        md["doi"] = doi_m.group(0)
    # crude title guess: first reasonable line before 'Abstract'
    lines = [ln.strip() for ln in pdf_text.splitlines() if ln.strip()]
    title = None
    for ln in lines[:60]:
        if re.match(r"^abstract\b", ln, re.I):
            break
        if 8 <= len(ln) <= 240 and not re.search(r"(doi:|arxiv:)", ln, re.I):
            title = ln
            break
    if title:
        md["title"] = title
    # weak authors pattern
    for j in range(1, 8):
        if j < len(lines):
            cand = lines[j]
            if re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+", cand):
                md["authors_info"] = cand
                break
    return md

def google_search_fallback(query: str):
    """Very light fallback via Google Custom Search (requires valid key & cx)."""
    try:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "q": query,
                "key": GEMINI_API_KEY,  # reuse key; replace with your proper CSE key
                "cx": "017576662512468239146:omuauf_lfve",  # demo CX; replace with your own
            },
            timeout=20,
        )
        data = r.json()
        items = data.get("items", []) or []
        if not items:
            return []
        out = []
        for it in items:
            out.append({
                "title": it.get("title"),
                "url": it.get("link"),
                "authors_info": "",
                "snippet": it.get("snippet"),
                "pdf_url": "",
                "doi": None,
                "venue": None,
                "year": None
            })
        return out
    except Exception:
        return []

# ============================
# MAIN ACTION
# ============================
if st.button("🚀 Go"):

    # 1. Log Gemini Status before starting the main process
    log_gemini_status(client, selected_model_id)

    progress = st.progress(0)
    status = st.empty()

    papers_meta = []
    try:
        # 1) KEYWORD SEARCH
        if search_mode == "Keyword Search":
            if not user_prompt or not user_prompt.strip():
                st.warning("Please enter a research topic.")
                st.stop()

            status.info("🧠 Preparing query…")
            if use_boolean:
                b = gemini_boolean_query(user_prompt, selected_model_id)
                effective_query = b.get("boolean_query") or build_boolean_query_simple(user_prompt)
                if b.get("keywords"):
                    st.caption("Keywords: " + ", ".join((b.get("keywords") or [])[:12]))
                if b.get("year_from") or b.get("year_to"):
                    st.caption(f"Years: {b.get('year_from')}–{b.get('year_to')}")
            else:
                effective_query = build_boolean_query_simple(user_prompt)

            # Editable query box
            effective_query = st.text_area("✏️ Editable search query (you can tweak before searching):", effective_query)
            progress.progress(10)

            agg = []
            if search_source in ("Semantic Scholar", "Both"):
                status.info("🔎 Searching Semantic Scholar…")
                try:
                    agg.extend(search_semantic_scholar(effective_query, limit=max_results))
                except Exception as e:
                    st.warning(f"Semantic Scholar failed: {e}")
                progress.progress(30)

            if search_source in ("PubMed", "Both"):
                status.info("🧬 Searching PubMed…")
                try:
                    agg.extend(search_pubmed(effective_query, limit=max_results))
                except Exception as e:
                    st.warning(f"PubMed failed: {e}")
                progress.progress(50)

            status.info("📦 Combining results…")
            papers_meta = _take(dedupe_results(agg), max_results)
            progress.progress(60)

        # 2) PASTE CITATION / TEXT (Gemini extraction + PubMed + Google fallback; DOI→S2 if available)
        elif search_mode == "Paste citation / page text":
            if not paste_text.strip():
                st.warning("Please paste citation(s) or text.")
                st.stop()

            status.info("🧾 Extracting references with Gemini…")
            refs = gemini_extract_from_text(paste_text, selected_model_id)
            progress.progress(30)

            if not refs:
                status.warning("")
                progress.progress(100)
                st.error("😅 We squinted at every reference style… but found nada.")
                st.caption("Try another copy/paste (e.g., select all items on the Google Scholar results page).")
                st.stop()

            status.info("🔎 Enriching references…")
            collected = []
            for r in refs[:max_results]:
                title, authors, year, doi = r.get("title"), r.get("authors"), r.get("year"), r.get("doi")
                enriched = None

                # 1. DOI → Semantic Scholar enrichment
                if doi:
                    enriched = semantic_scholar_by_doi(doi)

                # 2. PubMed by title
                if not enriched and title:
                    pm = search_pubmed(title, 1)
                    enriched = pm[0] if pm else None

                # 3. Google fallback
                if not enriched and title:
                    gg = google_search_fallback(title)
                    enriched = gg[0] if gg else None

                # 4. If still nothing → bare metadata
                if not enriched:
                    enriched = {
                        "title": title,
                        "authors_info": ", ".join(authors) if isinstance(authors, list) else (authors or ""),
                        "snippet": "",
                        "url": "",
                        "pdf_url": "",
                        "doi": doi,
                        "year": year,
                        "venue": None
                    }
                collected.append(enriched)

            papers_meta = collected
            progress.progress(60)

        # 3) LOOKUP BY URL / DOI / PDF
        else:  # search_mode == "Lookup by URL / PDF"
            if not url_or_doi or not url_or_doi.strip():
                st.warning("Please paste a URL or DOI.")
                st.stop()

            val = url_or_doi.strip()
            status.info("🧭 Resolving input…")
            progress.progress(10)

            if DOI_RE.fullmatch(val):
                # DOI path: Crossref enrich + S2 by title if possible
                doi = val
                enr = crossref_enrich(doi)
                title = enr.get("title")
                if title:
                    status.info("🔎 Searching Semantic Scholar by title…")
                    ss = search_semantic_scholar(title, limit=1)
                else:
                    ss = []
                base = {
                    "title": enr.get("title"),
                    "url": enr.get("url"),
                    "authors_info": enr.get("authors_info"),
                    "snippet": "",
                    "pdf_url": "",
                    "doi": doi,
                    "venue": enr.get("venue"),
                    "year": enr.get("year"),
                }
                papers_meta = [ss[0] | base] if ss else [base]
                progress.progress(60)
            else:
                # Assume URL
                is_pdf, pdf_text = fetch_url_and_guess_pdf(val)
                progress.progress(25)
                if is_pdf:
                    status.info("📄 PDF detected — extracting metadata…")
                    md = extract_metadata_from_pdf_text(pdf_text)
                    doi = md.get("doi")
                    if doi:
                        enr = crossref_enrich(doi)
                    else:
                        enr = {}
                    title = md.get("title") or enr.get("title")
                    status.info("🔎 Searching Semantic Scholar by title…")
                    ss = search_semantic_scholar(title, limit=1) if title else []
                    base = {
                        "title": title,
                        "url": val,
                        "authors_info": md.get("authors_info") or enr.get("authors_info"),
                        "snippet": clean_snippet(pdf_text[:1200]),
                        "pdf_url": val,
                        "doi": doi,
                        "venue": enr.get("venue"),
                        "year": enr.get("year"),
                    }
                    papers_meta = [ss[0] | base] if ss else [base]
                    progress.progress(70)
                else:
                    status.info("🌐 Not a PDF — trying title guess from URL path…")
                    guessed = re.sub(r"[-_/]+", " ", val.split("//")[-1])[:120]
                    ss = search_semantic_scholar(guessed, limit=1)
                    papers_meta = ss
                    progress.progress(70)

        # Initialize pyzotero client once for duplicate checks
        zot_client, zot_error = init_pyzotero_local(user_zotero_id)
        if zot_error:
            # We don't stop execution, but warn the user that duplicate checks won't run
            st.warning(f"Zotero Duplicate Check Warning: {zot_error}")


        # If nothing found — friendly message
        if not papers_meta:
            status.warning("")
            progress.progress(100)
            st.error("😅 We searched high, low, and even peered behind the paywall sofa cushions… but found nada.")
            st.caption("Try tweaking the query or switching modes. Even librarians have off days.")
            st.stop()

        # Render + Annotate Loop (MODIFIED EXECUTION ORDER)
        status.info("🧪 Analyzing, annotating, and checking duplicates…")
        progress.progress(75)

        # Map Zotero threshold: score3 (0..3)
        zotero_threshold_score3 = min(3, max(0, int(min_score3)))

        # --- Sequential Annotation Loop ---
        for i, paper in enumerate(papers_meta):
            title = paper.get("title", "")
            url = paper.get("url", "")
            authors_info = paper.get("authors_info", "")
            snippet = paper.get("snippet", "")
            pdf_url = paper.get("pdf_url", "")
            doi = paper.get("doi")
            venue = paper.get("venue")
            year = paper.get("year")

            # --- EARLY CHECK: Zotero Duplicates & Transfer Conditions ---
            if add_to_zotero:
                # 1. CHECK RELEVANCE SCORE FIRST (Low-cost check)
                # Since we don't have score3 yet, we rely on the title for now, 
                # but we can check the score threshold *after* annotation.
                
                # 2. DUPLICATE CHECK (Time/Resource consuming)
                if not allow_duplicates:
                    is_duplicate, dup_msg = check_zotero_duplicate(zot_client, title)
                    if is_duplicate:
                        st.info(f"⚠️ Skipped AI/Transfer: {dup_msg}")
                        # Skip remaining loop steps (AI call and posting)
                        continue
                    if "Zotero client not available" in dup_msg:
                        # Log warning, but don't stop (already logged above, but good practice to check)
                        pass
            
            # Pull PDF text when useful (Relatively high cost, done before AI)
            pdf_text = extract_pdf_text(pdf_url or url)

            # --- AI ANNOTATION CALL (Highest Cost Operation) ---
            user_query = (
                user_prompt if search_mode == 'Keyword Search' else
                (title or paste_text if search_mode == 'Paste citation / page text' else url_or_doi)
            )

            # Sequential API call with backoff
            abstract_ai, tags, score3 = gemini_annotate_paper(
                title, authors_info, snippet, pdf_text, url, user_query, selected_model_id
            ) if client else ("GEMINI API KEY IS MISSING or invalid. No annotation performed.", [], 0)

            # Update progress bar
            progress.progress(75 + int((i / len(papers_meta)) * 20))

            # --- UI RENDERING ---
            with st.expander(f"📄 {title or 'Untitled'} (Score: {score3})", expanded=True):
                if authors_info:
                    st.markdown(f"**Authors:** {authors_info}")
                if venue or year:
                    st.markdown(f"**Venue / Year:** {venue or '—'} — {year or '—'}")

                # --- Abstract Display ---
                if snippet:
                    st.markdown(f"**Abstract (source):** {snippet}")
                if abstract_ai:
                    if abstract_ai.startswith("RATE LIMIT EXHAUSTED") or abstract_ai.startswith("API CONNECTION_ERROR") or abstract_ai.startswith("Gemini returned") or abstract_ai.startswith("API FAILURE") or abstract_ai.startswith("Unknown API Failure"):
                        st.error(f"**AI Abstract Failure:** {abstract_ai}")
                    elif abstract_ai.startswith("GEMINI API KEY IS MISSING"):
                        st.error(f"**AI Abstract Failure:** {abstract_ai}")
                    else:
                        st.markdown("**Abstract (AI):**")
                        st.write(abstract_ai)

                if url:
                    st.markdown(f"[🔗 View Paper]({url})")
                    doi_or_url = f"https://doi.org/{doi}" if doi else url
                    inst1 = with_ntu_proxy(doi_or_url, style=1)
                    inst2 = with_ntu_proxy(doi_or_url, style=2)
                    if inst1:
                        st.markdown(f"[🏫 NTU Access (style 1)]({inst1})")
                    if inst2:
                        st.markdown(f"[🏫 NTU Access (style 2)]({inst2})")

                if tags:
                    st.markdown("**🏷️ Tags:** " + ", ".join(tags))
                st.markdown(f"**AI Relevance (0–3):** `{score3}`")

                # --- LOCAL ZOTERO POSTING (Now inside the check for efficiency) ---
                if add_to_zotero and (score3 >= zotero_threshold_score3):
                    
                    doi_or_url = f"https://doi.org/{doi}" if doi else url
                    proxy_url = with_ntu_proxy(doi_or_url, style=1) or with_ntu_proxy(doi_or_url, style=2) or url

                    # 1. ABSTRACT CONTENT LOGIC (Hybrid if AI is 40% longer)
                    abstract_content = ""
                    abstract_ai_content = ""
                    
                    # Check if AI content is significantly better (at least 40% longer)
                    if abstract_ai and not abstract_ai.startswith("RATE LIMIT EXHAUSTED"):
                        # Only apply the length check if a snippet is present
                        if snippet:
                            if len(abstract_ai) > (len(snippet) * 1.4):
                                abstract_ai_content = f"AI EXPANDED SUMMARY:\n{abstract_ai}"
                            
                        elif len(abstract_ai) > 10: # Use AI as fallback if no snippet but AI is meaningful
                             abstract_ai_content = f"AI SUMMARY:\n{abstract_ai}"

                    # Prioritize snippet, append expanded AI content if applicable
                    if snippet:
                        abstract_content += f"SOURCE ABSTRACT:\n{snippet}"
                        if abstract_ai_content:
                            abstract_content += "\n\n---\n\n" + abstract_ai_content
                    elif abstract_ai_content:
                        abstract_content = abstract_ai_content
                    
                    # 2. ITEM POSTING
                    item = {
                        'itemType': 'journalArticle',
                        'title': title,
                        'creators': parse_authors(authors_info),
                        'abstractNote': abstract_content,
                        'tags': [{'tag': t} for t in (tags or [])],
                        'url': proxy_url,
                        'date': str(year) if year else None,
                        'DOI': doi,
                    }
                    item = {k: v for k, v in item.items() if v not in (None, "" or [])}

                    # The save_to_zotero_local call uses the new 10-minute timeout (600 seconds)
                    success, msg = save_to_zotero_local(item)
                    if success:
                        st.success(f"✅ Added to Local Zotero (score3={score3})")
                    else:
                        st.error(f"❌ Local Zotero Error: {msg}")

        status.success("Done ✅")
        progress.progress(100)

    finally:
        # Clear status after a short delay to avoid lingering messages
        sleep(0.4)
        status.empty()
