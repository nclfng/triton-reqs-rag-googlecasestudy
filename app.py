"""Triton Requirements Guide - FastAPI app (endpoints are added in step 3)."""

# ---------- Config: change settings here ----------
EMBED_MODEL = "gemini-embedding-001"
GENERATE_MODEL = "gemini-2.5-flash"   # one constant; swap for the current Flash name
CHUNK_WORDS = 400                     # target words per chunk
CHUNK_OVERLAP = 40                    # words repeated between neighbouring chunks
TOP_K = 5                             # chunks sent to Gemini
DAILY_LIMIT = 10                      # questions per IP per day on the server key
MAX_QUESTION_CHARS = 400
# --------------------------------------------------
