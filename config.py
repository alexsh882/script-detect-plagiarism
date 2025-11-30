# --- DETECTION MODE CONFIGURATION ---
# Available modes: 'fast', 'thorough', 'hybrid', 'smart'
#
# fast:      Only TF-IDF document-level similarity (fastest, ~seconds)
#            - Detects exact copies and direct plagiarism
#            - Misses order-shuffling plagiarism
#
# thorough:  Only sentence-level matching (slower, ~1-3 min for 47 files)
#            - Detects order-shuffling and sophisticated plagiarism
#            - More exhaustive analysis
#
# hybrid:    Both TF-IDF AND sentence-level on all pairs (slowest)
#            - Detects all types of plagiarism
#            - Reports both metrics
#
# smart:     Intelligent two-phase approach (recommended, balanced)
#            - Phase 1: TF-IDF on all pairs
#            - Phase 2: Sentence-level ONLY on suspicious pairs (40-70% similarity)
#            - Best balance of speed and accuracy

DETECTION_MODE = "smart"  # Change this to: fast, thorough, hybrid, or smart

# --- DOCUMENT-LEVEL DETECTION (TF-IDF) ---
DOCUMENT_SIMILARITY_THRESHOLD = 0.70

# --- SENTENCE-LEVEL DETECTION ---
SENTENCE_MIN_LENGTH = 50  # Minimum chars for a sentence to be analyzed
SENTENCE_SIMILARITY_THRESHOLD = 0.80  # Minimum similarity to consider a match
SENTENCE_EXACT_MATCH_THRESHOLD = 0.95  # Threshold to consider a match "exact"

# Criteria to flag a pair as plagiarism (ANY condition triggers alert)
SENTENCE_MIN_EXACT_MATCHES = 5  # Flag if ≥5 exact/near-exact matches
SENTENCE_MIN_TOTAL_MATCHES = 10  # Flag if ≥10 total matches
SENTENCE_MIN_COVERAGE = 0.08  # Flag if ≥8% of content matches

# --- SMART MODE CONFIGURATION ---
# Range for "suspicious" pairs that trigger sentence-level analysis
SMART_MODE_MIN_SIMILARITY = (
    0.35  # Lower bound (lowered to catch order-shuffling plagiarism)
)
SMART_MODE_MAX_SIMILARITY = 0.70  # Upper bound (above this already flagged)

# --- OUTPUT CONFIGURATION ---
SHOW_DETAILED_SENTENCE_MATCHES = True  # Show sentence match details in output
MAX_SENTENCE_MATCHES_TO_SHOW = (
    5  # Maximum number of sentence matches to display per pair
)
