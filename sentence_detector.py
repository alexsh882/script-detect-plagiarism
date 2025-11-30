"""
Sentence-level plagiarism detection module.
Detects copied content even when document order is shuffled.
"""

import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple


def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing extra whitespace."""
    return " ".join(text.split()).strip()


def extract_sentences(text: str, min_length: int = 50) -> List[str]:
    """
    Extract sentences from text.

    Args:
        text: Raw text to extract sentences from
        min_length: Minimum character length for a sentence

    Returns:
        List of normalized sentences
    """
    # Split by sentence boundaries
    sentences = re.split(r"[.!?]+\s+", text)

    # Clean and filter
    cleaned = []
    for s in sentences:
        s = normalize_text(s)
        # Remove very short sentences and common headers
        if len(s) >= min_length and not s.startswith("Seminario de Actualización"):
            cleaned.append(s)

    return cleaned


def similarity_ratio(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts using SequenceMatcher."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def find_matching_sentences(
    text1: str,
    text2: str,
    min_length: int = 50,
    threshold: float = 0.80,
    exact_threshold: float = 0.95,
) -> Dict:
    """
    Find matching sentences between two texts.

    Args:
        text1: First text
        text2: Second text
        min_length: Minimum sentence length
        threshold: Minimum similarity to consider a match
        exact_threshold: Threshold to consider a match "exact"

    Returns:
        Dictionary with match statistics and details
    """
    # Extract sentences
    sents1 = extract_sentences(text1, min_length)
    sents2 = extract_sentences(text2, min_length)

    if not sents1 or not sents2:
        return {
            "total_matches": 0,
            "exact_matches": 0,
            "coverage": 0.0,
            "matches": [],
            "total_chars": 0,
        }

    # Find matches
    matches = []
    seen = set()  # Track matched sentence pairs to avoid duplicates

    for i, s1 in enumerate(sents1):
        for j, s2 in enumerate(sents2):
            ratio = similarity_ratio(s1, s2)
            if ratio >= threshold:
                key = (i, j)
                if key not in seen:
                    seen.add(key)
                    matches.append(
                        {
                            "idx1": i,
                            "idx2": j,
                            "ratio": ratio,
                            "text1": s1,
                            "text2": s2,
                            "length": len(s1),
                            "is_exact": ratio >= exact_threshold,
                        }
                    )

    # Sort by similarity
    matches.sort(key=lambda x: x["ratio"], reverse=True)

    # Calculate statistics
    total_chars = sum(len(s) for s in sents1)
    matched_chars = sum(m["length"] for m in matches)
    coverage = (matched_chars / total_chars) if total_chars > 0 else 0.0
    exact_matches = sum(1 for m in matches if m["is_exact"])

    return {
        "total_matches": len(matches),
        "exact_matches": exact_matches,
        "coverage": coverage,
        "matches": matches,
        "total_chars": total_chars,
        "matched_chars": matched_chars,
    }


def is_plagiarism_detected(
    match_stats: Dict,
    min_exact_matches: int = 5,
    min_total_matches: int = 10,
    min_coverage: float = 0.08,
) -> Tuple[bool, str]:
    """
    Determine if plagiarism is detected based on match statistics.

    Args:
        match_stats: Dictionary from find_matching_sentences
        min_exact_matches: Minimum exact matches to flag
        min_total_matches: Minimum total matches to flag
        min_coverage: Minimum coverage percentage to flag

    Returns:
        Tuple of (is_plagiarism, reason)
    """
    exact = match_stats["exact_matches"]
    total = match_stats["total_matches"]
    coverage = match_stats["coverage"]

    if exact >= min_exact_matches:
        return True, f"{exact} exact matches (threshold: {min_exact_matches})"

    if total >= min_total_matches and coverage >= min_coverage:
        return True, f"{total} matches with {coverage * 100:.1f}% coverage"

    return False, ""


def format_sentence_match_report(
    filename1: str, filename2: str, match_stats: Dict, max_matches_to_show: int = 5
) -> List[str]:
    """
    Format a detailed report of sentence matches.

    Args:
        filename1: First file name
        filename2: Second file name
        match_stats: Dictionary from find_matching_sentences
        max_matches_to_show: Maximum matches to include in report

    Returns:
        List of formatted report lines
    """
    lines = []

    exact = match_stats["exact_matches"]
    total = match_stats["total_matches"]
    coverage = match_stats["coverage"]

    lines.append("   📊 Sentence-level analysis:")
    lines.append(f"      - Total matches: {total}")
    lines.append(f"      - Exact matches: {exact}")
    lines.append(f"      - Coverage: {coverage * 100:.1f}%")

    if match_stats["matches"] and max_matches_to_show > 0:
        lines.append(
            f"      - Sample matches (showing top {min(max_matches_to_show, total)}):"
        )

        for idx, match in enumerate(match_stats["matches"][:max_matches_to_show], 1):
            ratio = match["ratio"]
            text = match["text1"]
            preview = text[:80] + "..." if len(text) > 80 else text
            lines.append(f'         {idx}. [{ratio * 100:.0f}%] "{preview}"')

    return lines
