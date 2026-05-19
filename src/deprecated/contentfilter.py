import re
import utils
from icecream import ic

import re
from collections import Counter

# --- Layer 1: Surface Signature Layer ---
def surface_signature_trigger(lines, threshold=0.3):
    surface_keywords = [
        "subscribe", "register", "select", "offer", "only $", "premium", 
        "gift article", "digital edition", "unlock", "access", "try for", "continue reading"
    ]
    match_count = sum(
        1 for line in lines if any(kw in line.lower() for kw in surface_keywords)
    )
    return match_count / max(len(lines), 1) > threshold

# --- Layer 2: Functional Intent Layer ---
def call_to_action_density(lines, threshold=0.25):
    cta_keywords = ["subscribe", "register", "sign up", "select", "explore", "start", "try", "get", "buy"]
    cta_lines = sum(
        1 for line in lines if any(word in line.lower() for word in cta_keywords)
    )
    return cta_lines / max(len(lines), 1) > threshold

# --- Layer 3: Redundancy Layer ---
def repeated_phrases(lines, min_len=5, threshold=2):
    normalized = [line.strip().lower() for line in lines if len(line.strip()) >= min_len]
    counts = Counter(normalized)
    return any(count >= threshold for count in counts.values())

# --- Layer 4: Narrative Density Layer ---
def narrative_density(lines, required_blocks=1):
    block_count = 0
    current_block = 0
    for line in lines:
        if re.search(r'\b\w+\b\s+(is|are|was|were|has|had|will|does|did|says|believes)\s+\b\w+', line, re.IGNORECASE):
            current_block += 1
        else:
            if current_block >= 2:
                block_count += 1
            current_block = 0
    if current_block >= 2:
        block_count += 1
    return block_count >= required_blocks

# --- Layer 5: Compression and Flow Layer (Optional / Diagnostic) ---
def compression_ratio(lines):
    total_chars = sum(len(line) for line in lines)
    unique_lines = set(line.strip().lower() for line in lines if len(line.strip()) > 5)
    compressed_chars = sum(len(line) for line in unique_lines)
    return compressed_chars / total_chars if total_chars > 0 else 1

# --- MASTER FILTER FUNCTION ---
def should_reject_input(text):
    lines = text.strip().splitlines()

    if not lines or len(lines) < 3:
        return True, "Insufficient line count"

    if surface_signature_trigger(lines):
        return True, "Surface signature triggered"

    if call_to_action_density(lines):
        return True, "Call-to-action density exceeded"

    if repeated_phrases(lines):
        return True, "Redundant boilerplate repetition"

    if not narrative_density(lines):
        return True, "Narrative structure absent"

    # Optional diagnostic: compression threshold
    if compression_ratio(lines) < 0.5:
        return True, "Compression ratio too low (likely repetitive)"

    return False, "Passed all checks"


