import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from constants import COMMON_PHRASES, END_MARKERS, STOP_WORDS_ES

# --- CONFIGURATION ---
FILES_DIR = Path("./files")
EXTENSIONS = ["*.pdf", "*.docx", "*.md"]
ALERT_THRESHOLD = 0.7
DEBUG_MODE = False  # Saves cleaned text to debug/ folder


def clean_text(text: str) -> str:
    """Borra consignas repetidas y recorta bibliografía."""

    # Normalize a bit (remove multiple spaces)
    text = " ".join(text.split())

    # 1. Remove repeated phrases
    for phrase in COMMON_PHRASES:
        # Remove regardless of case
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        text = pattern.sub("", text)

    # 2. Cut Bibliography
    text_lower = text.lower()
    cut_position = -1

    for marker in END_MARKERS:
        pos = text_lower.rfind(marker)  # Find the last occurrence
        # If found and it's in the last 20% of the document (to avoid false cuts at the start)
        if pos != -1 and pos > len(text) * 0.5:
            if cut_position == -1 or pos < cut_position:
                cut_position = pos

    if cut_position != -1:
        text = text[:cut_position]

    return text


def extract_text(file_path: Path) -> Optional[str]:
    """Extrae el texto del archivo."""
    text = ""
    ext = file_path.suffix.lower()

    try:
        if ext == ".pdf":
            reader = PdfReader(str(file_path))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif ext == ".docx":
            doc = Document(str(file_path))
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif ext == ".md":
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        print(f"[ERROR] {file_path.name}: {e}")
        return None

    # Aplicamos la limpieza aquí
    return clean_text(text)


def get_files(directory: Path, extensions: List[str]) -> List[Path]:
    """Busca archivos con las extensiones dadas en el directorio."""
    files = []
    for ext in extensions:
        files.extend(directory.glob(ext))
    return files


def save_debug_text(filename: str, content: str) -> None:
    """Guarda el texto procesado en la carpeta debug."""
    debug_dir = Path("debug")
    if not debug_dir.exists():
        debug_dir.mkdir()

    with (debug_dir / f"{filename}.txt").open("w", encoding="utf-8") as f:
        f.write(content)


def save_results(output_lines: List[str], filename: str = "results.txt") -> None:
    """Guarda los resultados en un archivo de texto dentro de la carpeta output."""
    output_dir = Path("output")
    if not output_dir.exists():
        output_dir.mkdir()

    output_path = output_dir / filename
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\nℹ️  Los resultados se guardaron en '{output_path}'")


def analyze_similarities() -> None:
    """
    Analiza las similitudes entre los archivos.
    """
    print("--- Analizando similitudes ---")
    files = get_files(FILES_DIR, EXTENSIONS)

    if not files:
        print("No se encontraron archivos. Tenes que ponerlos en la carpeta 'files'")
        return

    print(
        f"Procesando {len(files)} archivos (Limpieza de frases y bibliografía y cosas así)..."
    )

    if DEBUG_MODE:
        print("[DEBUG] Guardando textos limpiados en 'debug/' folder")

    texts: List[str] = []
    filenames: List[str] = []

    for file in files:
        content = extract_text(file)
        # Filter texts that are empty after cleaning
        if content and len(content.strip()) > 50:
            texts.append(content)
            filename = file.name
            filenames.append(filename)

            if DEBUG_MODE:
                save_debug_text(filename, content)

    if len(texts) < 2:
        print("At least 2 valid files are needed to compare.")
        return

    print("--- Calculando similitudes ---")
    # IMPROVEMENT: Stop words and N-grams (1 to 3 words)
    vectorizer = TfidfVectorizer(
        stop_words=STOP_WORDS_ES,
        ngram_range=(1, 3),  # Detect phrases up to 3 words
        min_df=1,  # Consider words even if they appear in only one doc
    ).fit_transform(texts)

    vectors = vectorizer.toarray()
    similarity_matrix = cosine_similarity(vectors)

    suspicious_pairs: List[Tuple[str, str, float]] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            score = similarity_matrix[i][j]
            if score > ALERT_THRESHOLD:
                suspicious_pairs.append((filenames[i], filenames[j], score))

    suspicious_pairs.sort(key=lambda x: x[2], reverse=True)

    output_lines = []
    output_lines.append("\nResultados de análisis (Mejorado con N-grams y Stop Words):")
    output_lines.append("=" * 50)
    if not suspicious_pairs:
        output_lines.append("✅ No se detectaron copias obvias.")
    else:
        output_lines.append(
            f"⚠️  Se encontraron {len(suspicious_pairs)} pares sospechosos:\n"
        )
        for f1, f2, score in suspicious_pairs:
            output_lines.append(f"🔴 {score * 100:.2f}% :: {f1} <--> {f2}")

    # Print to console
    for line in output_lines:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))

    # Save to file
    save_results(
        output_lines, "results" + datetime.now().strftime("_%Y%m%d_%H%M%S") + ".txt"
    )

    # Cleanup debug if requested
    if not DEBUG_MODE and Path("debug").exists():
        shutil.rmtree("debug")


if __name__ == "__main__":
    analyze_similarities()
