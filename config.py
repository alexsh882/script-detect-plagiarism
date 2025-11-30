# --- CONFIGURACIÓN DEL MODO DE DETECCIÓN ---
# Modos disponibles: 'fast', 'thorough', 'hybrid', 'smart'
#
# fast:      Solo similitud TF-IDF a nivel de documento (más rápido, ~segundos)
#            - Detecta copias exactas y plagio directo
#            - No detecta plagio por reordenamiento
#
# thorough:  Solo análisis por oraciones (lento, ~1-3 min para 47 archivos)
#            - Detecta plagio por reordenamiento y plagio sofisticado
#            - Análisis más exhaustivo
#
# hybrid:    TF-IDF Y análisis por oraciones en todos los pares (más lento)
#            - Detecta todos los tipos de plagio
#            - Reporta ambas métricas
#
# smart:     Enfoque inteligente en dos fases (recomendado, equilibrado)
#            - Fase 1: TF-IDF en todos los pares
#            - Fase 2: análisis por oraciones en pares sospechosos (40-70% similarity)
#            - Mejor equilibrio entre velocidad y precisión

# MODO DE DETECCIÓN, opciones disponibles: fast, thorough, hybrid, or smart
DETECTION_MODE = "smart"  # por defecto

# --- SIMILARITY THRESHOLD (TF-IDF) ---
# Umbral de similitud para considerar una coincidencia
DOCUMENT_SIMILARITY_THRESHOLD = 0.70

# --- SENTENCE-LEVEL DETECTION ---
# Longitud mínima de una oración para ser analizada
SENTENCE_MIN_LENGTH = 50

# Similitud mínima para considerar una coincidencia
SENTENCE_SIMILARITY_THRESHOLD = 0.80

# Umbral para considerar una coincidencia "exacta"
SENTENCE_EXACT_MATCH_THRESHOLD = 0.95

# Criterios para "flaggear" un par como plagio (cualquier condición activa alerta)
SENTENCE_MIN_EXACT_MATCHES = 5  # Flag si ≥5 exact/near-exact coincidencias
SENTENCE_MIN_TOTAL_MATCHES = 10  # Flag si ≥10 total coincidencias
SENTENCE_MIN_COVERAGE = 0.08  # Flag si ≥8% de contenido coincide

# --- CONFIGURACIÓN DEL MODO SMART ---
# Rango para pares sospechosos que activan el análisis por oraciones
# "Límite inferior" (reducido para detectar plagio por re-ordenamiento)
SMART_MODE_MIN_SIMILARITY = 0.35

# "Límite superior" (arriba de esto ya se detecta plagio)
SMART_MODE_MAX_SIMILARITY = 0.70

# --- CONFIGURACIÓN DEL OUTPUT ---

# Mostrar detalles de coincidencia de oraciones en el output
SHOW_DETAILED_SENTENCE_MATCHES = True

# Ajusta el número máximo de coincidencias de oraciones a mostrar por par.
MAX_SENTENCE_MATCHES_TO_SHOW = 5
