# 🕵️‍♂️ Detector de Plagio con Sistema Híbrido (Anti-Machete 🪓)

Bienvenido al **Detector de Plagio**. Esta herramienta avanzada identifica documentos con similitudes sospechosas utilizando un **sistema híbrido de detección** con 4 modos de análisis. Es ideal para analizar lotes de entregas y detectar tanto plagio directo como sofisticado.

## 🎯 ¿Cómo funciona?

El script procesa todos los archivos ubicados en la carpeta `files/` (soporta formatos PDF, DOCX y MD) y los compara utilizando dos técnicas complementarias:

### 1. **Análisis a Nivel de Documento (TF-IDF)**

- Compara la similitud general entre documentos completos
- Detecta copias directas y plagio "perezoso"
- Muy rápido (~segundos)

### 2. **Análisis a Nivel de Oraciones (Fuzzy Matching)**

- Compara oraciones individuales entre documentos
- **Detecta plagio sofisticado**: reordenamiento de párrafos, cambios de estructura
- Encuentra coincidencias exactas aunque estén en diferente orden
- Más lento pero más exhaustivo

## 🚀 Modos de Detección

Al ejecutar el script, se te presentará un **menú interactivo** para seleccionar el modo:

```
================================================================================
DETECTOR DE PLAGIO - Selección de Modo
================================================================================

Modos disponibles:

  1. FAST      - Solo TF-IDF (⚡ ~segundos, detecta copias directas)
  2. THOROUGH  - Solo análisis de oraciones (🔍 ~1-2 min, detecta plagio sofisticado)
  3. HYBRID    - Ambos análisis (🎯 ~2-3 min, máxima precisión)
  4. SMART     - Inteligente en 2 fases (🧠 ~20 seg, balanceado) [RECOMENDADO]

Modo por defecto: SMART
```

### Modo FAST ⚡

- **Velocidad**: ~segundos
- **Qué detecta**: Copias directas, plagio evidente
- **Qué NO detecta**: Plagio con reordenamiento
- **Cuándo usar**: Primera revisión rápida, muchos archivos

### Modo THOROUGH 🔍

- **Velocidad**: ~1-2 minutos (47 archivos)
- **Qué detecta**: Plagio sofisticado, reordenamiento, párrafos copiados
- **Cuándo usar**: Sospecha de plagio avanzado

### Modo HYBRID 🎯

- **Velocidad**: ~2-3 minutos
- **Qué detecta**: TODO - máxima precisión
- **Cuándo usar**: Análisis final definitivo

### Modo SMART 🧠 (Recomendado)

- **Velocidad**: ~20 segundos
- **Qué detecta**: Ambos tipos de plagio
- **Cómo funciona**:
  - Fase 1: TF-IDF en todos los pares (rápido)
  - Fase 2: Análisis de oraciones SOLO en pares sospechosos (35-70% similitud)
- **Cuándo usar**: Uso general, mejor balance velocidad/precisión

**NOTA**: El modo SMART es el más recomendado ya que balancea entre velocidad y precisión. El tiempo estimado es aproximado y puede variar según la cantidad de archivos y la velocidad de tu computadora.

## ⚙️ Configuración Importante

### Filtros de Contenido (en `constants.py`)

**Estas configuraciones son específicas para cada Trabajo Práctico**:

1. **`COMMON_PHRASES`**: Frases que se repiten en *todos* los trabajos (consignas, nombre de materia). Si no las ponés, el script va a pensar que se copiaron porque todos tienen el mismo texto.

2. **`END_MARKERS`**: Palabras clave para saber dónde termina el TP (generalmente "Bibliografía"). El script corta todo lo que viene después.

### Parámetros de Detección (en `config.py`)

Podés ajustar la sensibilidad editando `config.py`:

```python
# Modo por defecto (si solo presionás Enter)
DETECTION_MODE = "smart"

# Umbral de similitud documental (TF-IDF)
DOCUMENT_SIMILARITY_THRESHOLD = 0.70  # 70% - bajar para más sensibilidad

# Criterios de detección a nivel de oraciones
SENTENCE_MIN_EXACT_MATCHES = 5  # Mínimo de coincidencias exactas
SENTENCE_MIN_TOTAL_MATCHES = 10  # Mínimo de coincidencias totales
SENTENCE_MIN_COVERAGE = 0.08  # Mínimo 8% de cobertura

# Zona gris para modo SMART
SMART_MODE_MIN_SIMILARITY = 0.35  # Límite inferior
SMART_MODE_MAX_SIMILARITY = 0.70  # Límite superior
```

## 📋 Instrucciones de Uso

### 1. Instalación de `uv`

Si no tenés `uv` instalado:

```powershell
pip install uv # o de la forma que te complace
```

### 2. Instalación de Dependencias

```powershell
uv sync
```

### 3. Carga de Archivos

Colocá todos los trabajos prácticos que querés analizar (archivos .pdf, .docx, .md) dentro de la carpeta `files/`.

### 4. Ejecución

```powershell
uv run main.py
```

El script mostrará el menú interactivo:

- Presioná **Enter** para usar el modo por defecto (SMART)
- O escribí el **número** (1-4) o **nombre** del modo (fast/thorough/hybrid/smart)

## 📊 Interpretación de Resultados

Los resultados se muestran en consola y se guardan automáticamente en `output/results_[modo]_[fecha].txt`.

### Ejemplos de Salida

**Detección por TF-IDF:**

```
🔴 97.00% :: archivo1.pdf <--> archivo2.docx
   📄 Detectado por: TF-IDF (similitud documental)
```

**Detección por Análisis de Oraciones:**

```
🔴 Detectado :: archivo3.pdf <--> archivo4.docx
   📝 Detectado por: Análisis de oraciones
   📊 Sentence-level analysis:
      - Total matches: 12
      - Exact matches: 9
      - Coverage: 12.0%
      - Sample matches (showing top 5):
         1. [100%] "El backend permite entrar sin comprobar..."
         2. [100%] "El error de código es que la ruta quedó expuesta..."
```

### Indicadores

- **✅ Verde**: No se detectaron similitudes significativas
- **🔴 Rojo**: Par sospechoso - revisar manualmente
- **📄**: Detectado por similitud documental (TF-IDF)
- **📝**: Detectado por análisis de oraciones (plagio sofisticado)

## 🔍 Casos de Uso

### Caso 1: Primera Revisión Rápida

```
Modo: FAST
Tiempo: ~segundos
Detecta: 2 pares de copias directas
```

### Caso 2: Sospecha de Plagio Sofisticado

```
Modo: THOROUGH o SMART
Detecta: Estudiantes que copiaron pero reordenaron las respuestas
Ejemplo: Párrafos idénticos pero en diferente orden
```

### Caso 3: Análisis Final

```
Modo: HYBRID
Tiempo: ~2-3 min
Genera reporte completo con todas las métricas
```

## 🛠️ Debug Mode

Si necesitás ver los textos procesados, activá el modo debug en `constants.py`:

```python
DEBUG_MODE = True
```

Esto generará una carpeta `debug/` con el texto limpio de cada archivo.

## 📝 Nota sobre TF-IDF vs Sentence-Level

**¿Por qué dos métodos?**

- **TF-IDF** es excelente para detectar similitud general pero **falla cuando los estudiantes reordenan contenido**
- **Sentence-level** encuentra oraciones idénticas **independientemente del orden**, detectando plagio más sofisticado

El **modo SMART** combina ambos: primero filtra candidatos con TF-IDF (rápido), luego analiza oraciones en casos sospechosos (preciso).

---

*Desarrollado para facilitar la corrección y garantizar la originalidad de las entregas.*
*Sistema híbrido con detección de plagio sofisticado mediante análisis a nivel de oraciones.*
