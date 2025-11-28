# 🕵️‍♂️ Detector de Copias de TPs (Anti-Machete)

Bienvenido al **Detector de Copias**. Este script es una herramienta diseñada para identificar trabajos prácticos con similitudes sospechosas. Es ideal para analizar lotes de entregas y detectar posibles casos de plagio o "inspiración compartida" no atribuida.

## ¿Cómo funciona?

El script procesa todos los archivos ubicados en la carpeta `files/` (soporta formatos PDF, DOCX y MD) y los compara entre sí utilizando técnicas de procesamiento de lenguaje natural (TF-IDF + Similitud del Coseno).

El análisis está optimizado para evitar falsos positivos mediante los siguientes criterios:

* **Filtrado de palabras comunes**: Ignora conectores y palabras funcionales ("de", "la", "que", "el", etc.) para centrarse en el contenido relevante.
* **Análisis de frases (N-gramas)**: Utiliza secuencias de 1 a 3 palabras para detectar coincidencias en oraciones completas, no solo en vocabulario aislado.
* **Limpieza de estructura**: Elimina consignas repetidas (presentes en todos los TPs) y recorta las secciones de bibliografía para no afectar la comparación.

## Configuración Importante

Antes de correr el script, abrí el archivo `main.py` y fijate en estas dos listas al principio, que **son específicas para cada Trabajo Práctico**:

1. **`COMMON_PHRASES`**: Acá tenés que poner las frases que se repiten en *todos* los trabajos (como las consignas, el nombre de la materia, etc.). Si no las ponés, el script va a pensar que se copiaron porque todos tienen el mismo texto de las preguntas.
2. **`END_MARKERS`**: Son las palabras clave para saber dónde termina el TP (generalmente "Bibliografía"). El script corta todo lo que viene después de esto para no comparar autores citados.

## Instrucciones de Uso

### 1. Configuración del Entorno

Este proyecto utiliza `uv` para la gestión de dependencias, lo que lo hace mucho más rápido y confiable.

Si no tenés `uv` instalado, podés instalarlo con pip:

```powershell
pip install uv
```

### 2. Instalación de Dependencias

Para instalar todas las librerías necesarias, simplemente ejecutá:

```powershell
uv sync
```

### 3. Carga de Archivos

Colocá todos los trabajos prácticos que querés analizar (archivos .pdf, .docx, .md) dentro de la carpeta `files`.

### 4. Ejecución

Corré el script principal usando `uv`:

```powershell
uv run main.py
```

## Interpretación de Resultados

El script mostrará los resultados en la consola y generará un reporte detallado en el archivo `resultados.txt`.

* **Verde (✅)**: No se detectaron similitudes significativas.
* **Rojo (🔴)**: Se encontraron pares de archivos con un alto porcentaje de similitud. Se recomienda revisar estos casos manualmente.

Adicionalmente, si la opción `DEBUG_MODE` está activada (`True`), se generará una carpeta `debug/` con el texto procesado de cada archivo. **Por defecto está desactivado** para no llenar el disco de archivos temporales.

---
*Desarrollado para facilitar la corrección y garantizar la originalidad de las entregas.*
