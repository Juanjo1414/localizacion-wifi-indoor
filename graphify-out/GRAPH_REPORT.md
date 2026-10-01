# Graph Report - Proyecto ML - Localización WiFi indoor (UJIIndoorLoc)  (2026-10-01)

## Corpus Check
- Corpus is ~19,019 words - fits in a single context window. You may not need a graph.

## Summary
- 76 nodes · 128 edges · 16 communities (13 shown, 3 thin omitted)
- Extraction: 85% EXTRACTED · 14% INFERRED · 1% AMBIGUOUS · INFERRED: 18 edges (avg confidence: 0.79)
- Token cost: 568,622 input · 0 output

## Community Hubs (Navigation)
- Metadatos y fuga de datos
- Notebooks de referencia y pipeline
- Dataset y validez del proyecto
- Reglas del proyecto
- Mapa de pisos, edificio 0
- App y artefactos de rasgos
- Modelo final y F1 macro
- Correlacion entre WAPs
- Distribucion de clases en test
- Mapa de edificios
- Dispersion de senales por fila
- Distribucion de clases en entrenamiento
- Distribucion de intensidad RSSI
- Stack tecnologico y dependencias
- Paquete del proyecto

## God Nodes (most connected - your core abstractions)
1. `PLAN.md (plan de implementación)` - 29 edges
2. `02_PCA_Seleccion_Rasgos.ipynb` - 11 edges
3. `01_EDA_Preprocesamiento.ipynb` - 10 edges
4. `03_Modelos_Evaluacion.ipynb` - 10 edges
5. `Heatmap de correlación Phik entre BF y metadatos` - 10 edges
6. `app.py (app Gradio)` - 8 edges
7. `CLAUDE.md (instrucciones del proyecto)` - 7 edges
8. `Celda de configuración EN_COLAB / RUTA_BASE` - 6 edges
9. `Cronograma de 12 días` - 5 edges
10. `Pipeline de Machine Learning (diagrama mermaid)` - 5 edges

## Surprising Connections (you probably didn't know these)
- `Limitaciones conocidas (README)` --semantically_similar_to--> `Riesgo de validación cruzada demasiado optimista`  [INFERRED] [semantically similar]
  README.md → PLAN.md
- `Librerías del proyecto (gestionadas con uv)` --semantically_similar_to--> `Stack tecnológico (README)`  [INFERRED] [semantically similar]
  PLAN.md → README.md
- `CLAUDE.md (instrucciones del proyecto)` --references--> `Celda de configuración EN_COLAB / RUTA_BASE`  [EXTRACTED]
  CLAUDE.md → PLAN.md
- `PLAN.md (plan de implementación)` --references--> `Forma de trabajar (CLAUDE.md)`  [EXTRACTED]
  PLAN.md → CLAUDE.md
- `PLAN.md (plan de implementación)` --references--> `Reglas de código (CLAUDE.md)`  [EXTRACTED]
  PLAN.md → CLAUDE.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Flujo secuencial del pipeline: EDA -> PCA/selección -> modelos -> app** — plan_notebook_01, plan_notebook_02, plan_notebook_03, plan_app_py [EXTRACTED 0.90]
- **Artefactos .joblib persistidos que alimentan la app** — plan_label_encoder_joblib, plan_rasgos_usados_joblib, plan_modelo_final_joblib [EXTRACTED 0.90]
- **Notebooks de clase usados como guía de estilo para los tres notebooks del proyecto** — plan_ia_ml_example_dif_classifiers_cv, plan_ml_normalizacion, plan_pca_wine, plan_ml_featureselection_shap [EXTRACTED 0.85]
- **Visualización EDA de edificios por geolocalización y BUILDINGID** — figuras_mapa_edificios_image, figuras_mapa_edificios_concept_buildingid, figuras_mapa_edificios_concept_geolocalizacion [INFERRED 0.75]

## Communities (16 total, 3 thin omitted)

### Community 0 - "Metadatos y fuga de datos"
Cohesion: 0.30
Nodes (11): Variable objetivo BF (edificio+piso), BUILDINGID, FLOOR, Heatmap de correlación Phik entre BF y metadatos, LATITUDE, LONGITUDE, PHONEID, RELATIVEPOSITION (+3 more)

### Community 1 - "Notebooks de referencia y pipeline"
Cohesion: 0.31
Nodes (7): IA_ML_Example_Dif_Classifiers_CV.ipynb (notebook de referencia), ML_FeatureSelection_comparacion_metodos_SHAP.ipynb (notebook de referencia), ML_Normalización.ipynb (notebook de referencia), 01_EDA_Preprocesamiento.ipynb, 02_PCA_Seleccion_Rasgos.ipynb, PCA_Wine.ipynb (notebook de referencia), Pipeline de Machine Learning (diagrama mermaid)

### Community 2 - "Dataset y validez del proyecto"
Cohesion: 0.32
Nodes (6): Dataset UJIIndoorLoc (UCI id 310), Función eval_cv (StratifiedKFold + cross_validate), Torres-Sospedra et al. (2014) - UJIIndoorLoc paper (IPIN), Equipo del proyecto, Limitaciones conocidas (README), Mejoras futuras (README)

### Community 3 - "Reglas del proyecto"
Cohesion: 0.60
Nodes (3): CLAUDE.md (instrucciones del proyecto), PLAN.md (plan de implementación), Variable objetivo BF (edificio + piso)

### Community 4 - "Mapa de pisos, edificio 0"
Cohesion: 0.50
Nodes (5): Análisis exploratorio espacial (longitud/latitud), Dispersión de puntos de medición por piso, Edificio 0 (BUILDINGID=0), Variable FLOOR (piso), Mapa de pisos edificio 0 (figura)

### Community 5 - "App y artefactos de rasgos"
Cohesion: 0.40
Nodes (4): app.py (app Gradio), Cronograma de 12 días, label_encoder.joblib, rasgos_usados.joblib

### Community 6 - "Modelo final y F1 macro"
Cohesion: 0.40
Nodes (3): Celda de configuración EN_COLAB / RUTA_BASE, modelo_final.joblib, 03_Modelos_Evaluacion.ipynb

### Community 7 - "Correlacion entre WAPs"
Cohesion: 0.50
Nodes (4): Correlación top 20 WAPs (figura), Mapa de calor de correlación entre WAPs, Redundancia/colinealidad entre señales WiFi vecinas, 20 WAPs con más detecciones

### Community 8 - "Distribucion de clases en test"
Cohesion: 0.67
Nodes (4): Gráfico: Distribución de BF en el set de prueba, Variable BF (edificio_piso), Desbalance de clases entre pisos/edificios, Conjunto de prueba (test set)

### Community 9 - "Mapa de edificios"
Cohesion: 0.67
Nodes (4): Variable BUILDINGID, Análisis exploratorio de datos (EDA) del dataset UJIIndoorLoc, Geolocalización de puntos de medición (Latitud/Longitud), Mapa de edificios (dispersión lat/long)

### Community 10 - "Dispersion de senales por fila"
Cohesion: 0.67
Nodes (3): Dispersión/sparsity de las señales WiFi (520 WAPs), Distribución de cantidad de WAPs detectados por fila, Histograma: WAPs detectados por fila

### Community 11 - "Distribucion de clases en entrenamiento"
Cohesion: 0.67
Nodes (3): Desbalance de clases en datos de entrenamiento, Distribución de clases BF (edificio_piso), Gráfico: Distribución de BF en entrenamiento

### Community 12 - "Distribucion de intensidad RSSI"
Cohesion: 0.67
Nodes (3): Análisis exploratorio de datos (EDA) de señales WiFi, Distribución de intensidad de señal WiFi (RSSI en dBm), Histograma: Distribución de las señales detectadas

## Ambiguous Edges - Review These
- `requirements.txt` → `Librerías del proyecto (gestionadas con uv)`  [AMBIGUOUS]
  PLAN.md · relation: shares_data_with

## Knowledge Gaps
- **18 isolated node(s):** `localizacion-wifi-indoor`, `Equipo del proyecto`, `TIMESTAMP`, `USERID`, `PHONEID` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 19 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `requirements.txt` and `Librerías del proyecto (gestionadas con uv)`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **Why does `PLAN.md (plan de implementación)` connect `Reglas del proyecto` to `Notebooks de referencia y pipeline`, `Dataset y validez del proyecto`, `App y artefactos de rasgos`, `Modelo final y F1 macro`, `Stack tecnologico y dependencias`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **Why does `Riesgo de validación cruzada demasiado optimista` connect `Dataset y validez del proyecto` to `Reglas del proyecto`?**
  _High betweenness centrality (0.007) - this node is a cross-community bridge._
- **What connects `localizacion-wifi-indoor`, `Equipo del proyecto`, `TIMESTAMP` to the rest of the system?**
  _18 weakly-connected nodes found - possible documentation gaps or missing edges._