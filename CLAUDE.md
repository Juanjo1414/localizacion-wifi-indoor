# Proyecto: Localización WiFi indoor (UJIIndoorLoc) - Fundamentos de IA, EIA

## Contexto
- Clasificación de edificio + piso combinados (variable BF, ~13 clases) a partir de 520 señales WiFi.
- El plan completo está en PLAN.md. Seguirlo en orden y no saltarse pasos.
- Los notebooks de clase están en referencia/. Imitar su estructura, nombres y forma de escribir.

## Reglas de código
- Código de nivel básico-intermedio, como en los notebooks de referencia.
- NO usar list comprehensions, dict comprehensions, generadores ni lambdas. Usar ciclos for simples.
- Usar Pipeline, StratifiedKFold, GridSearchCV y joblib como en clase.
- La primera celda de cada notebook es la de configuración con EN_COLAB y RUTA_BASE.
- Nunca usar LONGITUDE, LATITUDE, BUILDINGID, FLOOR, SPACEID, RELATIVEPOSITION, USERID, PHONEID ni TIMESTAMP como rasgos.
- validationData.csv / test_procesado.csv solo se usa en la sección de evaluación final del notebook 03.
- random_state=42 en todo.

## Reglas de escritura
- Todo en español.
- Celdas markdown cortas, en voz grupal ("probamos", "notamos", "decidimos"), naturales, como las escribiría un estudiante.
- Nada de tablas en markdown: explicar los resultados en prosa.

## Forma de trabajar
- Si algo no está claro o un resultado no cuadra con lo esperado en PLAN.md, preguntar antes de seguir. No suponer.
- Después de editar un notebook, ejecutarlo completo para verificar que corre.
- El entorno se administra con uv (`pyproject.toml` + `uv.lock` + `.python-version`, Python 3.11). Nunca usar pip directamente; para correr código usar `uv run ...` (por ejemplo `uv run jupyter nbconvert ...`) y para instalar algo nuevo preguntar primero y luego usar `uv add nombre-del-paquete`.