RESULTADOS: explicación y guía práctica (4 notebooks)

Resumen rápido
- Este repositorio contiene 4 notebooks experimentales que entrenan modelos sobre el mismo conjunto de imágenes (X.npy / Y.npy) pero con propósitos distintos:
  - DedosCNN.ipynb — tarea multi‑etiqueta: predecir los 5 dedos (cada dedo es una salida binaria). Usa binary_crossentropy.
  - DedosCNN&Transfer .ipynb — variantes: CNN desde cero + versiones con transfer learning (MobileNet/EfficientNet importados). Mantén cuidado con el nombre (espacio extra en el fichero).
  - ManosCNN.ipynb — tarea multiclase (27 clases). Usa categorical_crossentropy, métricas accuracy y top‑3.
  - ManosCNN&tranfer.ipynb — transfer learning con MobileNetV2 (dos fases: feature‑extraction y fine‑tuning). Nota: "tranfer" está mal escrito en el nombre.

Valores y outputs verificables (extraídos de las ejecuciones incluidas)
- Conjunto completo: X.shape = (22801, 128, 128, 3)
- Tarea Dedos (filtrado de dígitos): N = 8650 muestras; Y_fingers shape = (8650, 5)
  - Split estratificado: Train 6058 | Val 1294 | Test 1298
  - Batches (bs=32): Train 190 | Val 41 | Test 41
  - Loss usado: binary_crossentropy; umbral por defecto para cada dedo: 0.5
  - Modelo CNN (desde cero): Total params: 323,109; Trainable: 322,213; Non‑trainable: 896

- Tarea Manos (multiclase): N total = 22801; clases = 27
  - Split estratificado: Train 15,969 | Val 3,411 | Test 3,421
  - Batches (bs=32): Train 500 | Val 107 | Test 107
  - Loss usado: categorical_crossentropy; métricas: accuracy y top3 accuracy
  - Modelo CNN (desde cero): Total params ~ 328,763; Trainable ~ 327,867; Non‑trainable 896

- Transfer learning (ManosCNN&transfer.ipynb)
  - Modelo HandSign_MobileNetV2 (include_top=False + cabezal nuevo): Total params 2,592,859
  - Fase 1 (backbone congelado): Entrenables (fase1) = 334,875; Congelados = 2,257,984
    * Mejor val_accuracy observada en Fase 1: 0.7722 (tras 10 epocas en la ejecución registrada)
  - Fase 2 (descongelar últimas 10 capas): Parâmetros entrenables al inicio fase2 ≈ 1,067,355
    * Durante el fine‑tuning la val_accuracy subió cerca de ~0.87–0.88 en los logs mostrados (mejor ~0.87599 observado en la salida)

Qué significa cada cosa (conceptos aplicados a estos notebooks)
- Train / Val / Test
  - Train: datos usados para actualizar pesos durante el fit. Alta accuracy en train indica el modelo puede representarlo.
  - Val (validation): datos usados para monitorizar generalización durante entrenamiento (early stopping, ajustar LR). Si val mejora y luego empeora: sobreajuste.
  - Test: datos separados que solo sirven para medir rendimiento final del modelo (no se deben mirar durante ajuste).

- Bias vs Variance (cómo diagnosticar con las curvas train/val)
  - Alto bias (underfitting): train loss alto y train accuracy baja, y val no mejora. Significa que el modelo no tiene capacidad suficiente o está mal parametrizado.
  - Alta varianza (overfitting): train loss bajo / train acc alto pero val loss alto / val acc baja. El modelo memoriza train y no generaliza.
  - Regla práctica para estos notebooks:
    1. Si train_acc y val_acc ambos son bajos → aumentar capacidad (más filtros, más epochs), revisar features.
    2. Si train_acc >> val_acc → aplicar regularización (dropout/weight decay), data augmentation, o usar transfer learning.
    3. Si transferencia (fase1 → fase2) mejora val_acc significativamente, es señal de que el backbone aporta features útiles.

- Loss vs Métrica
  - Loss (binary_crossentropy / categorical_crossentropy) es la función que el optimizador minimiza; no siempre correlaciona 1:1 con la métrica (accuracy) especialmente en clases desbalanceadas.
  - Para multi‑etiqueta (Dedos): se usa sigmoid + binary_crossentropy; la decisión final por dedo se toma aplicando un umbral (FINGER_THRESHOLD=0.5 en el notebook).

- Exact Match, Hamming, F1 para multi‑etiqueta
  - Exact Match Ratio (EMR): fracción de muestras donde las 5 etiquetas coinciden exactamente. Es la métrica más exigente.
  - Hamming Accuracy / Hamming Loss: mide aciertos por etiqueta (por dedo) promediados — menos estricta que EMR.
  - F1 por dedo: balance entre precision y recall para cada etiqueta binaria.

- Matriz de confusión
  - Multiclase (ManosCNN): matriz cuadrada (KxK) donde fila = clase real, columna = clase predicha.
    * Interpretación: la diagonal son aciertos; elementos fuera de diagonal indican qué clases se confunden entre sí.
    * Uso: encontrar pares de clases confusas (ej. 'hello' vs 'hi') para mejorar datos o arquitectura.
  - Multi‑etiqueta (Dedos): sklearn.multilabel_confusion_matrix devuelve, por etiqueta, una 2x2 confusion matrix (TN, FP, FN, TP).
    * Para cada dedo revisar TP/FP/FN ayuda a ver si el modelo predice demasiado conservadoramente (muchos FN) o con demasiada ambición (muchos FP).

Ejemplos de comandos / snippets útiles para reproducir los análisis
- Cargar predicciones y construir matriz de confusión (multiclase)
  - Asumiendo probs obtenidas del modelo y generador test_gen:
    ```python
    # obtener preds (probabilidades) y etiquetas reales
    probs = model.predict(test_gen, verbose=1)
    y_true = np.argmax(np.vstack([y for _, y in test_gen]), axis=1)
    y_pred = np.argmax(probs, axis=1)
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred, target_names=class_names))
    sns.heatmap(cm, fmt='d', cmap='Blues')
    ```

- Para multi‑etiqueta (Dedos): aplicar umbral y calcular métricas
  ```python
  probs = model.predict(test_gen)
  preds_bin = (probs >= 0.5).astype(int)
  y_true = np.vstack([y for _, y in test_gen])  # shape (N, 5)
  from sklearn.metrics import hamming_loss, f1_score
  exact_match = np.all(preds_bin == y_true, axis=1).mean()
  hamming = 1 - hamming_loss(y_true, preds_bin)
  f1_per_finger = f1_score(y_true, preds_bin, average=None)
  from sklearn.metrics import multilabel_confusion_matrix
  mcm = multilabel_confusion_matrix(y_true, preds_bin)
  # mcm[i] es la 2x2 confusion matrix de la etiqueta i (TN, FP, FN, TP)
  ```

Cómo interpretar logs reales incluidos (puntos clave)
- DedosCNN: se diseñó para EMR ambiciosa (umbrales: aceptable 0.85, robusto 0.92). Estos umbrales son de proyecto — verificar si son alcanzables en test.
- ManosCNN (desde cero): la arquitectura y los parámetros mostrados son razonables; hay riesgo de overfitting si val acc se separa de train acc.
- Manos + MobileNetV2 (transfer):
  - Fase 1 (solo cabezal) subió val_accuracy hasta ~0.77 — muestra que el backbone generaliza rápido.
  - Fase 2 (descongelar últimas 10 capas) aumentó val_accuracy hasta ~0.87–0.88 en los logs — fine‑tuning consiguió mejoras importantes.
  - Conclusión práctica: transfer learning con fine‑tuning mejora la generalización frente a entrenar desde cero en este dataset.

Diagnóstico rápido (lista de chequeo)
1. Comparar train vs val curves: si divergence → overfitting. Aplicar dropout/DataAug/regularización.
2. Revisar support por clase (número de ejemplos por clase). Clases con muy poco soporte suelen tener poor recall/F1.
3. Calcular y visualizar matriz de confusión (multiclase) o multilabel_confusion_matrix (por etiqueta) — priorizar correcciones en pares/classes con mayor confusión.
4. Ver curvas de precisión/recall por clase y ajustar umbral si conviene (especialmente en multi‑etiqueta). Para cada dedo quizá el 0.5 no sea óptimo.
5. En transfer learning: si fase1 ya da buen val_acc, el fine‑tuning selectivo (pocas capas y LR bajo) suele dar el mayor beneficio.

Recomendaciones prácticas (siguientes pasos)
- Genera y añade al notebook estas visualizaciones al final del entrenamiento:
  - Curvas train/val para loss y accuracy
  - Confusion matrix (y normalized confusion matrix)
  - Tabla de precision/recall/F1 y support por clase (classification_report)
  - Para multi‑etiqueta: TP/FP/FN por etiqueta y curva PR por etiqueta
- Experimentos que suelen ayudar:
  - Aumentar augmentations (rotaciones, recortes) si el modelo sobreajusta.
  - Usar class weights o oversampling para clases raras (manos multiclase).
  - Hacer grid de umbrales por etiqueta para maximizar F1 o EMR en Dedos.

Notas operativas que no debes olvidar
- Siempre citar los splits exactos cuando reportes resultados (ya están en los notebooks). No reutilices test para tomar decisiones de modelado.
- Al renombrar archivos con espacios o corregir typos, crea copias y conserva los originales a menos que quieras reemplazarlos explícitamente.

Si quieres, puedo:
1) Generar automáticamente los reports (confusion matrix, classification_report, curvas) y añadir celdas a cada notebook para que queden reproducibles.
2) Ejecutar los notebooks (si me confirmas que quieres que intente instalar dependencias y descargar los .npy desde Drive). Esto requiere GPU/tiempo.

Fin del informe.
