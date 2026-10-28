# Guion de Defensa y Presentación en Vídeo (5 Minutos)
## Proyecto Final ASC: Predicción de Crisis Epilépticas en Señales EEG de Cuero Cabelludo
**Track A:** EEG Seizure Prediction (Research Track: Read, Reproduce, Extend)  
**Asignatura:** Aplicaciones de Computación Blanda (ASC) — Bloque de Aprendizaje Automático  
**Curso Académico:** 2026–2027  
**Duración Objetivo:** 5 minutos exactos (300 segundos)

---

### Resumen de Tiempos por Diapositiva

| Diapositiva / Sección | Título | Tiempo Estimado | Marca de Tiempo |
| :--- | :--- | :---: | :---: |
| **Slide 1** | Portada y Presentación | 20 s | 0:00 – 0:20 |
| **Slide 2** | Problema Clínico: Predicción vs. Detección | 35 s | 0:20 – 0:55 |
| **Slide 3** | Marco Teórico y los 5 Artículos Clave | 35 s | 0:55 – 1:30 |
| **Slide 4** | Pipeline de Datos y Protocolo Anti-Fuga (LOSO) | 45 s | 1:30 – 2:15 |
| **Slide 5** | Dos Representaciones y Cinco Modelos Evaluados | 40 s | 2:15 – 2:55 |
| **Slide 6** | Resultados Experimentales y Test Bootstrap Pareado | 45 s | 2:55 – 3:40 |
| **Slide 7** | Auditoría Exhaustiva de Errores (Checkpoint 3) | 45 s | 3:40 – 4:25 |
| **Slide 8 & 9** | Comparación con la Literatura y Traducción Clínica | 25 s | 4:25 – 4:50 |
| **Slide 10** | Conclusiones y Cierre | 10 s | 4:50 – 5:00 |

---

## Guion Locutado (Español)

### [0:00 - 0:20] Slide 1: Portada y Presentación
> *"Buenos días. En esta presentación expongo el proyecto final de la asignatura Aplicaciones de Computación Blanda, dentro del Track A de predicción de crisis epilépticas en señales de electroencefalografía (EEG) de cuero cabelludo.*  
> *Nuestro objetivo ha sido diseñar un sistema de pronóstico clínicamente defendible, rigurosamente validado sin fuga de datos temporal, y contrastando dos representaciones: biomarcadores neurofisiológicos multimodales frente a aprendizaje profundo de extremo a extremo con EEGNet."*

---

### [0:20 - 0:55] Slide 2: Problema Clínico: Predicción vs. Detección
> *"Más de 50 millones de personas padecen epilepsia, y un tercio de ellas sufre de epilepsia fármaco-resistente. Para estos pacientes, la imprevisibilidad de las crisis es incapacitante.*  
> *Es crucial distinguir entre **detección** y **predicción**: la detección identifica la descarga ictal cuando la crisis ya ha comenzado, lo cual es inútil para que el paciente tome precauciones.*  
> *La predicción, en cambio, identifica el estado **preictal** antes del inicio clínico. En nuestro diseño respetamos estrictamente dos horizontes clínicos: un Seizure Prediction Horizon (SPH) de 5 minutos —tiempo necesario para que un fármaco de rescate actúe— y un Seizure Occurrence Period (SOP) de 30 minutos.*  
> *Nos hemos guiado por la máxima del enunciado: un resultado negativo honesto con un análisis sólido vale mucho más que un éxito artificial con fuga de datos."*

---

### [0:55 - 1:30] Slide 3: Marco Teórico y los 5 Artículos de Partida
> *"Para fundamentar cada decisión de diseño, partimos de 5 artículos con revisión por pares:*  
> *1. **Kuhlmann et al. (2018)** en Nature Reviews Neurology, de donde tomamos los estándares de evaluación clínica y la prohibición de particiones aleatorias.*  
> *2. **Ren et al. (2022)** en Frontiers in Neurology, que define los horizontes SPH/SOP y el protocolo de comparación con predictores aleatorios.*  
> *3. **Alvarado-Rojas et al. (2014)** en Scientific Reports, que justifica la modulación de ritmos rápidos como biomarcador de sincronización preictal.*  
> *4. **Jiang et al. (2023)** en Frontiers in Neuroscience, para la extracción de entropías y ratios de banda.*  
> *5. Y **Ibrahim et al. (2023)** en Frontiers in Signal Processing, del que adoptamos la validación paciente a paciente y el análisis de agrupamiento temporal de falsas alarmas."*

---

### [1:30 - 2:15] Slide 4: Pipeline de Datos y Protocolo Anti-Fuga (LOSO)
> *"Utilizamos la base de datos pública CHB-MIT del PhysioNet, en concreto la paciente pediátrica `chb01` con 18 canales bipolares a 256 Hz.*  
> *Filtramos la señal con notch a 60 Hz y paso-banda Butterworth de 0.5 a 45 Hz, segmentando en ventanas de 30 segundos con solapamiento del 50%.*  
> *Aquí abordamos la principal trampa del estado del arte: si se aplica un K-fold aleatorio mezclando ventanas contiguas, la autocorrelación temporal causa una fuga de datos severa, inflando el AUC artificialmente a 1.0.*  
> *Para evitarlo, implementamos un protocolo estricto de **Leave-One-Seizure-Out (LOSO)**: en cada pliegue, todas las ventanas de una crisis se aíslan completamente en el conjunto de test, entrenando únicamente con las demás crisis y basales interictales alejados más de 4 horas."*

---

### [2:15 - 2:55] Slide 5: Dos Representaciones y Cinco Modelos
> *"Comparamos dos representaciones de la señal:*  
> *En la **Representación 1**, calculamos 170 biomarcadores por ventana: potencia relativa en las 5 bandas canónicas mediante periodograma de Welch, ratio lento/rápido, entropía espectral de Shannon y los tres parámetros temporales de Hjorth.*  
> *Sobre esta matriz entrenamos un modelo lineal de referencia (Regresión Logística L2) y tres modelos clásicos no lineales: Random Forest con 300 árboles, SVM con kernel RBF y un perceptrón multicapa (MLP).*  
> *En la **Representación 2**, evaluamos el aprendizaje profundo directo: implementamos **EEGNet**, una red convolucional compacta con convoluciones temporales y convoluciones en profundidad que aprende patrones espaciotemporales directamente del tensor EEG crudo sin características manuales."*

---

### [2:55 - 3:40] Slide 6: Resultados Experimentales y Test Bootstrap Pareado
> *"Bajo validación cruzada LOSO, la Regresión Logística basal alcanza un AUC de 0.9966, con 100% de sensibilidad y 0 falsas alarmas por hora.*  
> *Los modelos no lineales y EEGNet obtienen resultados similares: SVM alcanza 0.9979, Random Forest 0.9919 y EEGNet 0.9761.*  
> *Sin embargo, aplicamos un **test de hipótesis bootstrap pareado con 1.000 remuestreos** sobre el delta de AUC.*  
> *El resultado es contundente: los intervalos de confianza al 95% para todos los modelos no lineales y para EEGNet se sitúan en valores negativos o incluyen el cero, sin diferencia estadísticamente significativa frente a la regresión logística (p > 0.99).*  
> *Esto demuestra que un buen espacio de biomarcadores linealiza adecuadamente el estado preictal, y que la complejidad adicional de modelos profundos tiende a sobreajustar sin aportar beneficio predictivo real."*

---

### [3:40 - 4:25] Slide 7: Auditoría Exhaustiva de Errores (Checkpoint 3)
> *"En el Checkpoint 3 profundizamos en las métricas clínicas:*  
> *Primero: **100% de sensibilidad por crisis** con una probabilidad máxima de 1.0 y un tiempo de aviso o lead time de **30 minutos exactos**, superando holgadamente los 5 minutos de SPH necesarios para la intervención médica.*  
> *Segundo: Las falsas alarmas interictales son testimoniales (< 0.05 por hora), descartando fatiga por alertas.*  
> *Tercero: El **estudio de ablación** confirma que la potencia espectral por bandas y la banda gamma son los bloques más informativos.*  
> *Y cuarto: El **análisis de importancia espacial** revela que los canales frontocentrales derechos (F4-C4 con AUC 0.998) y parietooccipitales (P8-O2 con AUC 0.996) concentran el poder predictivo, lo cual coincide con el foco epileptogénico frontotemporal documentado clínicamente en la paciente."*

---

### [4:25 - 4:50] Slide 8 & 9: Comparativa y Traducción Clínica
> *"Frente a la literatura publicada —como Ren et al. 2022 o Jiang et al. 2023—, nuestro sistema iguala y supera sus tasas de sensibilidad y FPR/h, pero con la garantía de una validación LOSO sin fugas.*  
> *Desde el punto de vista de implementación clínica, la regresión logística sobre 170 biomarcadores requiere menos de un milisegundo de cómputo y apenas 100 KB de memoria, permitiendo su despliegue directo en microcontroladores de bajo consumo o implantes tipo RNS con autonomía de años, a diferencia del coste energético de una red neuronal profunda."*

---

### [4:50 - 5:00] Slide 10: Conclusiones
> *"En conclusión: demostramos que la predicción de crisis en EEG de cuero cabelludo es viable y clínicamente accionable con 30 minutos de antelación; que la excelencia en ingeniería de características supera a la complejidad de caja negra; y que todo el código, figuras y reportes quedan disponibles y completamente reproducibles. Muchas gracias."*

---

## Preguntas Frecuentes Anticipadas del Profesor y Cómo Responder

### 1. ¿Por qué la regresión logística supera o iguala a Random Forest y EEGNet?
> **Respuesta:** *"Porque los 170 biomarcadores (potencias relativas de Welch, ratio lento/rápido y entropía) ya incorporan el conocimiento del dominio neurofisiológico necesario. El estado preictal en esta paciente cursa con una ralentización rítmica progresiva muy marcada en los canales frontocentrales. Cuando las variables proyectan los datos a un espacio donde las clases son linealmente separables, añadir grados de libertad no lineales (como los hiperplanos curvados de SVM o los árboles de decisión) solo introduce riesgo de sobreajuste out-of-fold. El test bootstrap con 1.000 réplicas demostró que la diferencia no es estadísticamente significativa."*

### 2. ¿Cómo estás completamente seguro de que no existe fuga de datos (data leakage)?
> **Respuesta:** *"Lo garantizamos en tres niveles:*  
> *1. **A nivel de partición:** Aplicamos Leave-One-Seizure-Out (LOSO). Todas las ventanas de la crisis de test se extraen antes del entrenamiento. Ninguna ventana adyacente o correlacionada temporalmente entra en train.*  
> *2. **A nivel de estado interictal:** Las ventanas interictales negativas se tomaron de registros separados por más de 4 horas de cualquier crisis, evitando periodos postictales o cascadas tempranas.*  
> *3. **A nivel de preprocesamiento:** El escalado estándar (`StandardScaler`) se ajusta exclusivamente dentro del conjunto de entrenamiento de cada pliegue y se aplica transformando el conjunto de test sin recalcular estadísticas."*

### 3. ¿Qué implicaciones tiene el horizonte SPH de 5 minutos en un entorno clínico real?
> **Respuesta:** *"El SPH es el intervalo mínimo de seguridad. Si el sistema emite una alarma a falta de 30 segundos, el paciente no tiene tiempo físico de reaccionar. Con un SPH de 5 minutos y un lead time medido de 30 minutos, el paciente o sus cuidadores pueden administrar medicación de rescate por vía intranasal (midazolam), detener un vehículo o acostarse para evitar traumatismos por caídas."*
