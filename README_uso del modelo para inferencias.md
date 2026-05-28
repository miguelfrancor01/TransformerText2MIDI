

# Text2midi — Generación de Música Simbólica desde Descripciones Textuales

Este proyecto implementa el modelo **Text2midi** propuesto por Bhandari et al. (2024) para inferencia local, permitiendo generar archivos MIDI a partir de descripciones en lenguaje natural mediante una arquitectura encoder-decoder de extremo a extremo.

---

## 1. Resumen

A pesar del auge de los modelos generativos en los últimos años, la generación de música simbólica multiinstrumental había sido un campo poco explorado, en parte por la ausencia de datasets a gran escala que relacionaran archivos MIDI con descripciones textuales ricas. Text2midi aborda este problema combinando un encoder de lenguaje preentrenado con un decoder Transformer autoregresivo, permitiendo que usuarios técnicos y no técnicos generen piezas musicales completas a partir de instrucciones en lenguaje natural.

La arquitectura utiliza **FLAN-T5 base** como encoder, cuyos 113 millones de parámetros permanecen congelados durante toda la inferencia, produciendo representaciones semánticas del caption de texto que condicionan al decoder a través de un mecanismo de cross-attention. El decoder, compuesto por 18 capas Transformer con 159 millones de parámetros entrenables, genera secuencias de hasta 2048 tokens en formato REMI+, una representación simbólica que codifica eventos musicales como notas, velocidades, duraciones e instrumentos de forma discreta, permitiendo aplicar directamente la infraestructura de modelos de lenguaje al dominio musical.

Entre las innovaciones del modelo se destacan el uso de **Flash Attention** para el procesamiento eficiente de secuencias largas y el **sentence omission** como técnica de regularización durante el fine-tuning, que mejora la robustez del modelo ante descripciones incompletas o informales.

**Palabras clave:** generación de música simbólica, arquitectura encoder-decoder, REMI+, Transformer autoregresivo, text-to-music.

---

## 2. Introducción

Este proyecto está basado en el artículo *"Text2midi: Generating Symbolic Music from Captions"*.

- **Paper:** https://arxiv.org/pdf/2412.16526
- **Repositorio original:** https://github.com/AMAAI-Lab/Text2midi

### 2.1 Contexto

El avance acelerado de los modelos de lenguaje de gran escala ha permitido desarrollar sistemas capaces de generar contenido multimodal a partir de descripciones textuales, incluyendo texto, imágenes y audio. Sin embargo, dentro del dominio de la música generativa, la atención se ha concentrado predominantemente en la generación de audio directamente desde texto, dejando de lado la música simbólica en formato MIDI, que constituye una representación fundamental en la creación musical profesional.

Esta brecha es significativa por varias razones. En primer lugar, el formato MIDI es ampliamente utilizado por compositores y productores musicales en estaciones de trabajo de audio digital (DAWs), ya que permite editar, transponer y manipular la música de forma estructurada, algo que no es posible con archivos de audio. En segundo lugar, la ausencia de modelos texto-a-MIDI puede atribuirse en gran medida a la falta histórica de datasets a gran escala que relacionen archivos MIDI con descripciones textuales ricas.

El único antecedente relevante antes de este trabajo es **MuseCoco** (Lu et al., 2023), cuyo enfoque de dos etapas —primero extrayendo atributos musicales del texto y luego generando música a partir de dichos atributos— introduce complejidad computacional considerable tanto en entrenamiento como en inferencia, además de restringir la flexibilidad del usuario al requerir descripciones estructuradas.

### 2.2 Motivación

La música es una de las formas de expresión humana más universales; sin embargo, su creación formal ha estado históricamente restringida a quienes poseen conocimientos técnicos en teoría musical, composición o manejo de herramientas especializadas. Esta barrera ha limitado la participación de un amplio grupo de personas que, aunque tienen ideas musicales claras, carecen de los medios para materializarlas.

El surgimiento de los modelos de lenguaje de gran escala y los avances en inteligencia artificial generativa han abierto una nueva posibilidad: permitir que cualquier persona describa en lenguaje natural la música que desea crear y obtenga como resultado un archivo musical editable y de alta calidad. En este contexto, el formato MIDI cobra especial relevancia: a diferencia del audio, un archivo MIDI no es una grabación sino una partitura digital que puede ser modificada, instrumentada y adaptada según las necesidades del compositor.

La motivación central de este proyecto radica en explorar y comprender cómo una arquitectura encoder-decoder, apoyada en modelos de lenguaje preentrenados como FLAN-T5 y en representaciones simbólicas de música como REMI+, puede cerrar la brecha entre la intención textual del usuario y la generación musical simbólica.

### 2.3 Objetivo

Desarrollar el primer modelo end-to-end que permita la generación de archivos MIDI a partir de descripciones textuales brindadas por usuarios técnicos y no técnicos, haciendo uso de un encoder preentrenado que codifica las descripciones textuales y un decoder Transformer autoregresivo para generar las correspondientes secuencias MIDI, mediante una estrategia de entrenamiento semi-supervisado sobre el dataset SymphonyNet y fine-tuning sobre MidiCaps.

---

## 3. Marco Teórico

### 3.1 Arquitectura del Encoder — FLAN-T5 base (frozen)

**1. Text caption (entrada)**
El proceso comienza con una descripción textual en lenguaje natural escrita por el usuario, por ejemplo: *"A cheerful piano piece in C major, 4/4, Allegro tempo"*. Esta descripción es simplemente una cadena de texto sin ningún procesamiento previo.

**2. Tokenización SentencePiece**
El texto se convierte en una secuencia de números enteros mediante el algoritmo SentencePiece, que divide el texto en subpalabras pertenecientes a un vocabulario fijo de 32.128 tokens. Cada palabra o fragmento de palabra se mapea a un ID numérico único. **Entrada:** texto en lenguaje natural. **Salida:** secuencia de enteros de longitud n, donde n depende de la longitud del caption.

**3. Input Embedding**
Cada ID numérico se convierte en un vector denso de 768 dimensiones mediante una tabla de embeddings aprendida durante el preentrenamiento. Este vector captura el significado semántico de cada token en un espacio matemático continuo. **Entrada:** secuencia de n enteros. **Salida:** matriz de n × 768.

**4. Relative Position Bias**
A diferencia del Transformer original que usa codificación posicional fija basada en senos y cosenos, FLAN-T5 usa un sesgo de posición relativa aprendido. Este sesgo no se suma a los embeddings sino que se agrega directamente a los scores de atención para indicar la distancia relativa entre pares de tokens. Usa 32 buckets para agrupar diferentes distancias. **Entrada:** n × 768. **Salida:** n × 768.

#### Bloque × 12 capas — dimensión n × 768 constante

**5. Dropout (0.1) — entrada del bloque**
Al inicio de cada bloque, el 10% de las activaciones se apagan aleatoriamente durante el entrenamiento. Esto previene que el modelo dependa excesivamente de características específicas desde el inicio de cada capa, mejorando su capacidad de generalización. **Entrada / Salida:** n × 768.

**6. Norm — Pre-LN (RMSNorm, sin additive bias)**
Antes de pasar por la self-attention, la secuencia se normaliza usando RMSNorm, una versión simplificada de la Layer Normalization que únicamente reescala las activaciones dividiéndolas por su magnitud, sin sumar ningún término de desplazamiento (bias). Esto estabiliza los valores antes de la operación de atención. **Entrada / Salida:** n × 768.

**7. Multi-Head Self-Attention (12 cabezas, d_k = 64)**
Cada token de la secuencia atiende a todos los demás tokens simultáneamente mediante 12 cabezas de atención paralelas. Cada cabeza trabaja en un subespacio de 64 dimensiones (768 / 12 = 64). En cada cabeza se calculan los vectores Query, Key y Value, se computan los scores de atención escalados y se pondera la información de todos los tokens. Los resultados de las 12 cabezas se concatenan y se proyectan de vuelta a 768 dimensiones. Adicionalmente, el Relative Position Bias se suma a los scores de atención de cada cabeza para incorporar información posicional. **Entrada / Salida:** n × 768.

**8. Add residual + Dropout (0.1)**
La entrada original que llegó al bloque de normalización se suma con la salida de la self-attention. Esta conexión residual garantiza que el gradiente pueda fluir directamente hacia capas anteriores sin degradarse, evitando el problema del vanishing gradient en redes profundas. Antes de realizar la suma, se aplica Dropout del 10% sobre la salida de la atención. **Entrada:** n × 768 (residual) + n × 768 (salida atención). **Salida:** n × 768.

**9. Norm — Pre-LN (RMSNorm, sin additive bias)**
Se aplica nuevamente RMSNorm antes de la red Feed-Forward, siguiendo el mismo principio del paso 6. **Entrada / Salida:** n × 768.

**10. Feed-Forward (Gated-GELU) 768 → 2048 → 768**
Cada token pasa de forma independiente por una red neuronal de dos capas. La primera proyección expande la dimensión de 768 a 2048. A diferencia del Transformer original que usa ReLU, FLAN-T5 usa la activación Gated-GELU: hay dos proyecciones lineales en paralelo, una pasa por la función GELU y la otra actúa como puerta multiplicativa, controlando cuánta información fluye. El resultado se proyecta de vuelta a 768 dimensiones. **Entrada:** n × 768. **Dimensión interna:** n × 2048. **Salida:** n × 768.

**11. Add residual + Dropout (0.1)**
La entrada que llegó al bloque de normalización del paso 9 se suma con la salida de la red Feed-Forward. El Dropout del 10% se aplica sobre la salida de la FFN antes de sumar el residual. **Entrada:** n × 768 (residual) + n × 768 (salida FFN). **Salida:** n × 768.

**12. Final Layer Norm (RMSNorm)**
Después de las 12 capas apiladas, se aplica una normalización final sobre toda la representación acumulada. Esta capa garantiza que los hidden states que se enviarán al decoder tengan una magnitud controlada y compatible con lo que el decoder espera recibir en su mecanismo de cross-attention. **Entrada / Salida:** n × 768.

**13. Dropout (0.1) — salida del stack**
Un último Dropout del 10% se aplica sobre la salida completa del encoder antes de entregarla al decoder. **Entrada / Salida:** n × 768.

**14. Hidden States HT — shape (n, 768)**
El resultado final del encoder es una matriz de n filas y 768 columnas, donde cada fila es un vector que representa la comprensión semántica de cada token del caption original. Este tensor HT se envía al decoder donde será usado como Key y Value en el mecanismo de cross-attention, condicionando toda la generación musical. **Dimensión final:** n × 768.

### 3.2 Arquitectura del Decoder — Transformer (18 capas, entrenable)

**1. MIDI tokens — shifted right (entrada)**
La entrada al decoder durante el entrenamiento es la secuencia de tokens MIDI desplazada una posición a la derecha (shifted right). Esto significa que en cada posición el modelo recibe el token anterior como entrada y debe predecir el token siguiente. El desplazamiento se logra agregando un token especial de inicio (BOS) al comienzo de la secuencia. Durante la inferencia, el decoder comienza únicamente con el token BOS y va construyendo la secuencia token a token de manera autoregresiva. **Entrada:** secuencia de 2048 enteros.

**2. Tokenización REMI+**
Los eventos musicales del archivo MIDI son representados como una secuencia de tokens discretos mediante el esquema REMI+. Cada nota, instrumento, posición temporal, tempo y compás se convierte en un token específico del vocabulario REMI+. El vocabulario incluye tokens de tipo Bar, Position, Tempo, Program, Pitch, Velocity, Duration y TimeSignature, además de tokens especiales como BOS, EOS y PAD. **Entrada:** secuencia de 2048 enteros. **Salida:** secuencia de 2048 IDs del vocabulario REMI+.

**3. Input Embedding**
Cada ID del vocabulario REMI+ se convierte en un vector denso de 768 dimensiones mediante una tabla de embeddings aprendida durante el entrenamiento. A diferencia del encoder donde los embeddings provienen de FLAN-T5 preentrenado, estos embeddings del decoder se aprenden desde cero durante el entrenamiento de text2midi. **Entrada:** secuencia de 2048 enteros. **Salida:** matriz de 2048 × 768.

**4. Positional Encoding**
A diferencia del encoder que usa Relative Position Bias, el decoder usa codificación posicional vanilla —vectores de posición que se suman directamente a los embeddings para indicar el orden de cada token en la secuencia. **Entrada / Salida:** 2048 × 768.

#### Bloque × 18 capas — dimensión 2048 × 768 constante

**5. Masked Multi-Head Self-Attention (Flash Attention, 8 cabezas, d_k = 96)**
Cada token MIDI atiende a todos los tokens anteriores en la secuencia, pero nunca a los tokens futuros. Esto se logra mediante una máscara causal que bloquea las posiciones futuras asignándoles un valor de menos infinito antes del softmax. Se usan 8 cabezas de atención, cada una trabajando en un subespacio de 96 dimensiones (768 / 8 = 96). En lugar del mecanismo de atención estándar, se implementa Flash Attention, un algoritmo optimizado que reduce el uso de memoria y acelera el cómputo aprovechando la jerarquía de memoria de la GPU. **Entrada / Salida:** 2048 × 768.

**6. Add & Norm — Post-LN**
La salida de la self-attention se suma con la entrada original del sublayer (conexión residual) y luego se normaliza. A diferencia del encoder que usa Pre-LN, el decoder usa Post-LN —primero se suma el residual y luego se normaliza. **Entrada:** 2048 × 768 (residual) + 2048 × 768 (salida atención). **Salida:** 2048 × 768.

**7. Multi-Head Cross-Attention (8 cabezas, d_k = 96)**
Este es el mecanismo que conecta el encoder con el decoder. Los Queries provienen de la salida del sublayer anterior del decoder, mientras que las Keys y Values se calculan a partir de los Hidden States HT del encoder. Esto permite que cada token MIDI en generación consulte la representación completa del caption de texto para decidir qué token musical generar a continuación. **Q (decoder):** 2048 × 768. **K, V (encoder HT):** n × 768. **Salida:** 2048 × 768.

**8. Add & Norm — Post-LN**
La salida del cross-attention se suma con la entrada del sublayer de cross-attention (conexión residual) y se normaliza. **Entrada:** 2048 × 768 (residual) + 2048 × 768 (salida cross-attention). **Salida:** 2048 × 768.

**9. Feed-Forward (768 → 1024 → 768)**
Cada token pasa de forma independiente por una red neuronal de dos capas. La primera proyección expande la dimensión de 768 a 1024, se aplica la función de activación ReLU que introduce no-linealidad eliminando valores negativos, seguida de Dropout para regularización, y finalmente una segunda proyección que regresa la dimensión a 768. **Entrada:** 2048 × 768. **Dimensión interna:** 2048 × 1024. **Salida:** 2048 × 768.

**10. Add & Norm — Post-LN**
La salida de la red Feed-Forward se suma con su entrada original (conexión residual) y se normaliza. Con esto se completa un bloque completo del decoder. Este proceso se repite 18 veces en total. **Entrada:** 2048 × 768 (residual) + 2048 × 768 (salida FFN). **Salida:** 2048 × 768.

**11. Final Layer Norm**
Después de las 18 capas apiladas se aplica una normalización final sobre toda la representación acumulada del decoder, garantizando que los valores estén en un rango estable antes de la proyección lineal final. **Entrada / Salida:** 2048 × 768.

**12. Linear projection (768 → vocab_size)**
Una capa lineal sin función de activación proyecta cada vector de 768 dimensiones al tamaño completo del vocabulario REMI+. Esta capa tiene sus propios pesos aprendibles y transforma la representación interna del modelo en puntuaciones (logits) para cada token posible del vocabulario. La operación se aplica independientemente a cada una de las 2048 posiciones. **Entrada:** 2048 × 768. **Salida:** 2048 × vocab_size.

**13. Softmax**
Los logits se convierten en una distribución de probabilidad válida mediante la función Softmax, aplicada fila por fila. Cada fila suma exactamente 1.0 y representa la probabilidad de que cada token del vocabulario REMI+ sea el siguiente token correcto en esa posición. Durante el entrenamiento se usa la distribución completa para calcular la cross-entropy loss. Durante la inferencia se muestrea el token con mayor probabilidad, modulada por la temperatura configurada. **Entrada:** 2048 × vocab_size (logits). **Salida:** 2048 × vocab_size (probabilidades).

**14. Output Probabilities → Token MIDI predicho**
El token con mayor probabilidad en la última posición de la secuencia se selecciona como el siguiente token MIDI generado. Este token se agrega al final de la secuencia de entrada y el proceso completo se repite desde el paso 5, ahora con una secuencia de longitud aumentada en uno. Este ciclo autoregresivo continúa hasta que el modelo genera el token especial de fin de secuencia (EOS) o se alcanza el límite de 2048 tokens.

**15. Detokenización REMI+ → archivo .mid**
Una vez generada la secuencia completa de tokens MIDI, el decodificador REMI+ convierte cada token de vuelta a su evento musical correspondiente —reconstruyendo las notas, instrumentos, tiempos y duraciones— y los escribe en un archivo MIDI binario. Este archivo puede reproducirse directamente o convertirse a audio mediante FluidSynth. **Entrada:** secuencia de tokens REMI+. **Salida:** archivo output.mid.

<img width="1312" height="1019" alt="text2midi_architecture Diapositiva-Page-1 drawio" src="https://github.com/user-attachments/assets/6e60718f-9df6-436b-b2f8-e5d48873ba01" />

---

## 4. Metodología

La implementación de Text2midi para inferencia siguió un proceso estructurado en cuatro fases: configuración del entorno, gestión de dependencias, descarga y organización de los pesos preentrenados, e integración de la interfaz gráfica. El punto de partida fue un fork del repositorio original `AMAAI-Lab/Text2midi`, sobre el cual se construyó toda la implementación sin modificar la arquitectura del modelo descrita en Bhandari et al. (2024).

Dado que varias dependencias del proyecto requieren un entorno Linux nativo, se trabajó sobre **WSL2 (Windows Subsystem for Linux)** con Ubuntu, copiando el proyecto directamente al filesystem de Linux para evitar la penalización de rendimiento que introduce la capa de traducción de archivos entre Windows y Linux. El entorno de ejecución se configuró con **Python 3.10** dentro de un entorno virtual aislado, garantizando compatibilidad con todas las versiones de las librerías requeridas.

En cuanto a las dependencias, el `requirements.txt` original incluye paquetes diseñados exclusivamente para entrenamiento distribuido —como `triton`, `deepspeed`, `wandb` y `accelerate`— que no son necesarios para inferencia y que en algunos entornos generan errores de compilación. Por esta razón se instaló únicamente el conjunto mínimo requerido: `torch`, `transformers`, `miditok`, `sentencepiece`, `streamlit` y `symusic`. La versión de PyTorch se seleccionó según la disponibilidad de hardware: `cu124` para equipos con GPU NVIDIA y la variante CPU para equipos sin acelerador gráfico, lo que implica tiempos de generación de aproximadamente 3 a 5 minutos por pieza en lugar de los 55 segundos aproximados que tarda con GPU.

Los pesos preentrenados se descargaron automáticamente desde HuggingFace Hub (`amaai-lab/text2midi`) mediante la librería `huggingface_hub`. El repositorio publica dos artefactos: `pytorch_model.bin` (~1.2 GB), que contiene los pesos del decoder Transformer de 18 capas con 159M parámetros entrenables, y `vocab_remi.pkl`, que almacena el tokenizador REMI+ con un vocabulario de 524 tokens. Los pesos del encoder FLAN-T5 base se descargan por separado desde HuggingFace Transformers y se mantienen congelados durante toda la inferencia. Una vez cargados los pesos, el modelo se configura en modo evaluación mediante `model.eval()` dentro de un contexto `torch.no_grad()`, deshabilitando el cálculo de gradientes para reducir el consumo de memoria.

La lógica del proyecto se organizó en módulos independientes con responsabilidades claramente delimitadas:

- **`model_loader.py`** — encapsula la detección del dispositivo, la descarga de pesos y la inicialización del modelo.
- **`generator.py`** — implementa el pipeline completo de inferencia en cuatro pasos secuenciales: tokenizar el caption con el tokenizador T5, generar tokens REMI+ de forma autoregresiva, decodificar la secuencia a un objeto MIDI mediante MidiTok, y serializar el resultado a bytes para su descarga o reproducción.
- **`app.py`** — construye la interfaz Streamlit de tres paneles con caché del modelo vía `@st.cache_resource`, evitando recargas en cada petición del usuario.

### 4.1 Uso de pesos

El proceso comienza detectando el dispositivo de cómputo disponible, revisando en orden de prioridad: CUDA para GPUs NVIDIA, MPS para Apple Silicon, o CPU como alternativa por defecto. Posteriormente se descargan los artefactos necesarios desde el repositorio `amaai-lab/text2midi` en HuggingFace Hub. Ambos archivos se almacenan en caché local, por lo que la descarga solo ocurre la primera vez.

Con el vocabulario disponible, se deserializa el tokenizador REMI+ desde el archivo pickle, obteniendo un tokenizador MidiTok de 524 tokens capaz de representar eventos musicales como notas, tiempos, instrumentos y dinámicas. A continuación se instancia la arquitectura Transformer con los hiperparámetros definidos en el paper: 768 dimensiones de embedding, 8 cabezas de atención, 18 capas de decoder y una dimensión de 1024 en la red feedforward. Sobre esta arquitectura se cargan los pesos descargados mediante `load_state_dict()` y el modelo se coloca en modo evaluación con `eval()`. Finalmente se inicializa el tokenizador de texto `T5Tokenizer` de `google/flan-t5-base`.

### 4.2 Inferencia del modelo

Una vez el modelo está listo, el proceso de inferencia inicia recibiendo una descripción textual de la música deseada, por ejemplo: *"A happy jazz song in A major at 120 BPM with piano"*. Esta descripción es tokenizada por el T5Tokenizer, produciendo tensores de `input_ids` y `attention_mask` que representan el texto como secuencias numéricas.

Con estos tensores, el decoder comienza la generación autoregresiva de tokens REMI+ bajo `torch.no_grad()` para desactivar el cálculo de gradientes y optimizar el uso de memoria. En cada paso, el mecanismo de cross-attention permite que el token MIDI actual consulte los hidden states producidos por el encoder FLAN-T5 a partir del texto, mientras que el masked self-attention garantiza que cada nuevo token solo tenga visibilidad sobre los tokens musicales ya generados.

La distribución de probabilidad sobre el vocabulario se escala mediante un parámetro de temperatura: valores menores a 1.0 producen música más predecible y conservadora, mientras que valores mayores generan resultados más variados y creativos. Este proceso se repite hasta alcanzar el número máximo de tokens definido, donde aproximadamente 500 tokens corresponden a un minuto de música.

Con la secuencia completa de tokens generada, el tokenizador MidiTok la decodifica a un objeto `symusic.Score`, reconstruyendo las notas, velocidades, tiempos e instrumentos de la composición. El MIDI se serializa a bytes y se retornan junto con el conteo de tokens generados para su uso en la interfaz Streamlit.

<img width="1542" height="791" alt="Copy of text2midi_architecture Diapositiva-Page-3 drawio (1)" src="https://github.com/user-attachments/assets/56306c43-80b5-451a-9610-652dbf603ce1" />

---

## 5. Métricas y Resultados

### 5.1 Métricas Oficiales de Evaluación

La evaluación oficial del modelo se realizó sobre 100 ejemplos del test set de MidiCaps, comparando contra MuseCoco (xlarge) y el ground truth.

| Métrica | text2midi | MuseCoco | Significancia |
|---|---|---|---|
| Compression Ratio ↑ | 2.31 | 2.12 | p < 0.0001 |
| CLAP Score ↑ | 0.22 | 0.21 | p = 0.0102 |
| Tempo Bin (%) ↑ | 39.70 | 21.71 | p = 0.1102 |
| Tempo Bin Tolerance (%) ↑ | 65.80 | 54.63 | p = 0.2051 |
| Correct Key (%) ↑ | 33.60 | 13.70 | p < 0.0001 |
| Correct Key + Duplicates (%) ↑ | 35.60 | 14.59 | p < 0.0001 |

Text2midi supera a MuseCoco en todas las métricas, siendo Compression Ratio y Correct Key estadísticamente significativos (p < 0.0001). La tonalidad correcta (33.60% vs 13.70%) es especialmente destacable dado que text2midi entrenó con 1/5 de los datos de MuseCoco. La inferencia tarda ~55 segundos versus ~120 segundos de MuseCoco.

### 5.2 Listening Study

11 participantes evaluaron 15 MIDIs en escala Likert 1-7 (muy malo a muy bueno):

| Criterio | MidiCaps GT | text2midi | MuseCoco |
|---|---|---|---|
| Calidad musical | 5.79 | 4.62 | 4.40 |
| Match general | 5.42 | 4.67 | 4.07 |
| Match de género | 5.54 | 4.98 | 4.40 |
| Match de mood | 5.70 | 5.00 | 4.32 |
| Match de tonalidad | 4.61 | 3.64 | 3.36 |
| Match de acordes | 3.20 | 2.50 | 2.00 |
| Match de tempo | 5.89 | 5.42 | 4.94 |

Text2midi supera a MuseCoco en todos los criterios. El criterio de menor puntaje para todos fue el match de acordes —reconocido por los autores como la característica más difícil de controlar. La diferencia con el ground truth no es grande, lo que indica que el modelo aprende bien la distribución de MidiCaps.

### 5.3 Resultados Obtenidos

Se realizaron múltiples pruebas de inferencia variando el nivel de detalle del caption, la temperatura y la longitud de generación. Las observaciones fueron obtenidas directamente de las pruebas realizadas sobre la interfaz Streamlit desarrollada:

- El modelo demostró capacidad para generar música coherente con la descripción textual en la mayor parte de la secuencia, con correspondencia más clara a partir de los primeros 5 a 10 segundos, una vez que el decoder acumula suficiente contexto MIDI.
- La generación con `max_len=1000` tomó aproximadamente 38.5 segundos sobre una GPU NVIDIA RTX 5070 Ti, resultado consistente con los ~55 segundos reportados por los autores para `max_len=2000`, confirmando que el tiempo escala aproximadamente de forma lineal con la longitud.
- Con temperatura 1.0 los resultados fueron coherentes y alineados con el caption. Por encima de 1.2 se observó degradación notable, confirmando que el rango de temperatura útil del modelo es estrecho.
- El modelo respetó correctamente instrumentos principales como piano, guitarra y saxofón, pero omitió con frecuencia instrumentos secundarios mencionados en el caption.
- Un prompt que solicitaba *"flute music 15 seconds"* generó un MIDI de ~20 segundos, evidenciando que el modelo no interpreta instrucciones de duración temporal.
- Los primeros segundos del MIDI tendieron a ser menos coherentes con el caption, comportamiento consistente con la limitación arquitectural de la masked self-attention operando con contexto mínimo al inicio de la generación.
- La interfaz Streamlit funcionó correctamente en todas las pruebas. El reproductor waterfall, la descarga en MIDI y la conversión a MP3 mediante FluidSynth operaron sin inconvenientes.

---

## 6. Conclusiones

### 6.1 Aprendizajes

- La arquitectura encoder-decoder con encoder congelado es una estrategia efectiva para aprovechar modelos de lenguaje preentrenados en tareas multimodales sin los costos computacionales de un entrenamiento completo.
- Flash Attention es crítico para el procesamiento de secuencias largas. La atención vanilla escribe y lee la matriz de atención completa en la memoria principal de la GPU, lo que se vuelve costoso a medida que crece la secuencia. Flash Attention hace el mismo cálculo pero en bloques pequeños en una memoria más rápida de la GPU, sin nunca escribir la matriz completa.
- La representación simbólica REMI+ permite aplicar directamente arquitecturas de LLMs al dominio musical, convirtiendo la generación de música en un problema análogo a la generación de texto.
- El sentence omission como técnica de regularización aplicada durante la etapa de fine-tuning es una innovación simple pero efectiva. Durante cada iteración de entrenamiento, entre el 20% y el 50% de las frases del caption son eliminadas aleatoriamente con una probabilidad del 50%, exponiendo al modelo a descripciones incompletas e impredecibles.

### 6.2 Limitaciones Identificadas

- **Instrumentación incompleta:** el modelo ignora instrumentos específicos del caption, especialmente los que aparecen tarde en la generación cuando el contexto de 2048 tokens ya está saturado con otros eventos musicales.
- **Tempo y acordes difíciles de controlar con precisión:** el chord matching obtuvo el puntaje más bajo del listening study (2.50/7). REMI+ no incluye tokens de acorde explícitos; los acordes emergen implícitamente de la co-ocurrencia de tokens de pitch en la misma posición, sin una instrucción directa que el modelo pueda seguir.
- **Sin control de duración:** el modelo no interpreta instrucciones de longitud temporal en el text caption. La duración del output depende exclusivamente de la variable `max_len`.
- **Sensibilidad alta a la temperatura:** valores superiores a 1.0 degradan rápidamente la coherencia musical, lo que limita la creatividad controlable del modelo.
- **Incoherencia al inicio de la generación:** al inicio de los archivos MIDI generados se evidencia la dificultad del modelo para predecir los tokens adecuados; posteriormente el modelo se estabiliza generando una salida aceptable.

---

## 7. Posibles Mejoras

- Ampliar la ventana de contexto a 4096 tokens (aprox. 10 min) para soportar piezas más largas con estructura musical de largo plazo y mayor desarrollo temático.
- Agregar un mecanismo de control de duración explícito que permita al modelo identificar y codificar la duración deseada, ya sea expresada en segundos o en número de compases.
- Explorar fine-tuning del encoder FLAN-T5 en el dominio musical para mejorar la comprensión de terminología técnica específica y progresiones armónicas.
- Implementar generación condicionada por segmentos para evitar la incoherencia al inicio, inicializando la secuencia con tokens de estructura musical que orienten al modelo desde el primer token generado.
- Agregar una estrategia que le permita al modelo interpretar referencias culturales o estilísticas en el caption, de tal forma que cuando un usuario solicite una pieza musical con similitudes a una obra o estilo reconocido, el modelo pueda tomar esa referencia como base para la generación.

---

## 8. Referencias

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., y Re, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS*. https://arxiv.org/abs/2205.14135
- Bhandari, K., Roy, A., Wang, K., Puri, G., Colton, S., y Herremans, D. (2024). Text2midi: Generating Symbolic Music from Captions. arXiv:2412.16526. https://arxiv.org/abs/2412.16526
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., y Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS*, vol. 30. https://arxiv.org/abs/1706.03762
- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., y Liu, P. J. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *Journal of Machine Learning Research*, 21(140), 1–67. https://arxiv.org/abs/1910.10683
- Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., et al. (2024). Scaling Instruction-Finetuned Language Models. *Journal of Machine Learning Research*, 25(70), 1–53. https://arxiv.org/abs/2210.11416
- Huang, Y. S., y Yang, Y. H. (2020). Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions. *ACM Multimedia*. https://arxiv.org/abs/2002.00212
- von Rutte, D., Biggio, L., Kilcher, Y., y Hofmann, T. (2023). FIGARO: Controllable Music Generation using Learned and Expert Features. *ICLR*. https://arxiv.org/abs/2201.10936
- Hendrycks, D., y Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). arXiv:1606.08415. https://arxiv.org/abs/1606.08415
- Google. (2023). FLAN-T5 Base Model Card and Configuration. HuggingFace Hub. https://huggingface.co/google/flan-t5-base/blob/main/config.json
- Fradet, N., Briot, J. P., Chhel, F., El Fallah Seghrouchni, A., y Gutowski, N. (2021). MidiTok: A Python Package for MIDI File Tokenization. *ISMIR 2021 LBD*. https://miditok.readthedocs.io/en/v3.0.1/tokenizations.html
