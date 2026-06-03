
# Preguntas sobre el ejemplo de clasificación de imágenes con PyTorch y MLP

## 1. Dataset y Preprocesamiento
- ¿Por qué es necesario redimensionar las imágenes a un tamaño fijo para una MLP?
Porque una MLP recibe vectores de longitud fija. Si las imágenes tienen distintos tamaños, el número de píxeles cambia y la cantidad de entradas de la primera capa también.
- ¿Qué ventajas ofrece Albumentations frente a otras librerías de transformación como `torchvision.transforms`?
Mayor velocidad, más transformaciones disponibles, mejor integración con OpenCV y manejo sencillo de máscaras y bounding boxes.
- ¿Qué hace `A.Normalize()`? ¿Por qué es importante antes de entrenar una red?
Normaliza los valores de los píxeles, permite que todas las entradas tengan escalas similares, facilitando el entrenamiento y mejorando la estabilidad numérica.
- ¿Por qué convertimos las imágenes a `ToTensorV2()` al final de la pipeline?
Convierte la imagen desde un array de NumPy a un tensor de PyTorch y reorganiza las dimensiones al formato esperado por PyTorch

## 2. Arquitectura del Modelo
- ¿Por qué usamos una red MLP en lugar de una CNN aquí? ¿Qué limitaciones tiene?
Porque es más simple, computacionalmente eficiente y flexible que una CNN. Su limitación es que no aprovecha estructura espacial, contiene muchos parámetros y no generaliza tan bien en imágenes.
- ¿Qué hace la capa `Flatten()` al principio de la red?
Transforma la imagen en un vector 1D.
- ¿Qué función de activación se usó? ¿Por qué no usamos `Sigmoid` o `Tanh`?
Se utiliza ReLu que es más rápida y tiene menos problemas de gradientes desvanecidos, a diferencia de sigmoid y Tanh. 
- ¿Qué parámetro del modelo deberíamos cambiar si aumentamos el tamaño de entrada de la imagen?
Al aumentar el tamaño de la imagen aumenta la cantidad de píxeles de entrada. Como una MLP utiliza una capa Flatten, debe modificarse el parámetro input_size de la primera capa lineal para que coincida con la nueva dimensión del vector de entrada.

## 3. Entrenamiento y Optimización
- ¿Qué hace `optimizer.zero_grad()`?
PyTorch acumula gradientes por defecto, por lo que esto los reinicia para evitar sumar gradientes de batches anteriores.
- ¿Por qué usamos `CrossEntropyLoss()` en este caso?
Porque es clasificación multiclase. Combina Softmax y Negative Log Likelihood.
- ¿Cómo afecta la elección del tamaño de batch (`batch_size`) al entrenamiento?
El tamaño del batch influye en la estabilidad del entrenamiento. Los batches pequeños generan gradientes más ruidosos, lo que puede mejorar la generalización. Los batches grandes producen actualizaciones más estables y eficientes computacionalmente, pero consumen más memoria y pueden favorecer el overfitting. 
- ¿Qué pasaría si no usamos `model.eval()` durante la validación?
Dropout seguiría apagando neuronas aleatoriamente durante la validación.

## 4. Validación y Evaluación
- ¿Qué significa una accuracy del 70% en validación pero 90% en entrenamiento?
Indica overfitting, ya que el modelo memorizó entrenamiento pero generaliza mal.
- ¿Qué otras métricas podrían ser más relevantes que accuracy en un problema real?
Otras métricas relevantes pueden ser Precision, Recall y F1-score.
- ¿Qué información útil nos da una matriz de confusión que no nos da la accuracy?
Muestra exactamente qué clases se confunden entre sí, mientras que la accuracy no indica dónde ocurren los errores.
- En el reporte de clasificación, ¿qué representan `precision`, `recall` y `f1-score`?
Precision representa cuantas de las predicciones positivas son correctas. Recall representa cuanto de los positivos reales detecto. F1-score representa el balance entre precision y recall.

## 5. TensorBoard y Logging
- ¿Qué ventajas tiene usar TensorBoard durante el entrenamiento?
Permite monitorear el entrenamienot, detectar overfitting, comparar experimentos y visualizar métricas.
- ¿Qué diferencias hay entre loguear `add_scalar`, `add_image` y `add_text`?
Guardan números, imágenes y texto, respectivametne.
- ¿Por qué es útil guardar visualmente las imágenes de validación en TensorBoard?
Nos permite detectar errores, calidad de los datos, progreso del modelo, etc.
- ¿Cómo se puede comparar el desempeño de distintos experimentos en TensorBoard?
Ejecutando distintos runs y visualizándolos juntos en TensorBoard.

## 6. Generalización y Transferencia
- ¿Qué cambios habría que hacer si quisiéramos aplicar este mismo modelo a un dataset con 100 clases?
Se debería cambiar el número de neuronas de salida en la última capa lineal de la red para que coincida con las 100 clases.
- ¿Por qué una CNN suele ser más adecuada que una MLP para clasificación de imágenes?
Porque explota correlaciones espaciales mediante convoluciones, tiene muchos menos parámetros.
- ¿Qué problema podríamos tener si entrenamos este modelo con muy pocas imágenes por clase?
Pueden aparecer overfitting, alta varianza y mala generalización.
- ¿Cómo podríamos adaptar este pipeline para imágenes en escala de grises?
Habría que cargar lás imagenes con un único canal en vez de 3, o sea cambia la dimensión de entrada de la red.

## 7. Regularización

### Preguntas teóricas:
- ¿Qué es la regularización en el contexto del entrenamiento de redes neuronales?
Es un conjunto de técnicas para reducir overfitting.
- ¿Cuál es la diferencia entre `Dropout` y regularización `L2` (weight decay)?
El Dropout apaga neuronas aleatoriamente mientras que L2 penaliza pesos grandes:
- ¿Qué es `BatchNorm` y cómo ayuda a estabilizar el entrenamiento?
Es una capa que normaliza las activaciones de una red neuronal utilizando la media y la desviación estándar calculadas sobre cada batch de entrenamiento, manteniendo las activaciones aproximadamente centradas y con una escala controlada.
- ¿Cómo se relaciona `BatchNorm` con la velocidad de convergencia?
Reduce cambios bruscos en distribuciones internas, permite usar learning rates más altos.
- ¿Puede `BatchNorm` actuar como regularizador? ¿Por qué?
Si porque cada batch tiene estadístiucas lgieramente distintas, introduciendo ruido. 
- ¿Qué efectos visuales podrías observar en TensorBoard si hay overfitting?
Veríamos:
   - Loss train baja.
   - Loss validation sube.
   - Accuracy train sigue creciendo.
   - Accuracy validation se estanca.
- ¿Cómo ayuda la regularización a mejorar la generalización del modelo?
Evita que el modelo memorice ejemplos específicos.

### Actividades de modificación:
1. Agregar Dropout en la arquitectura MLP:
   - Insertar capas `nn.Dropout(p=0.5)` entre las capas lineales y activaciones.
   - Comparar los resultados con y sin `Dropout`.

2. Agregar Batch Normalization:
   - Insertar `nn.BatchNorm1d(...)` después de cada capa `Linear` y antes de la activación:
     ```python
     self.net = nn.Sequential(
         nn.Flatten(),
         nn.Linear(in_features, 512),
         nn.BatchNorm1d(512),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(512, 256),
         nn.BatchNorm1d(256),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(256, num_classes)
     )
     ```

3. Aplicar Weight Decay (L2):
   - Modificar el optimizador:
     ```python
     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
     ```

4. Reducir overfitting con data augmentation:
   - Agregar transformaciones en Albumentations como `HorizontalFlip`, `BrightnessContrast`, `ShiftScaleRotate`.

5. Early Stopping (opcional):
   - Implementar un criterio para detener el entrenamiento si la validación no mejora después de N épocas.

### Preguntas prácticas:
Sin Dropout se obtuvo la mejor accuracy de validación puntual, con 57.22% en la época 8. Con Dropout sin BatchNorm, la validación final fue similar, 56.67%, pero el entrenamiento llegó a mayor accuracy, 73.08%, lo que sugiere que el modelo siguió aprendiendo más sobre train sin mejorar claramente en validación.
- ¿Qué efecto tuvo `BatchNorm` en la estabilidad y velocidad del entrenamiento?
BatchNorm hizo que la loss fuera más estable y terminara más baja en validación. La configuración con BatchNorm tuvo la menor val_loss final: 1.1778, frente a 1.4594 sin Dropout y 1.4858 con Dropout sin BatchNorm. No se observó una aceleración significativa de la convergencia al utilizar BatchNorm, ya que la mejor accuracy de validación se alcanzó en una cantidad de épocas similar al modelo base.
- ¿Cambió la performance de validación al combinar `BatchNorm` con `Dropout`? 
La combinación no mejoró la accuracy de validación. Su mejor valor fue 54.44%, menor que las otras configuraciones. Sin embargo, sí mostró una validación más estable en términos de loss.
- ¿Qué combinación de regularizadores dio mejores resultados en tus pruebas?
Si se prioriza accuracy, la mejor fue sin Dropout, con 57.22% de validación. Si se prioriza menor loss y estabilidad, la mejor fue Dropout + BatchNorm.
- ¿Notaste cambios en la loss de entrenamiento al usar `BatchNorm`?
Sí, la loss de entrenamiento bajó de forma más progresiva y la loss de validación terminó más baja. Esto indica mayor estabilidad, aunque no necesariamente mejor accuracy.

## 8. Inicialización de Parámetros

### Preguntas teóricas:
- ¿Por qué es importante la inicialización de los pesos en una red neuronal?
Porque afecta a la velocidad de convergencia, estabilidad y calidad final. 
- ¿Qué podría ocurrir si todos los pesos se inicializan con el mismo valor?
Todas las neuronas aprenderían lo mismo.
- ¿Cuál es la diferencia entre las inicializaciones de Xavier (Glorot) y He?
Zavier está diseñada para Sigmoid y Tanh mientras que He está diseñada para ReLu.
- ¿Por qué en una red con ReLU suele usarse la inicialización de He?
Porque ReLu elimina activaciones y He copensa las neuronas que qudan en cero. 
- ¿Qué capas de una red requieren inicialización explícita y cuáles no?
Lo requieren las capas que contienen pesos que la red debe aprender (Linear, Conv2D, etc). Las que no lo requieren es porque no tienen pesos entrebales o Pytorch ya los maneja (Relu, sigmoid, etc).

### Actividades de modificación:
1. Agregar inicialización manual en el modelo:
   - En la clase `MLP`, agregar un método `init_weights` que inicialice cada capa:
     ```python
     def init_weights(self):
         for m in self.modules():
             if isinstance(m, nn.Linear):
                 nn.init.kaiming_normal_(m.weight)
                 nn.init.zeros_(m.bias)
     ```

2. Probar distintas estrategias de inicialización:
   - Xavier (`nn.init.xavier_uniform_`)
   - He (`nn.init.kaiming_normal_`)
   - Aleatoria uniforme (`nn.init.uniform_`)
   - Comparar la estabilidad y velocidad del entrenamiento.

3. Visualizar pesos en TensorBoard:
   - Agregar esta línea en la primera época para observar los histogramas:
     ```python
     for name, param in model.named_parameters():
         writer.add_histogram(name, param, epoch)
     ```

### Preguntas prácticas:
- ¿Qué diferencias notaste en la convergencia del modelo según la inicialización?
La inicialización He fue la que alcanzó la mejor accuracy de validación final, con 51.67%, y es consistente con el uso de activaciones ReLU. Xavier también convergió correctamente, llegando a 50.56%. La inicialización uniforme tuvo una convergencia aceptable, pero más irregular ya que alcanzó 50.56% en la época 8, pero luego bajó a 49.44%.
- ¿Alguna inicialización provocó inestabilidad (pérdida muy alta o NaNs)?
No se observaron NaNs ni una pérdida que hiciera divergir completamente el entrenamiento. Al inicio todas tuvieron losses altas, especialmente He, con una train loss inicial de 14.32, y Xavier, con 11.86. Aun así, ambas lograron reducir la loss con las épocas.
- ¿Qué impacto tiene la inicialización sobre las métricas de validación?
La inicialización modificó levemente el rendimiento de validación. La mejor fue He, con 51.67%, seguida por Xavier, con 50.56%, y luego uniforme, con 49.44% final.
- ¿Por qué `bias` se suele inicializar en cero?
Porque el bias no genera el problema de simetría que sí pueden generar los pesos. Si todos los pesos fueran iguales, las neuronas aprenderían lo mismo. Inicializar los sesgos en cero no impide que las neuronas aprendan distinto, porque la diversidad ya viene dada por la inicialización aleatoria de los pesos.