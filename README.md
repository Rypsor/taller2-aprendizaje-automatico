# Taller 2 - Aprendizaje Automático

Este repositorio contiene la solución al Taller 2, dividida en dos notebooks principales que abordan la implementación de funciones de pérdida, diferenciación automática y redes neuronales desde cero.

## Contenido

### 1. [punto-1.ipynb](punto-1.ipynb) - Entropía Cruzada (Cross Entropy)

Este notebook se enfoca en la implementación y análisis de la función de costo **Cross Entropy Loss**.

*   **Implementaciones**:
    *   **Naive**: Implementación directa usando la fórmula estándar. Susceptible a problemas de estabilidad numérica (overflow/underflow) con valores grandes.
    *   **Optimized**: Implementación numéricamente estable utilizando el truco **Log-Sum-Exp**.
*   **Experimentos**:
    *   Comparación de tiempos de ejecución entre ambas versiones.
    *   Prueba de estabilidad numérica con valores extremos (logits grandes), demostrando que la versión optimizada maneja correctamente estos casos donde la versión naive falla (NaN).

### 2. [punto-2.ipynb](punto-2.ipynb) - Red Neuronal desde Cero y Experimentos

Este notebook es más extenso y construye una librería de diferenciación automática básica para entrenar una red neuronal.

*   **Motor de Autograd**:
    *   Implementación de la clase `Tensor` que soporta operaciones matemáticas y el cálculo automático de gradientes (`backward`).
    *   Implementación de operaciones como `Add`, `Mul`, `MatMul`, `Log`, `ReLU`, `Softmax` y `Sum`.
*   **Red Neuronal**:
    *   Construcción de una clase `NeuralNetwork` modular utilizando el motor de autograd.
    *   Soporte para capas lineales, activaciones ReLU y salida Softmax.
*   **Experimentos de Inicialización de Pesos**:
    *   Evaluación de diferentes estrategias de inicialización: `zeros`, `ones`, `uniform` y `gaussian` (He Initialization).
    *   **Conclusión**: La inicialización Gaussiana (He) demuestra ser la más efectiva para la convergencia del modelo.
*   **Experimentos de Tamaño de Batch**:
    *   Entrenamiento del modelo variando el tamaño del batch: 16, 32, 64, 128, 256, Full Batch y SGD (Batch size 1).
    *   Análisis del impacto del batch size en el tiempo de entrenamiento y la convergencia de la pérdida.

## Requisitos

*   Python 3.x
*   Numpy
*   Pandas
*   Matplotlib
*   Scikit-learn (para preprocesamiento de datos)
