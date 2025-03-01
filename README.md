

¡Bienvenido al proyecto **DP-Neuronal-Example**! Este es un ejemplo práctico de una red neuronal implementada en PyTorch para predecir valores numéricos basados en una operación matemática. Este proyecto es ideal para principiantes que quieren aprender cómo funcionan las redes neuronales y cómo implementarlas en PyTorch.

## ¿Qué hace este proyecto?

Este proyecto entrena una red neuronal para predecir un valor numérico basado en la operación matemática `y = 5 * x / 2 + 15`. A partir de un conjunto de datos de entrenamiento, el modelo ajusta sus parámetros para hacer predicciones precisas.

## Características principales

- **Arquitectura de la red neuronal**: Una capa oculta con 20,000 neuronas y una capa de salida con 1 neurona.
- **Entrenamiento personalizado**: Puedes elegir cuántas épocas (iteraciones de entrenamiento) deseas que el modelo entrene.
- **Guardado y carga del modelo**: Puedes guardar el modelo entrenado y cargarlo más tarde para continuar el entrenamiento o hacer predicciones.
- **Menú interactivo**: Un menú fácil de usar te permite entrenar el modelo, hacer predicciones o salir del programa.
- **Extensible**: Puedes modificar la arquitectura de la red, la función de activación y otros parámetros para experimentar.

## Requisitos

Para ejecutar este proyecto, necesitas tener instalado lo siguiente:

- **Python 3.x**: El lenguaje de programación en el que está escrito el proyecto.
- **PyTorch**: Una biblioteca de aprendizaje profundo que utilizamos para construir y entrenar la red neuronal.
- **Numpy**: Una biblioteca para trabajar con matrices y operaciones matemáticas.

Instala las dependencias ejecutando el siguiente comando en tu terminal:

```bash
pip install torch numpy
```
## ¿Cómo usar este proyecto?

Sigue estos pasos para clonar, instalar y ejecutar el proyecto:

1. Clona el repositorio en tu computadora:

```bash
git clone https://github.com/cardona-tech/DP-Neuronal-Example.git
cd DP-Neuronal-Example
```

2. Ejecuta el script principal:

```bash
python models/NumericalTraining.py
```

3. Selecciona una opción del menú:
   - **1. Entrenar modelo**: Entrena el modelo con un número específico de épocas.
   - **2. Usar modelo para predecir**: Ingresa un número y obtén una predicción.
   - **3. Salir**: Termina el programa.
<details>
<summary><b>Ejemplo de salida</b> (Click para expandir)</summary>

```text
--- Menu ---
1. Train model
2. Use model to predict
3. Exit
Select an option:  1
⏳ Training model...
Epoch 0, Loss: 17885.968750, LR: [0.001]
Epoch 500, Loss: 0.001740, LR: [0.001]
Epoch 1000, Loss: 0.001322, LR: [0.0005]
Epoch 1500, Loss: 0.001097, LR: [0.0005]
Epoch 2000, Loss: 0.000862, LR: [0.00025]
Epoch 2500, Loss: 0.000738, LR: [0.00025]
Epoch 3000, Loss: 0.000611, LR: [0.000125]
Epoch 3500, Loss: 0.000543, LR: [0.000125]
Epoch 4000, Loss: 0.000467, LR: [6.25e-05]
Epoch 4500, Loss: 0.000425, LR: [6.25e-05]
Epoch 5000, Loss: 0.000377, LR: [3.125e-05]
Epoch 5500, Loss: 0.000350, LR: [3.125e-05]
Epoch 6000, Loss: 0.000318, LR: [1.5625e-05]
Epoch 6500, Loss: 0.000300, LR: [1.5625e-05]
Epoch 7000, Loss: 0.000278, LR: [7.8125e-06]
Epoch 7500, Loss: 0.000265, LR: [7.8125e-06]
Epoch 8000, Loss: 0.000250, LR: [3.90625e-06]
Epoch 8500, Loss: 0.000240, LR: [3.90625e-06]
Epoch 9000, Loss: 0.000229, LR: [1.953125e-06]
Epoch 9500, Loss: 0.000221, LR: [1.953125e-06]
✅ Model saved successfully.

--- Menu ---
1. Train model
2. Use model to predict
3. Exit
Select an option:  2
Enter a number:  10
Model prediction: 39.96
```
</details>

## Estructura del proyecto

El proyecto está organizado de la siguiente manera:

```
DP-Neuronal-Example/
│
├── models/
│   └── NumericalTraining.py       # Código principal de la red neuronal
├── README.md                      # Este archivo (guía del proyecto)
└── LICENSE                        # Licencia del proyecto (opcional)
```
## ¿Cómo funciona la red neuronal?

La red neuronal en este proyecto tiene la siguiente estructura:

1. **Capa de entrada**: Recibe un valor numérico (por ejemplo, `10`).
2. **Capa oculta**: Una capa con 20,000 neuronas que aplica una función de activación (ReLU) para procesar la entrada.
3. **Capa de salida**: Produce un valor numérico (por ejemplo, `40`).

Durante el entrenamiento, la red neuronal ajusta sus parámetros para minimizar la diferencia entre las predicciones y los valores reales. Esto se hace utilizando una función de pérdida (en este caso, el error cuadrático medio) y un optimizador (Adam).

## Experimentación

El código está diseñado para ser fácilmente modificable. Aquí tienes algunos ejemplos de cómo puedes experimentar:

### 1. Cambiar el número de neuronas
Puedes modificar el número de neuronas en la capa oculta. Por ejemplo, para usar 1,000 neuronas en lugar de 20,000, cambia esta línea en la clase `NeuralNetwork`:

```python 
self.hidden1 = nn.Linear(1, 20000) # Modifica la cantidad de neuronas 

# Ejemplo para usar 1,000

self.hidden1 = nn.Linear(1, 1,000)
```

### 2. Agregar más capas ocultas
Puedes agregar más capas ocultas a la red. Para hacerlo, edita la clase `NeuralNetwork` y agrega una segunda capa oculta:

```pythoon
self.hidden2 = nn.Linear(20000, 20000)
```

Luego, en la función `forward`, aplica la segunda capa oculta:

```python
x = torch.relu(self.hidden2(x))
```

### 3. Cambiar la función de activación
Puedes probar otras funciones de activación, como `tanh` o `sigmoid`, en lugar de `ReLU`. Por ejemplo:

```python
x = torch.tanh(self.hidden1(x))
```
### 4. Modificar la tasa de aprendizaje
La tasa de aprendizaje controla cuánto ajusta el modelo sus parámetros durante el entrenamiento. Puedes cambiarla modificando el optimizador:

```python
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

### 5. Cambiar los datos de entrenamiento
Puedes agregar más datos de entrenamiento para mejorar la precisión del modelo. Por ejemplo:

```python
x_train = torch.tensor([
    [10.0], [20.0], [15.0], [25.0], [30.0],
    [40.0], [50.0], [60.0], [70.0], [80.0],
    [90.0], [100.0]  # Agrega más datos
], dtype=torch.float32)

y_train = torch.tensor([
    [40.0], [65.0], [52.5], [77.5], [90.0],
    [115.0], [140.0], [165.0], [190.0], [215.0],
    [240.0], [265.0]  # Agrega más resultados
], dtype=torch.float32)
```

### 6. Cambiar la función de pérdida
Puedes usar una función de pérdida diferente, como `L1Loss`, que mide el error absoluto en lugar del error cuadrático:

```python
criterion = nn.L1Loss()
```

## Ejemplos de predicción

Aquí tienes algunos ejemplos de cómo funciona el modelo:

1. **Entrada**: `10` → **Salida esperada**: `40` (ya que `5 * 10 / 2 + 15 = 40`).
2. **Entrada**: `20` → **Salida esperada**: `65`.
3. **Entrada**: `30` → **Salida esperada**: `90`.

## Preguntas frecuentes (FAQ)

### ¿Cómo puedo mejorar la precisión del modelo?
- Aumenta el número de épocas de entrenamiento.
- Agrega más datos de entrenamiento.
- Experimenta con diferentes arquitecturas de red (más capas o neuronas).

### ¿Qué es el "early stopping"?
El "early stopping" es una técnica que detiene el entrenamiento si la pérdida no mejora después de un número específico de épocas. En este proyecto, el entrenamiento se detiene si no hay mejora en 1,000 épocas.

## Contribuciones

¡Las contribuciones son bienvenidas! Si encuentras algún error o tienes sugerencias, abre un issue o envía un pull request.

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.
