import numpy as np


def initialize_weights(n_features, n_classes):
    """
    Inicializa la matriz de pesos W
    n_features: int - número de características (incluye bias)
    n_classes: int - número de clases (3 en tu caso: Bajo=0, Medio=1, Alto=2)
    Returns: W [n_features, n_classes] - matriz de pesos inicializada
    
    CAMBIO vs binario:
    - Antes: w = np.ones(n_features) -> vector
    - Ahora: W = inicialización -> matriz [n_features, n_classes]
    """
    pass

def h(x, w):
    return np.dot(x, w)
def s(X, W):
    """
    Calcula la función softmax (equivalente a sigmoide en multiclase)
    X: [n_samples, n_features] - matriz de características
    W: [n_features, n_classes] - matriz de pesos
    Returns: probabilidades [n_samples, n_classes] que suman 1 por fila
    
    CAMBIO vs binario:
    - Antes: 1/(1 + exp(-h(x,w))) -> una probabilidad
    - Ahora: softmax(h(X,W)) -> K probabilidades
    """
    pass

def Loss_function(X, y, W):
    """
    Calcula la función de pérdida cross-entropy categórica
    X: [n_samples, n_features] - matriz de características
    y: [n_samples] - etiquetas verdaderas (0, 1, 2)
    W: [n_features, n_classes] - matriz de pesos
    Returns: scalar - pérdida promedio
    
    CAMBIO vs binario:
    - Antes: log-likelihood binario
    - Ahora: cross-entropy categórica con indicador 1{y=k}
    """
    pass

def Derivatives(X, y, W):
    """
    Calcula los gradientes de la función de pérdida con respecto a W
    X: [n_samples, n_features] - matriz de características
    y: [n_samples] - etiquetas verdaderas (0, 1, 2)
    W: [n_features, n_classes] - matriz de pesos
    Returns: gradients [n_features, n_classes] - gradientes de W
    
    CAMBIO vs binario:
    - Antes: gradiente vectorial
    - Ahora: gradiente matricial usando softmax
    """
    pass

def change_parameters(W, derivatives, alpha):
    """
    Actualiza los pesos usando un paso de gradiente descendente
    W: [n_features, n_classes] - matriz de pesos actual
    derivatives: [n_features, n_classes] - gradientes calculados
    alpha: float - tasa de aprendizaje
    Returns: W_new [n_features, n_classes] - pesos actualizados
    
    CAMBIO vs binario:
    - Antes: w - alpha * derivatives (vectores)
    - Ahora: W - alpha * derivatives (matrices)
    - MISMA LÓGICA, diferentes dimensiones
    """
    pass

def training(X_train, y_train, epochs, alpha, X_test, y_test):
    """
    Función principal de entrenamiento del modelo softmax
    X_train: [n_samples, n_features] - datos de entrenamiento
    y_train: [n_samples] - etiquetas de entrenamiento (0, 1, 2)
    epochs: int - número de épocas
    alpha: float - tasa de aprendizaje
    X_test, y_test: datos de validación
    Returns: 
        - W [n_features, n_classes] - pesos finales
        - LossTrain, LossTest - historial de pérdidas
    
    CAMBIO vs binario:
    - MISMA ESTRUCTURA del loop
    - Inicialización: W = initialize_weights() en lugar de w = np.ones()
    - Llama las mismas funciones con parámetros diferentes
    """
    pass

def Testing(X_test, y_test, W):
    """
    Evalúa el modelo en datos de prueba
    X_test: [n_samples, n_features] - datos de prueba
    y_test: [n_samples] - etiquetas verdaderas
    W: [n_features, n_classes] - pesos entrenados
    
    CAMBIO vs binario:
    - Antes: np.round(s(x,w)) para 0/1
    - Ahora: np.argmax(s(X,W), axis=1) para 0/1/2
    - MISMA LÓGICA de evaluación
    """
    pass