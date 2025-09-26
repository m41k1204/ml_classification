import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train_df = pd.read_csv('datos_entrenamiento_riesgo.csv')
test_df = pd.read_csv('datos_prueba_riesgo.csv')

def h(x,w):
  return np.dot(x, w)

def s(x,w):
  return 1/ (1 + np.exp(-h(x, w)))

def Loss_function(x, y, w):
  epsilon = 1e-15
  n = x.shape[0]
  sum = np.sum(y * np.log(s(x, w) + epsilon)  +
                 (1 - y) * np.log(epsilon + 1 - s(x, w)))
  return (-1/n) * sum

def Derivatives(x,y,w):
  sum = np.dot(-x.T,y - (s(x,w)))
  n = x.shape[0]
  return (1/n) * sum

# usando gradiente descendiente
def change_parameters(w, derivatives, alpha):
  return w - alpha * derivatives

def training(x_train,y_train, epochs, alpha, x_test, y_test):
  print(len(x_train))
  w_train=np.ones(x_train.shape[1])
  LossTrain = []
  LossTest = []
  for i in range(epochs):
    L_Train =  Loss_function(x_train,y_train,w_train)
    L_Test = Loss_function(x_test, y_test, w_train)
    dw = Derivatives(x_train,y_train,w_train)
    w_train =  change_parameters(w_train, dw, alpha)
    if i % 100 == 0:
      print("L_Train:", L_Train)
    LossTrain.append(L_Train)
    LossTest.append(L_Test)
  return w_train,LossTrain, LossTest

def Testing(x_test, y_test,w):
   y_pred = s(x_test,w)
   y_pred = np.round(y_pred)
   correctos = np.sum(y_pred == y_test)
   print(f"Número de datos correctos:{correctos}")
   porc_aciertos= (correctos/len(y_test))*100
   print(f"Porcentaje de aciertos:{porc_aciertos}%")
   print(f"Porcentaje de error:{100-porc_aciertos}%")

def normalizar(x):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x, scaler

def addBias(x):
  return np.column_stack([np.ones(x.shape[0]), x])

def one_vs_all_training(X_train, y_train, X_test, y_test, epochs, alpha):
    models = {}  
    class_names = ['Bajo', 'Medio', 'Alto']
    
    for i, class_name in enumerate(class_names):
        print(f"\n--- Entrenando modelo {class_name} vs Resto ---")
        
        y_binary = (y_train == i).astype(int)
        y_test_binary = (y_test == i).astype(int)
        
        print(f"Distribución {class_name}: {np.sum(y_binary)}/{len(y_binary)}")
        
        # Entrenar modelo binario
        w_model, loss_train, loss_test = training(
            X_train, y_binary, epochs, alpha, X_test, y_test_binary
        )
        
        models[i] = w_model
    
    return models

def predict_one_vs_all(X_test, models):
    n_samples = X_test.shape[0]
    n_classes = len(models)
    
    probabilities = np.zeros((n_samples, n_classes))
    
    for i, w_model in models.items():
        probabilities[:, i] = s(X_test, w_model)  
    
    y_pred = np.argmax(probabilities, axis=1)
    
    return y_pred, probabilities