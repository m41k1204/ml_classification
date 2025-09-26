import numpy as np

def initialize_weights(n_features, n_classes):
    np.random.seed(42)  
    return np.random.randn(n_features, n_classes) * 0.01
def h(x, w):
    return np.dot(x, w)

def s(x, w):
    exp_scores = np.exp(h(x, w))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def Loss_function(x, y, w):
    probabilities = s(x, w)  
    n_samples = x.shape[0]
    correct_class_probs = probabilities[np.arange(n_samples), y]
    epsilon = 1e-15  
    return -np.mean(np.log(correct_class_probs + epsilon))

def Derivatives(x, y, w):
    n_samples = x.shape[0]
    probabilities = s(x, w)
    
    probabilities[np.arange(n_samples), y] -= 1
    
    gradients = (1/n_samples) * np.dot(x.T, probabilities)
    return gradients

def change_parameters(w, derivatives, alpha):
  return w - alpha * derivatives

def train(x_train, y_train, epochs, alpha, x_test, y_test):
    print(f"Número de muestras de entrenamiento: {len(x_train)}")
    
    n_features = x_train.shape[1]
    n_classes = len(np.unique(y_train))  
    w_train = initialize_weights(n_features, n_classes)
    
    LossTrain = []
    LossTest = []
    
    for i in range(epochs):
        L_Train = Loss_function(x_train, y_train, w_train)
        L_Test = Loss_function(x_test, y_test, w_train)
        
        dW = Derivatives(x_train, y_train, w_train)
        w_train = change_parameters(w_train, dW, alpha)
        
        if i %100 == 0:
            print(f"Época {i+1}: L_Train = {L_Train:.4f}")
        
        LossTrain.append(L_Train)
        LossTest.append(L_Test)
    
    return w_train, LossTrain, LossTest

def test(x_test, y_test, w_train):
    
    y_pred_probs = s(x_test, w_train)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    correctos = np.sum(y_pred == y_test)
    total = len(y_test)
    porc_aciertos = (correctos / total) * 100
    
    print(f"Número de datos correctos: {correctos}")
    print(f"Porcentaje de aciertos: {porc_aciertos:.2f}%")
    print(f"Porcentaje de error: {100 - porc_aciertos:.2f}%")
    
    return y_pred, porc_aciertos

