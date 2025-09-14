import numpy as np 
import random
import matplotlib.pyplot as plt




# ======================================= FUNCIONES A PROBAR =======================================


def esfera(*x):
    return np.sum(np.array(x)**2)   # x es un vector (np.array), válido para n=2 o n=5


def bukin(x, y):
    return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)


def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def eggholder(x, y):
    return -(y + 47) * np.sin(np.sqrt(np.abs(x/2 + (y + 47)))) \
           - x * np.sin(np.sqrt(np.abs(x - (y + 47))))


def easom(x, y):
    return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi)**2 + (y - np.pi)**2))



# ============================================ EJECUCIÓN DEL PROGRAMA ============================================

INSTANCIAS = {
    "ESFERA 5-D": dict(f=esfera, variables_rango=[[-10,10],[-10,10],[-10,10],[-10,10],[-10,10]], precision=3, pob_size=30, num_gen=100, p_cruce=0.9),
    "BUKIN":      dict(f=bukin,  variables_rango=[[-15,-5],[-3,3]], precision=3, pob_size=20, num_gen=100, p_cruce=0.3),
    "HIMMELBLAU": dict(f=himmelblau, variables_rango=[[-5,5],[-5,5]], precision=3, pob_size=40, num_gen=200, p_cruce=0.9),
    "EGGHOLDER":  dict(f=eggholder,  variables_rango=[[-512,512],[-512,512]], precision=3, pob_size=250, num_gen= 1000, p_cruce=0.9),
    "EASOM":      dict(f=easom,      variables_rango=[[-100,100],[-100,100]], precision=3, pob_size=250, num_gen=500, p_cruce=0.9),
}

# ----------- Generar población -------------------
# Se genera la población de manera aleatoria sobre un rango de valores 


def generar_poblacion(instancia):
    pob_size = instancia["pob_size"]
    decimales = instancia["precision"]

    rangos = np.array(instancia["variables_rango"], dtype=float)  # shape (n, 2)
    lows  = rangos[:, 0]  # (n,) #Toma toda la primera columna (valores mínimos)
    highs = rangos[:, 1]  # (n,) Toma toda la segunda columna (valores máximos)
    
    X = np.random.uniform(lows, highs, size=(pob_size, len(rangos)))  # (pob_size, n)
    return np.round(X, decimals=decimales)


# ----------- Evaluar población -------------------
# Se evalúa la población con la función objetivo

def evaluar_fitness_2D(instancia, poblacion): # esta es para funciones en 2D
    f = instancia["f"]
    fitness = [f(ind[0], ind[1]) for ind in poblacion]
    return np.array(fitness)


def evaluar_fitness_5D(instancia, poblacion): #esta es para la esfera en 5D
    f = instancia["f"]
    fitness = [f(*ind) for ind in poblacion]  # desempaqueta los 5 valores
    return np.array(fitness)


# ------------ Torneo -------------------
def torneo(poblacion_evaluada, k):
    # poblacion_evaluada: vector con los fitness de cada individuo
    indices = np.random.choice(len(poblacion_evaluada), k, replace=False)
    # índice con el fitness mínimo
    minimo = np.argmin(poblacion_evaluada[indices])
    ganador = indices[minimo]
    return poblacion_evaluada[ganador]



# eggholder = INSTANCIAS["EGGHOLDER"]
# pob_eggholder = generar_poblacion(eggholder)
# poblacion_evaluada_eggholder = evaluar_fitness_5D(eggholder, pob_eggholder)
# print(poblacion_evaluada_eggholder)


esfera = INSTANCIAS["ESFERA 5-D"]
pob_esfera = generar_poblacion(esfera)
poblacion_evaluada_esfera = evaluar_fitness_5D(esfera, pob_esfera)
print(poblacion_evaluada_esfera)


print(torneo(poblacion_evaluada_esfera, k=5))
