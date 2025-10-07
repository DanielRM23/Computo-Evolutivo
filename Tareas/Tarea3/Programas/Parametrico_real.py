import numpy as np 
import random
import matplotlib.pyplot as plt



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
def torneo(poblacion, poblacion_evaluada, k):
    # poblacion_evaluada: vector con los fitness de cada individuo
    indices = np.random.choice(len(poblacion_evaluada), k, replace=False)
    # índice con el fitness mínimo
    minimo = np.argmin(poblacion_evaluada[indices])
    ganador = indices[minimo]
    return poblacion[ganador]


# ------------ Cruza ------------

def cruza(instancia, padre, madre):
    rangos = np.array(instancia["variables_rango"])
    lows  = rangos[:,0]
    highs = rangos[:,1]

    hijo1 = []
    hijo2 = []
    for (gen_padre, gen_madre) in zip(padre, madre):
        alpha = np.random.uniform(-0.25, 1.25)  # α distinto por gen
        beta = np.random.uniform(-0.25, 1.25) 

        gen_hijo_1 = gen_padre + alpha*(gen_madre-gen_padre)
        gen_hijo_2 = gen_madre + beta*(gen_padre-gen_madre)
        
        hijo1.append(gen_hijo_1)
        hijo2.append(gen_hijo_2)

    hijo1 = np.array(hijo1)
    hijo1 = np.clip(hijo1, lows, highs)  # recortar al rango permitido

    hijo2 = np.array(hijo2)
    hijo2 = np.clip(hijo2, lows, highs)  # recortar al rango permitido

    return hijo1, hijo2


# ------------ Mutación ------------

def mutacion(instancia, hijo, m): 
    
    numero_genes = len(hijo)
    p = 1/numero_genes # proba por cada gen 
    
    rangos = np.array(instancia["variables_rango"])
    hijo_mutado = []

    for i, gen in enumerate(hijo):
        if np.random.rand() < p: #aquí muta
            
            low  = rangos[i,0]
            high = rangos[i,1]
            rango = 0.5* (high - low)
            
            delta = 0            
            for j in range(0, m):
                if np.random.rand() < 1/m: 
                    delta += 2**(-j)
                
            s = np.random.choice([-1,1])
            nuevo_gen = gen + s*rango*delta

            nuevo_gen = np.clip(nuevo_gen, low, high)
            hijo_mutado.append(nuevo_gen) 
        else:
            hijo_mutado.append(gen)
    
    return hijo_mutado

# ------------ Ciclo del algoritmo ------------

def ciclo_algoritmo(instancia, e, m):
    """
    Ejecuta el GA con elitismo de tamaño e y mutación tipo 'non-uniform' por bits (m).
    Devuelve: (best_x, best_f, historial_best)
    """
    decimales = instancia["precision"]
    M = instancia["pob_size"]
    G = instancia["num_gen"]
    p_cruce = instancia["p_cruce"]
    f = instancia["f"]

    # 1) Población inicial
    poblacion = generar_poblacion(instancia)  # (M, n)

    historial_best = []

    for _ in range(G):
        # 2) Evaluación
        n = len(instancia["variables_rango"])
        if n == 2:
            fitness = evaluar_fitness_2D(instancia, poblacion)
        else:
            fitness = evaluar_fitness_5D(instancia, poblacion)

        # 3) Elitismo
        idx_orden = np.argsort(fitness)           # menor = mejor
        idx_elite = idx_orden[:e]
        elite      = poblacion[idx_elite].copy()
        elite_fit  = fitness[idx_elite].copy()

        # registrar mejor
        historial_best.append(elite_fit[0])

        # 4) Reproducción hasta completar M - e hijos
        hijos = []
        while len(hijos) < M - e:
            padre = torneo(poblacion, fitness, k=5)
            madre = torneo(poblacion, fitness, k=5)

            if np.random.rand() < p_cruce:
                h1, h2 = cruza(instancia, padre, madre)
            else:
                h1, h2 = padre.copy(), madre.copy()

            h1 = mutacion(instancia, h1, m)
            h2 = mutacion(instancia, h2, m)

            # redondeo a la precisión deseada
            h1 = np.round(h1, decimals=decimales)
            h2 = np.round(h2, decimals=decimales)

            hijos.append(h1)
            if len(hijos) < M - e:
                hijos.append(h2)

        hijos = np.array(hijos, dtype=float)

        # 5) Nueva generación = élite + hijos
        poblacion = np.vstack([elite, hijos])

    # evaluación final
    n = len(instancia["variables_rango"])
    if n == 2:
        fitness = evaluar_fitness_2D(instancia, poblacion)
    else:
        fitness = evaluar_fitness_5D(instancia, poblacion)

    best_idx = np.argmin(fitness)
    best_x   = poblacion[best_idx]
    best_f   = fitness[best_idx]
    return best_x, best_f, np.array(historial_best)
