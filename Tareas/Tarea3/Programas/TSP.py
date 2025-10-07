import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# ======================================= PARÁMETROS GLOBALES =======================================
matriz_distancias = pd.read_csv('Matriz_de_Distancias.csv', header=None)
matriz_distancias = matriz_distancias.drop([0], axis=0).drop([0], axis=1).reset_index(drop=True)
matriz_distancias = matriz_distancias.astype(float)
matriz_distancias = matriz_distancias.values

N_CITIES = matriz_distancias.shape[0]

# Parámetros del AG
POP_SIZE = 100
GENERATIONS = 500
TOURNAMENT_SIZE = 3
MUT_RATE = 0.2
MAX_NO_IMPROVE = 50
EXPERIMENTS = 10


# ======================================= FUNCIONES DE EVALUACIÓN ========================================

def ruta_cost(ruta):
    total = 0
    for i in range(N_CITIES):
        total += matriz_distancias[ruta[i], ruta[(i + 1) % N_CITIES]]  
    return total

# ======================================= CODIFICACIÓN A USAR =======================================

def decode(chrom):
    if codificacion == 'entero':
        return chrom
    elif codificacion == 'real':
        return np.argsort(chrom)
    elif codificacion == 'binario':
        num_bits = int(np.ceil(np.log2(N_CITIES)))
        values = [int(''.join(str(b) for b in chrom[i * num_bits:(i + 1) * num_bits]), 2) % N_CITIES for i in range(N_CITIES)]
        _, perm = np.unique(values, return_index=True)  
        ruta = np.array(values)[sorted(perm)]
        missing = [c for c in range(N_CITIES) if c not in ruta]
        return np.concatenate([ruta, missing])
        

# ====== INICIALIZACIÓN DE LA POBLACIÓN =========

def inicializar_poblacion():
    if codificacion == 'entero':
        return [np.random.permutation(N_CITIES) for _ in range(POP_SIZE)]
    elif codificacion == 'real':
        return [np.random.rand(N_CITIES) for _ in range(POP_SIZE)]
    elif codificacion == 'binario':
        num_bits = int(np.ceil(np.log2(N_CITIES)))
        return [np.random.randint(0, 2, N_CITIES * num_bits) for _ in range(POP_SIZE)]


# ====== SELECCIÓN POR TORNEO ======

def torneo(pop, costs):
    selected = np.random.choice(len(pop), TOURNAMENT_SIZE, replace=False)
    winner = selected[np.argmin([costs[i] for i in selected])]
    return pop[winner]


# ====== OPERADORES DE CRUZA ======

def cruza(padre, madre):
    size = len(padre)
    if codificacion == 'entero':
        a, b = sorted(np.random.choice(range(size), 2, replace=False))
        hijo = -np.ones(size, dtype=int)
        hijo[a:b+1] = padre[a:b+1]
        fill = [city for city in madre if city not in hijo]
        idx = 0
        for i in range(size):
            if hijo[i] == -1:
                hijo[i] = fill[idx]
                idx += 1
        return hijo
    else:
        cp = np.random.randint(1, size - 1)
        return np.concatenate([padre[:cp], madre[cp:]])


# ====== OPERADORES DE MUTACIÓN ======

def mutacion_real(hijo, m=20): 
    """
    Usa la variable global N_CITIES para los rangos.
    """
    numero_genes = len(hijo)
    p = 1.0 / numero_genes # Probabilidad de mutar cada gen
    
    # Los rangos para TSP con esta codificación son siempre [0, 1]
    # si la población se genera con np.random.rand(), o [0, N_CITIES]
    # si se usa np.random.uniform(0, N_CITIES).
    # Asumimos [0, 1] por simplicidad.
    low = 0.0
    high = 1.0
    rango_mutacion = 0.5 * (high - low)
    
    hijo_mutado = []
    for gen in hijo:
        if np.random.rand() < p:
            delta = 0.0
            for j in range(m):
                if np.random.rand() < (1.0 / m): 
                    delta += 2**(-j)
            
            s = np.random.choice([-1, 1])
            nuevo_gen = gen + s * rango_mutacion * delta

            # Aseguramos que el nuevo gen no se salga del rango [0, 1]
            nuevo_gen = np.clip(nuevo_gen, low, high)
            hijo_mutado.append(nuevo_gen)
        else:
            hijo_mutado.append(gen)
    
    return np.array(hijo_mutado)


def mutacion(cromosoma):
    if np.random.rand() < MUT_RATE:
        if codificacion == 'entero':
            i, j = np.random.choice(N_CITIES, 2, replace=False)
            cromosoma[i], cromosoma[j] = cromosoma[j], cromosoma[i]
        
        elif codificacion == 'real':
            return mutacion_real(cromosoma)
            
        elif codificacion == 'binario':
            num_bits = int(np.ceil(np.log2(N_CITIES)))
            i, j = np.random.choice(N_CITIES, 2, replace=False)
            start_i, start_j = i * num_bits, j * num_bits
            # Es importante usar .copy() al intercambiar slices de numpy
            gene_j_copy = cromosoma[start_j:start_j + num_bits].copy()
            cromosoma[start_j:start_j + num_bits] = cromosoma[start_i:start_i + num_bits]
            cromosoma[start_i:start_i + num_bits] = gene_j_copy
            
    # Devuelve el cromosoma sin cambios si no se cumplió la prob. de mutación
    return cromosoma


# ======================================= BUCLE PRINCIPAL DEL AG =======================================

resultados_tsp = {}
NUM_EJECUCIONES = 30 

# Itera sobre cada tipo de codificación
for codificacion in ['entero', 'real', 'binario']:
    codificacion = codificacion # Establece la codificación global para esta serie de experimentos
    nombre_exp = f"TSP_{codificacion}"
    resultados_tsp[nombre_exp] = []
    
    print(f"==========================================================")
    print(f"-> EJECUTANDO EXPERIMENTO: {nombre_exp} ({NUM_EJECUCIONES} veces)")

    for exp in range(NUM_EJECUCIONES):
        print(f"   Corrida {exp+1}/{NUM_EJECUCIONES}...", end='\r')
        random.seed(exp)
        np.random.seed(exp)

        poblacion = inicializar_poblacion()
        best_cost = float('inf')
        cost_history = []

        for gen in range(GENERATIONS):
            decoded = [decode(c) for c in poblacion]
            costs = [ruta_cost(r) for r in decoded]
            
            current_best_cost = np.min(costs)
            if current_best_cost < best_cost:
                best_cost = current_best_cost
            
            cost_history.append(best_cost) # Guardamos el mejor costo hasta ahora

            elite_idx = np.argmin(costs)
            elite = poblacion[elite_idx]
            next_gen = [elite.copy()]
            while len(next_gen) < POP_SIZE:
                padre = torneo(poblacion, costs)
                madre = torneo(poblacion, costs)
                hijo = cruza(padre, madre)
                hijo = mutacion(hijo)
                next_gen.append(hijo)
            poblacion = next_gen
            
        resultados_tsp[nombre_exp].append(cost_history)
    print(f"\n-> Experimento {nombre_exp} completado. Mejor costo promedio: {np.mean([run[-1] for run in resultados_tsp[nombre_exp]]):.2f}")


# ======================================= GRÁFICAS DE RESULTADOS =======================================

print("\nGENERANDO GRÁFICA PARA TSP...")
plt.figure(figsize=(15, 7))
colores = {'entero': 'blue', 'real': 'green', 'binario': 'red'}

for nombre, historiales in resultados_tsp.items():
    codificacion = nombre.split('_')[1] # Extrae 'entero', 'real', o 'binario'
    
    mean = np.mean(historiales, axis=0)
    std = np.std(historiales, axis=0)
    
    plt.plot(mean, label=f'Codificación {codificacion.capitalize()}', color=colores[codificacion])
    plt.fill_between(range(len(mean)), mean - std, mean + std, color=colores[codificacion], alpha=0.1)

plt.title(f'Desempeño de Codificaciones para TSP ({N_CITIES} ciudades)\n(Promedio de {NUM_EJECUCIONES} ejecuciones)')
plt.xlabel('Generación')
plt.ylabel('Mejor Costo de Ruta')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f'TSP_{N_CITIES}_ciudades.png')
plt.show()
