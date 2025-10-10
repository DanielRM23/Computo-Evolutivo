import copy
import random

class Nodo: 
    def __init__(self, valor, hijos): 
        """Constructor de la clase Nodo.
        Args: 
            valor: El valor del nodo; AND, OR, NOT, A, B, C.
            hijos: Una lista de nodos hijos.
        """
        self.valor = valor 
        self.hijos = hijos if hijos is not None else []

    def agregar_hijo(self, hijo): 
        """Agrega un hijo al nodo.
        Args: 
            hijo: El nodo hijo a agregar.
        """
        self.hijos.append(hijo)
    

class Individuo:
    def __init__(self, nodo_raiz):
        """Constructor de la clase Individuo.
        
        Args:
            nodo_raiz: El nodo raíz del árbol que representa este individuo.
        """
        self.nodo_raiz = nodo_raiz
        self.fitness = 0

    def evaluar(self, valores_entrada):
        """Evalúa el árbol del individuo completo.
        
        Args:
            valores_entrada: Un diccionario con los valores de las variables,
                             ej: {'A': 1, 'B': 0, 'C': 1}.
                             
        Returns:
            El resultado booleano (0 o 1) de la evaluación.
        """
        return self._evaluar_recursivo(self.nodo_raiz, valores_entrada)


    def _evaluar_recursivo(self, nodo_actual, valores_entrada):
        """Función auxiliar para evaluar el árbol de forma recursiva."""
        
        # Caso Base: Si el nodo es un terminal (A, B, o C).
        if nodo_actual.valor in ['A', 'B', 'C']:
            return valores_entrada[nodo_actual.valor]

        # Caso Recursivo: Si el nodo es una función.
        # 1. Evalúa los hijos primero para obtener sus resultados.
        resultados_hijos = []
        for hijo in nodo_actual.hijos:
            #print(f"Evaluando hijo: {hijo.valor}")
            resultado_hijo = self._evaluar_recursivo(hijo, valores_entrada)
            resultados_hijos.append(resultado_hijo)

        # 2. Aplica la operación del nodo actual a los resultados de los hijos.
        if nodo_actual.valor == 'AND':
            return 1 if resultados_hijos[0] == 1 and resultados_hijos[1] == 1 else 0
        
        elif nodo_actual.valor == 'OR':
            return 1 if resultados_hijos[0] == 1 or resultados_hijos[1] == 1 else 0
            
        elif nodo_actual.valor == 'XOR':
            return 1 if resultados_hijos[0] != resultados_hijos[1] else 0
            
        elif nodo_actual.valor == 'NOT':
            return 1 if resultados_hijos[0] == 0 else 0
        

    def __str__(self):
        """
        Devuelve una representación en string del árbol (notación LISP).
        Cuando llames a print(tu_individuo), Python ejecutará este método.
        """
        return self._imprimir_recursivo(self.nodo_raiz)

    def _imprimir_recursivo(self, nodo_actual):
        """Función auxiliar que recorre el árbol para construir el string."""

        # Caso Base: Si el nodo es un terminal, solo devuelve su valor.
        if not nodo_actual.hijos:
            return nodo_actual.valor
        
        # Caso Recursivo: Si es una función, construye la expresión.
        # Primero, obtiene la representación de todos sus hijos.
        hijos_str = " ".join([self._imprimir_recursivo(hijo) for hijo in nodo_actual.hijos])
        
        # Luego, los une en el formato (FUNCION HIJO1 HIJO2 ...)
        return f"({nodo_actual.valor} {hijos_str})"


def crear_arbol_aleatorio(profundidad_maxima, profundidad_actual, conjunto_funciones):
    
    # Si alcanzamos la profundidad máxima, o si decidimos al azar, creamos un terminal.
    if profundidad_actual == profundidad_maxima or random.random() < 0.4:
        valor_aleatorio = random.choice(TERMINALES) # Se elige un valor de los terminales
        return Nodo(valor=valor_aleatorio, hijos=[]) # Aquí se crea el nodo, con el terminal elegido en la línea anterior. 
                # El decir que los hijos de este nodo son [], se da a entender que no tiene hijos, por lo que es una hoja

    # Si no, creamos una función
    else:
        # Elegimos una función al azar del conjunto que nos pasen   
        nombre_funcion = random.choice(list(conjunto_funciones.keys())) #Tomamos una función al azar
        aridad = conjunto_funciones[nombre_funcion]
        
        # Creamos el nodo para esta función, de momento no tiene hijos
        nodo_funcion = Nodo(valor=nombre_funcion, hijos=[])
        
        # Aquí se crean recursivamente sus hijos
        for _ in range(aridad):
            # Aumentamos en 1 la profundidad, hasta llegar a la proundidad máxima
            hijo = crear_arbol_aleatorio(profundidad_maxima, profundidad_actual + 1, conjunto_funciones)
            nodo_funcion.agregar_hijo(hijo) # Agregamos al hijo
            
        return nodo_funcion
    


def calcular_fitness(individuo):
    """Calcula el fitness de un individuo contando los aciertos en la tabla de verdad."""
    aciertos = 0
    for fila in TABLA_VERDAD: # Tabla de verdad es una constante, y es la tabla que nos dan
        # Extraemos los valores de entrada y el resultado esperado
        valores_entrada = {'A': fila['A'], 'B': fila['B'], 'C': fila['C']}
        salida_esperada = fila['S']
        
        try:
            # Evaluamos el individuo con las entradas actuales
            resultado_evaluado = individuo.evaluar(valores_entrada)
            
            # Si el resultado es el esperado, contamos un acierto
            if resultado_evaluado == salida_esperada:
                aciertos += 1
        except (IndexError, TypeError):
            # Si el árbol está mal formado (raro, pero posible), su fitness es 0
            # Casos particulares, pues, por si acaso lo dejo en este bloque except
            aciertos = 0
            break
            
    individuo.fitness = aciertos # El fitness ahora es el número de aciertos


def seleccion_por_torneo(poblacion, tamaño_torneo=3):
    """
    Selecciona un individuo de la población mediante el método de torneo.
    """
    # Se elige un grupo de competidores al azar de la población
    competidores = random.sample(poblacion, tamaño_torneo)
    
    # Se encuentra al competidor con el mejor fitness
    ganador = max(competidores, key=lambda individuo: individuo.fitness)
    
    return ganador

def _obtener_nodos(nodo_raiz):
    """Función auxiliar para obtener una lista de todos los nodos de un árbol."""
    nodos = []
    _obtener_nodos_recursivo(nodo_raiz, nodos)
    return nodos


def _obtener_nodos_recursivo(nodo_actual, lista_nodos):
    """Recorre el árbol y agrega cada nodo a la lista."""
    if nodo_actual is not None:
        lista_nodos.append(nodo_actual)
        for hijo in nodo_actual.hijos:
            _obtener_nodos_recursivo(hijo, lista_nodos)



def cruzamiento(padre1, padre2):
    """
    Crea dos hijos combinando los árboles de dos padres mediante el
    intercambio de subárboles.
    """
    hijo1 = Individuo(copy.deepcopy(padre1.nodo_raiz))
    hijo2 = Individuo(copy.deepcopy(padre2.nodo_raiz))

    # Obtenemos una lista de todos los nodos de cada hijo
    nodos_hijo1 = _obtener_nodos(hijo1.nodo_raiz)
    nodos_hijo2 = _obtener_nodos(hijo2.nodo_raiz)

    # Elegimos un nodo al azar de cada árbol para el intercambio
    nodo_a_reemplazar_en_hijo1 = random.choice(nodos_hijo1)
    subarbol_de_hijo2 = random.choice(nodos_hijo2)

    # Guardamos una copia del subárbol que vamos a reemplazar en el hijo1
    subarbol_original_de_hijo1 = copy.deepcopy(nodo_a_reemplazar_en_hijo1)

    # Reemplazamos el nodo en hijo1 con el subárbol de hijo2
    # Esto es conceptualmente correcto: se reemplaza todo el subárbol
    nodo_a_reemplazar_en_hijo1.valor = subarbol_de_hijo2.valor
    nodo_a_reemplazar_en_hijo1.hijos = subarbol_de_hijo2.hijos

    # Ahora, encontramos un punto de cruce en el hijo2 y lo reemplazamos
    # con el subárbol que guardamos del hijo1
    nodo_a_reemplazar_en_hijo2 = random.choice(nodos_hijo2)
    nodo_a_reemplazar_en_hijo2.valor = subarbol_original_de_hijo1.valor
    nodo_a_reemplazar_en_hijo2.hijos = subarbol_original_de_hijo1.hijos

    return hijo1, hijo2


def calcular_profundidad(nodo_raiz):
    """Calcula la profundidad de un árbol de forma recursiva."""
    if not nodo_raiz or not nodo_raiz.hijos:
        return 0
    else:
        return 1 + max(calcular_profundidad(hijo) for hijo in nodo_raiz.hijos)
    

def mutacion(individuo, profundidad_maxima, conjunto_funciones):
    """
    Muta un individuo reemplazando un nodo aleatorio con un nuevo sub-árbol.
    """
    individuo_mutado = Individuo(copy.deepcopy(individuo.nodo_raiz))
    
    nodos = _obtener_nodos(individuo_mutado.nodo_raiz)
    nodo_a_mutar = random.choice(nodos)
    
    # Creamos un nuevo sub-árbol aleatorio
    profundidad_actual = 0 # Simplificación, idealmente se calcularía la profundidad real
    nuevo_sub_arbol = crear_arbol_aleatorio(profundidad_maxima, profundidad_actual, conjunto_funciones)
    
    # Reemplazamos el nodo seleccionado
    nodo_a_mutar.valor = nuevo_sub_arbol.valor
    nodo_a_mutar.hijos = nuevo_sub_arbol.hijos
    
    return individuo_mutado


# =======================================================
# EL BUCLE PRINCIPAL
# =======================================================
if __name__ == "__main__":
    
    # --- PASO 1: Configuración del Algoritmo ---
    TAMAÑO_POBLACION = 300      # Aumentado para mayor diversidad
    NUM_GENERACIONES = 300
    PROB_CRUZAMIENTO = 0.9
    PROB_MUTACION = 0.15        # Ligeramente aumentado para salir de óptimos locales
    TAMAÑO_TORNEO = 3           # Reducido para disminuir presión de selección
    PROFUNDIDAD_MAXIMA = 8
    ELITISMO = True
    TERMINALES = ['A', 'B', 'C']

    TABLA_VERDAD = [
        {'A': 0, 'B': 0, 'C': 0, 'S': 1},
        {'A': 0, 'B': 0, 'C': 1, 'S': 0},
        {'A': 0, 'B': 1, 'C': 0, 'S': 0},
        {'A': 0, 'B': 1, 'C': 1, 'S': 1},
        {'A': 1, 'B': 0, 'C': 0, 'S': 0},
        {'A': 1, 'B': 0, 'C': 1, 'S': 1},
        {'A': 1, 'B': 1, 'C': 0, 'S': 1},
        {'A': 1, 'B': 1, 'C': 1, 'S': 0},
    ]

    # --- PASO 2: Definición de los Experimentos ---
    experimentos = [
        {
            "nombre": "Experimento 1: F1 = {AND, OR, NOT}",
            "funciones": {'AND': 2, 'OR': 2, 'NOT': 1}
        },
        {
            "nombre": "Experimento 2: F2 = {AND, OR, NOT, XOR}",
            "funciones": {'AND': 2, 'OR': 2, 'NOT': 1, 'XOR': 2}
        }
    ]

    # --- PASO 3: Bucle Principal de Experimentos ---
    for exp in experimentos:
        print(f"\n=================================================")
        print(f"INICIANDO {exp['nombre']}")
        print(f"=================================================")

        poblacion = []
        for _ in range(TAMAÑO_POBLACION):
            nodo_raiz = crear_arbol_aleatorio(PROFUNDIDAD_MAXIMA, 0, exp['funciones'])
            poblacion.append(Individuo(nodo_raiz))

        mejor_solucion_global = None

        for gen in range(NUM_GENERACIONES):
            for individuo in poblacion:
                calcular_fitness(individuo)

            mejor_de_la_generacion = max(poblacion, key=lambda ind: ind.fitness)
            if mejor_solucion_global is None or mejor_de_la_generacion.fitness > mejor_solucion_global.fitness:
                mejor_solucion_global = copy.deepcopy(mejor_de_la_generacion) 

            print(f"Generación {gen+1}/{NUM_GENERACIONES} | Mejor Fitness: {mejor_de_la_generacion.fitness}/8")

            if mejor_de_la_generacion.fitness == 8:
                print(f"\n¡SOLUCIÓN ENCONTRADA EN LA GENERACIÓN {gen+1}!")
                break

            # --- BUCLE DE CREACIÓN DE POBLACIÓN CORREGIDO ---
            nueva_poblacion = []

            if ELITISMO:
                nueva_poblacion.append(copy.deepcopy(mejor_de_la_generacion))
            
            while len(nueva_poblacion) < TAMAÑO_POBLACION:
                padre1 = seleccion_por_torneo(poblacion, TAMAÑO_TORNEO)
                padre2 = seleccion_por_torneo(poblacion, TAMAÑO_TORNEO)
                
                if random.random() < PROB_CRUZAMIENTO:
                    hijo1, hijo2 = cruzamiento(padre1, padre2)
                else:
                    hijo1, hijo2 = copy.deepcopy(padre1), copy.deepcopy(padre2)
                
                # Mutación aplicada con probabilidad
                if random.random() < PROB_MUTACION:
                    hijo1 = mutacion(hijo1, PROFUNDIDAD_MAXIMA, exp['funciones'])
                
                if random.random() < PROB_MUTACION:
                    hijo2 = mutacion(hijo2, PROFUNDIDAD_MAXIMA, exp['funciones'])
                
                # Control de profundidad y adición a la nueva población
                if calcular_profundidad(hijo1.nodo_raiz) <= PROFUNDIDAD_MAXIMA + 2:
                    nueva_poblacion.append(hijo1)
                else:
                    nueva_poblacion.append(copy.deepcopy(padre1))

                if len(nueva_poblacion) < TAMAÑO_POBLACION:
                    if calcular_profundidad(hijo2.nodo_raiz) <= PROFUNDIDAD_MAXIMA + 2:
                        nueva_poblacion.append(hijo2)
                    else:
                        nueva_poblacion.append(copy.deepcopy(padre2))
            
            poblacion = nueva_poblacion
        
        # --- PASO 6: Resultados Finales del Experimento ---
        print("\n--- RESULTADO FINAL DEL EXPERIMENTO ---")
        if mejor_solucion_global and mejor_solucion_global.fitness == 8:
            print(f"Solución perfecta encontrada:")
            print(f"Fórmula: {mejor_solucion_global}")
            print(f"Fitness: {mejor_solucion_global.fitness}/8")
        else:
            print(f"No se encontró una solución perfecta en {NUM_GENERACIONES} generaciones.")
            if mejor_solucion_global:
                print(f"La mejor fórmula encontrada fue:")
                print(f"Fórmula: {mejor_solucion_global}")
                print(f"Fitness: {mejor_solucion_global.fitness}/8")