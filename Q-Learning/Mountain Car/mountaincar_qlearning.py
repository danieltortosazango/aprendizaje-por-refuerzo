import gymnasium as gym
import numpy as np

# Crear el entorno
env = gym.make('MountainCar-v0')

# Definir los parámetros de Q-Learning
num_estados = env.observation_space.shape[0]
num_acciones = env.action_space.n
num_episodios = 10000

# Definir los hiperparámetros de Q-Learning
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 1.0  # Exploración inicial
epsilon_decay = 0.999  # Decaimiento de la exploración
epsilon_min = 0.01  # Exploración mínima

# Inicializar la tabla Q como un diccionario
q_table = {}

# Cargar la tabla Q desde el archivo si existe
try:
    q_table = np.load("q_table.npy", allow_pickle=True).item()
    print("Tabla Q cargada desde el archivo.")
except FileNotFoundError:
    print("Archivo de tabla Q no encontrado. Se utilizará una tabla Q vacía.")

recompensas_entrenamiento = []

# Función de política epsilon-greedy
def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(num_acciones)
    else:
        return np.argmax(q_table.get(state, np.zeros(num_acciones)))
    

recompensa_total = 0

# Bucle de entrenamiento
for episodio in range(num_episodios + 1):
    estado = env.reset()[0]
    estado = tuple(np.round(estado, decimals=2))  # Redondear los valores del estado y convertirlo en una tupla
    total_recompensa = 0
    done = False
    truncated = False

    while not done and not truncated:
        # Seleccionar una acción según la política epsilon-greedy
        accion = epsilon_greedy_policy(estado, epsilon)

        # Tomar la acción y obtener el siguiente estado, recompensa y si el episodio ha terminado
        siguiente_estado, recompensa, done, truncated, _ = env.step(accion)

        total_recompensa += recompensa

        siguiente_estado = tuple(np.round(siguiente_estado, decimals=2))  # Redondear los valores del siguiente estado y convertirlo en una tupla

        # Actualizar la tabla Q utilizando la ecuación de Q-Learning
        q_valor_actual = q_table.get(estado, np.zeros(num_acciones))[accion]
        max_q_valor_siguiente = np.max(q_table.get(siguiente_estado, np.zeros(num_acciones)))
        q_valor_nuevo = (1 - alpha) * q_valor_actual + alpha * (recompensa + gamma * max_q_valor_siguiente)
        
        if estado not in q_table:
            q_table[estado] = np.zeros(num_acciones)
        q_table[estado][accion] = q_valor_nuevo

        estado = siguiente_estado

    
    if episodio != 0:
        recompensa_total += total_recompensa

    if episodio % 100 == 0 and episodio != 0: #cada 100 episodios imprimimimos el tiempo promedio y su recompensa promedia
        recompensa_promedia = recompensa_total / 100 #como es la recompensa de 100 episodios hacemos la media
        recompensas_entrenamiento.append(recompensa_promedia)
        print("Recompensa promedia en pasar 100 episodios: " + str(recompensa_promedia))
        recompensa_total = 0 #reseteamos la recompensa total

    # Reducir la exploración (epsilon) con el tiempo
    epsilon *= epsilon_decay
    epsilon = max(epsilon, epsilon_min)

# Guardar la tabla Q en un archivo
np.save("q_table.npy", q_table)
print("Tabla Q guardada en el archivo 'q_table.npy'.")

with open("recompensas_mountaincar_qlearning.txt", "w") as txt_file:
    for line in recompensas_entrenamiento:
        txt_file.write(str(line) + "\n") #escribimos cada recompensa en una linea del txt

# Cerrar el entorno
env.close()
