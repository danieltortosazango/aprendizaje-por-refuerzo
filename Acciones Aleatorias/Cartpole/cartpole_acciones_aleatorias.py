#importamos la libreria de gym
import gymnasium as gym
#importamos os para crear el txt con los datos de las recompensas
import os

#creamos el entorno
env = gym.make('CartPole-v1')

#ponemos el numero maximo de episodios
MAX_NUM_EPISODES = 1000

array_recompensas = []

for episode in range(MAX_NUM_EPISODES):
    done = False    
    truncated = False
    obs = env.reset()
    total_reward = 0.0; #variable para guardar la recompensa total obtenida en cada episodio (Float)
    step = 0
    while not done and not truncated:
        action = env.action_space.sample() #accion aleatoria
        #next_state, reward, done, info = env.step(action) #ejecutamos la accion aleatoria
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward #vamos acumulando la recompensa
        step += 1
        obs = next_state

    print("\n Episodio numero {} finalizado con {} iteraciones. Recompensa final={}".format(episode, step+1, total_reward))
    array_recompensas.append(total_reward)

env.close()


with open("recompensas_cartpole_acciones_aleatorias.txt", "w") as txt_file:
    for line in array_recompensas:
        txt_file.write(str(line) + "\n") #escribimos cada recompensa en una linea del txt