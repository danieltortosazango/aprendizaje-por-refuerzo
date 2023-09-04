import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import imageio


array_recompensas = []

# Definir la red neuronal
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Definir el agente DQN
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return q_values.argmax(1).item()

    def train(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return

        batch = random.sample(replay_buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, truncated_batch = zip(*batch)

        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
        truncated_batch = torch.tensor(truncated_batch, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.model(state_batch).gather(1, action_batch)
        next_q_values = self.target_model(next_state_batch).max(1)[0].unsqueeze(1)
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = self.loss_fn(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

# Parámetros del agente
lr = 0.001  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 1.0  # Valor de epsilon inicial
epsilon_decay = 0.995  # Factor de decaimiento de epsilon
replay_buffer_size = 10000  # Tamaño del buffer de experienciaspython 
batch_size = 64  # Tamaño del lote para el entrenamiento

# Crear el entorno CartPole
env = gym.make("CartPole-v1", render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Crear el agente DQN
agent = DQNAgent(state_dim, action_dim, lr, gamma, epsilon, epsilon_decay)

# Inicializar el buffer de experiencias
replay_buffer = []
numero_episodios = 10


recompensa_total = 0

# Comprobar si existe el archivo con la red neuronal entrenada
if os.path.exists("dqn_model.pth"):
    agent.model.load_state_dict(torch.load("dqn_model.pth"))
    agent.target_model.load_state_dict(agent.model.state_dict())
    print("Red neuronal cargada desde el archivo 'dqn_model.pth'.")
else:
    print("Creando nueva red neuronal.")


output_folder = "output_episodios"
os.makedirs(output_folder, exist_ok=True)

# Entrenamiento del agente
for episode in range(numero_episodios + 1):
    state = env.reset()[0]
    done = False
    truncated = False
    total_reward = 0
    episode_frames = []  # Almacenar frames del episodio para visualización

    while not done and not truncated:
        # Seleccionar acción
        action = agent.select_action(state)

        # Tomar acción y obtener siguiente estado, recompensa y done
        next_state, reward, done, truncated, _ = env.step(action)

        # Almacenar transición en el buffer de experiencias
        replay_buffer.append((state, action, reward, next_state, done, truncated))

        # Actualizar estado actual
        state = next_state

        # Realizar un paso de entrenamiento
        agent.train(replay_buffer, batch_size)

        # Actualizar recompensa acumulada
        total_reward += reward

        # Almacenar el frame actual en la lista de frames del episodio
        episode_frames.append(env.render())

    # Actualizar epsilon
    agent.decay_epsilon()

    # Guardar el episodio como un GIF usando imageio
    if episode_frames:
        gif_file = os.path.join(output_folder, f"episode_{episode:03d}_recompensa_{total_reward:f}.gif")
        imageio.mimsave(gif_file, episode_frames, duration=0.03)  # Duración de cada frame en segundos


    #print("Episodio:", episode, "Recompensa:", total_reward)

# Cerrar el entorno
env.close()
