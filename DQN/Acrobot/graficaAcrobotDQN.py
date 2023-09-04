import matplotlib.pyplot as plt

array_recompensas = []
array_posiciones = []
i = 0

valor_mayor = 999


with open("recompensas_acrobot_dqn.txt") as fname:
	lineas = fname.readlines()
	for linea in lineas:
		array_recompensas.append(float(linea.strip('\n')))


for recompensa in array_recompensas:
    array_posiciones.append(i)
    i = i+1

    if(abs(recompensa) < abs(valor_mayor)):
        valor_mayor = recompensa


print("El valor mas alto de recompensa es: " + str(valor_mayor))

#print(array_recompensas)
#print(array_posiciones)
plt.plot(array_posiciones, array_recompensas)
plt.title('Acrobot DQN')
plt.xlabel('Numero de episodios')
plt.ylabel('Valor de recompensa')
plt.show()











