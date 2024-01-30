'''
    This file contains all the function used to generate graphic representations
    used to construct the presentation PowerPoint used to present
    the assignment.
'''

import numpy as np
import matplotlib.pyplot as plt

def generate_domain(max_x, max_t, num_points_x, num_points_t):
    # Genera le coordinate per l'asse x e t
    x_values = np.linspace(0, max_x, num_points_x)
    t_values = np.linspace(0, max_t, num_points_t)

    # Plot delle linee della suddivisione
    for i in range(1, num_points_t):
        plt.plot([0, max_x], [t_values[i], t_values[i]], color='black', linestyle='-', linewidth=0.5)

    for j in range(1, num_points_x):
        plt.plot([x_values[j], x_values[j]], [0, max_t], color='black', linestyle='-', linewidth=0.5)

    # Imposta il limite degli assi
    plt.xlim(0, max_x)
    plt.ylim(0, max_t)

    # Etichette degli assi
    plt.xlabel('Lenght rod')
    plt.ylabel('Time')

    # Titolo del grafico
    plt.title('Domain representation')

    # Mostra il grafico
    plt.show()

    # Impostazioni personalizzate (modifica questi valori secondo le tue esigenze)
    max_x_value = 20
    max_t_value = 30
    num_points_x_axis = 50
    num_points_t_axis = 50

    # Genera e mostra il grafico della suddivisione
    generate_domain(max_x_value, max_t_value, num_points_x_axis, num_points_t_axis)

def schemas():
    p = [1, 2, 4]  # Numero di processi
    t_par = [436, 281.93342730001314, 317.97042690002127]  # Tempi paralleli in secondi
    t_seq = [436, 436, 436]  # Tempo sequenziale in secondi
    t_com = [t_par[0] - t_seq[0] / p[0], t_par[1] - t_seq[1] / p[1], t_par[2] - t_seq[2] / p[2]]
    speed_up = [t_seq[0] / t_par[0], t_seq[1] / t_par[1], t_seq[2] / t_par[2]]  # Speed-up
    efficiency = [speed_up[0] / p[0], speed_up[1] / p[1], speed_up[2] / p[2]]  # Efficienza

    plt.style.use('seaborn')
    
    # Figura 1
    fig1, ax1 = plt.subplots()

    # Plotting del tempo parallelo di esecuzione
    ax1.plot(p, t_seq, label='Sequential Time')
    ax1.plot(p, t_par, label='Parallel Time')
    ax1.plot(p, t_com, label='Communications Time', linestyle='--')

    # Numeri interi lungo l'asse x
    ax1.set_xticks(p)
    ax1.set_xlabel('Number of Processes (p)')
    ax1.set_ylabel('Time (s)')

    # Aggiunta della legenda
    ax1.legend()

    # Evidenziazione dei punti
    ax1.scatter(p, t_seq, marker='o', color='blue')  # Punti per il tempo sequenziale
    ax1.scatter(p, t_par, marker='o', color='green')  # Punti per il tempo parallelo
    ax1.scatter(p, t_com, marker='o', color='red')  # Punti per il tempo di comunicazione

    # Figura 2
    fig2, ax3 = plt.subplots()

    # Plotting dello speed-up e dell'efficienza
    ax3.plot(p, speed_up, label='Speed-up', marker='o')
    ax3.plot(p, efficiency, label='Efficiency', marker='o')

    # Numeri interi lungo l'asse x
    ax3.set_xticks(p)
    ax3.set_xlabel('Number of Processes (p)')
    ax3.set_ylabel('Value')

    # Aggiunta della legenda
    ax3.legend()

    plt.show()

p = [1, 2, 4]  # Numero di processi
t_par = [436, 281.93342730001314, 317.97042690002127]  # Tempi paralleli in secondi
t_seq = [436, 436, 436]  # Tempo sequenziale in secondi
t_com = [t_par[0] - t_seq[0] / p[0], t_par[1] - t_seq[1] / p[1], t_par[2] - t_seq[2] / p[2]]
speed_up = [t_seq[0] / t_par[0], t_seq[1] / t_par[1], t_seq[2] / t_par[2]]  # Speed-up
efficiency = [speed_up[0] / p[0], speed_up[1] / p[1], speed_up[2] / p[2]]  # Efficienza

print(f"{t_com}")
print(f"{speed_up}")
print(f"{efficiency}")