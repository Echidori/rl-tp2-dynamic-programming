import numpy as np

from dynamic_programming.grid_world_env import GridWorldEnv
from dynamic_programming.mdp import MDP
from dynamic_programming.stochastic_grid_word_env import StochasticGridWorldEnv

# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.


def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    values = np.zeros(mdp.observation_space.n)
    # BEGIN SOLUTION
    for _ in range(max_iter):
        prev_val = values.copy()
        for state in range(mdp.observation_space.n):
            state_val = []
            for action in range(mdp.action_space.n):
                next_state, reward, _ = mdp.P[state][action]
                value = reward + gamma * prev_val[next_state]
                state_val.append(value)

            values[state] = np.max(state_val)
    # END SOLUTION
    return values


def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    for _ in range(max_iter):
        prev_val = values.copy()
        delta = 0
        for row in range(env.height):
            for col in range(env.width):
                if env.grid[row, col] in  {"P", "N", "W"}:
                    continue
                state_val = []
                for action in range(env.action_space.n):
                    env.set_state(row, col)
                    next_state, reward, end, _ = env.step(action, False)
                    if end:
                        value = reward
                    else:
                        value = reward + gamma * prev_val[next_state[0], next_state[1]]
                    state_val.append(value)
                values[row][col] = np.max(state_val)
        delta = np.max(np.abs(prev_val - values))
        if delta < theta:
            break
    return values
    # END SOLUTION


def value_iteration_per_state(env, values, gamma, prev_val, delta):
    row, col = env.current_position
    values[row, col] = float("-inf")
    for action in range(env.action_space.n):
        next_states = env.get_next_states(action=action)
        current_sum = 0
        for next_state, reward, probability, _, _ in next_states:
            # print((row, col), next_state, reward, probability)
            next_row, next_col = next_state
            current_sum += (
                probability
                * env.moving_prob[row, col, action]
                * (reward + gamma * prev_val[next_row, next_col])
            )
        values[row, col] = max(values[row, col], current_sum)
    delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))
    return delta


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma: float = 1.0,
    theta: float = 1e-5,
) -> np.ndarray:
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    delta = float("inf")
    for _ in range(max_iter):
        prev_val = values.copy()
        for row in range(env.height):
            for col in range(env.width):
                env.set_state(row, col)
                delta = value_iteration_per_state(env, values, gamma, prev_val, delta)
        if delta < theta:
            break
    return values
