# Quantum Game Simulator (`qgsim`)

This is a replication of quantum game simulation procedure described by [[Eisert et al. (1999)]](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.83.3077) and [[Du et al. (2002)]](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.88.137902) using [[Qiskit]](https://qiskit.org). The simulator allows to run two-player games in normal form in which each player has two strategies and the payoffs can be represented in a 2x2 matrix.

## Basic Usage

See `example.ipynb` for a more detailed overview and additional examples and `analysis.pdf` for replication results. 

Users can provide a default name of a game to simulate or specify a custom payoff matrix. Currently, only `"prisoners_dilemma"` and `"matching_pennies"` are supported. 
```python
# Pass the game's name or a custom payoff matrix of shape (2, 2, 2)
pennies = QuantumGame("matching_pennies", gamma = 0, thetaA = 0, thetaB = 0, phiA = 0, phiB = 0)
pennies.run()
```

Nash equilibria can be identified visually by studying the payoff surfaces for each player.
```python
# Full entanglement
plot_full_payoff_surface(payoff_matrix = "prisoners_dilemma", gamma = np.pi / 2, player = 0)
```
