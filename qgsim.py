import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info.operators import Operator
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator

def plot_full_payoff_surface(payoff_matrix, gamma = 0, player = 0):
    '''
    Simulates games for the case when both players are allowed to play the full range of strategies and plots the resulting
    payoff surface.

    Parameters:
    ----------
    payoff_matrix: str or ndarray(shape = (2, 2, 2))
            If string, name of a default game, one of ["prisoners_dilemma", "matching_pennies"].

            If array, a payoff matrix of shape (num_players, num_strat_A, num_strat_B).
            Currently, only the shape (2, 2, 2) is supported.

    gamma: float
        Entanglement strength. Must be in range [min = 0, max = pi/2].

    player: int
        Index of Player for whom to plot the payoff surface.
    '''
    # Strategy parametrization by t
    tsA = np.linspace(-1, 1, 100)
    tsB = np.linspace(-1, 1, 100)

    # Run the games
    payoffs = []
    for tA in tsA:
        for tB in tsB:
            if (tA > 0) & (tB > 0):
                game = QuantumGame(payoff_matrix = payoff_matrix, gamma = gamma, thetaA = tA * np.pi, thetaB = tB * np.pi, phiA = 0, phiB = 0)
            elif (tA > 0) & (tB < 0):
                game = QuantumGame(payoff_matrix = payoff_matrix, gamma = gamma, thetaA = tA * np.pi, thetaB = 0, phiA = 0, phiB = -tB * np.pi / 2)
            elif (tA < 0) & (tB > 0):
                game = QuantumGame(payoff_matrix = payoff_matrix, gamma = gamma, thetaA = 0, thetaB = tB * np.pi, phiA = -tA * np.pi / 2, phiB = 0)
            elif (tA < 0) & (tB < 0):
                game = QuantumGame(payoff_matrix = payoff_matrix, gamma = gamma, thetaA = 0, thetaB = 0, phiA = -tA * np.pi / 2, phiB = -tB * np.pi / 2)
            _, _, pA, pB = game.run()
            payoffs.append((pA, pB)[player])

    # Plot the surface
    xx, yy = np.meshgrid(tsA, tsB)
    p = np.array(payoffs).reshape((100, 100))

    fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
    surf = ax.plot_surface(xx, yy, p, cmap = cm.coolwarm, antialiased = False)

    # Due to cartesian indexing, reverse the labeling
    ax.set_xlabel("t_B")
    ax.set_ylabel("t_A")

    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    plt.show()

class QuantumGame():
    '''
    Simulates a quantum game as specified by Eisert et al. (1999) and Du et al. (2002) using Qiskit.
    '''

    def __init__(self, payoff_matrix, gamma, thetaA, phiA, thetaB, phiB):
        '''
        Parameters:
        ----------
        payoff_matrix: str or ndarray(shape = (2, 2, 2))
            If string, name of a default game, one of ["prisoners_dilemma", "matching_pennies"].

            If array, a payoff matrix of shape (num_players, num_strat_A, num_strat_B).
            Currently, only the shape (2, 2, 2) is supported.

        gamma: float
            Entanglement strength. Must be in range [min = 0, max = pi/2].

        thetaA: float
            Angle which specifies the real part of Player A's strategies. Must be in range [min = 0, max = pi].

        phiA: float
            Angle which specifies the imaginary part of Player A's strategies. Must be in range [min = 0, max = pi/2].

        thetaB: float
            Angle which specifies the real part of Player B's strategies. Must be in range [min = 0, max = pi].

        phiB: float
            Angle which specifies the imaginary part of Player B's strategies. Must be in range [min = 0, max = pi/2].

        '''
        # Checks
        self.payoff_matrix = None
        self.check_parameters(payoff_matrix, gamma, thetaA, phiA, thetaB, phiB)

        # Define the circuit with 2 qubits
        self.circuit = QuantumCircuit(2)
        # Add gates
        self.circuit.unitary(Operator(self.J(gamma)), [0, 1], label = "J")
        self.circuit.unitary(Operator(self.U(thetaA, phiA)), [0], label = "U_A")
        self.circuit.unitary(Operator(self.U(thetaB, phiB)), [1], label = "U_B")
        self.circuit.unitary(Operator(np.linalg.pinv(self.J(gamma))), [0, 1], label = "J^d")
        # Measure
        self.circuit.measure_all()

    def run(self, num_rounds = 1000):
        '''
        Run the game with Qiskit's AerSimulator.

        Parameters:
        ----------
        num_rounds: int = 1000
            Number of times to run the game for.

        Returns:
        ----------
        outcomes: list[tuple(int, int)]
            The list of outcomes for the game. The integers represent indexes of strategies.
        
        probabilities: list[float]
            Probabilities for the respecitve outcomes.

        payoffA: float
            Expected payoff of Player A.

        payoffB: float
            Expected payoff of Player B.
        '''
        # Run the simulation
        simulator = AerSimulator()
        circuit_simulator = simulator.run(transpile(self.circuit, simulator), shots = num_rounds)
        statistics = circuit_simulator.result().get_counts()

        # Compute statistics
        payoffA = 0
        payoffB = 0
        outcomes = []
        probabilities = []

        for k in statistics.keys():
            sA = int(k[1])
            sB = int(k[0])
            outcomes.append((sA, sB))
            probabilities.append(statistics[k])
            payoffA += statistics[k] / 1000 * self.payoff_matrix[0][sA][sB]
            payoffB += statistics[k] / 1000 * self.payoff_matrix[1][sA][sB]
    
        return outcomes, probabilities, payoffA, payoffB
        
    def U(self, theta, phi):
        '''
        Strategy matrix parameterized by angles theta and phi.

        Parameters:
        ----------
        theta: float
            Angle which specifies the real part of the strategy.

        phi: float
            Angle which specifies the imaginary part of the strategy.

        Returns:
        ----------
        U: ndarray(shape = (2, 2))
            Strategy matrix.
        '''
        return np.array([[np.exp(1j * phi) * np.cos(theta / 2), np.sin(theta / 2)],
                         [-np.sin(theta / 2), np.exp(-1j * phi) * np.cos(theta / 2)]])
    
    def J(self, gamma):
        '''
        Entanglement operator. Based on both https://www.mdpi.com/1099-4300/23/5/506 and 
        https://arxiv.org/pdf/quant-ph/9806088v2.pdf

        Parameters:
        ----------
        gamma: float
            Entanglement strength.

        Returns:
        ----------
        J: ndarray(shape = (2, 2))
            Entanglement matrix.
        '''
        I = np.eye(2)
        D = self.U(np.pi, 0)
        return np.cos(gamma / 2) * np.kron(I, I) - 1j * np.sin(gamma / 2) * np.kron(D, D)
    
    def check_parameters(self, payoff_matrix, gamma, thetaA, phiA, thetaB, phiB):
        '''
        Check input parameters are correctly specified.
        '''
        if type(payoff_matrix) == str:
            if payoff_matrix == "prisoners_dilemma":
                self.payoff_matrix = np.array([[[3, 0],
                                                [5, 1]],
                                               [[3, 5],
                                                [0, 1]]])
            elif payoff_matrix == "matching_pennies":
                self.payoff_matrix = np.array([[[1, -1],
                                                [-1, 1]],
                                               [[-1, 1],
                                                [1, -1]]])
            else:
                RuntimeError("Unsupported game. The game name should be one of ['prisoners_dilemma', 'matching_pennies'].")
        elif type(payoff_matrix) == np.ndarray:
            if payoff_matrix.shape == (2, 2, 2):
                self.payoff_matrix = payoff_matrix
            else:
                RuntimeError("Unsupported game. Currently only games of shape (2, 2, 2) are supported.")

        if (gamma < 0) | (gamma > np.pi / 2):
            RuntimeError("Gamma should be in range [0, pi/2].")
        if (thetaA < 0) | (thetaA > np.pi):
            RuntimeError("Theta should be in range [0, pi].")
        if (phiA < 0) | (phiA > np.pi / 2):
            RuntimeError("Phi should be in range [0, pi/2].")
        if (thetaB < 0) | (thetaB > np.pi):
            RuntimeError("Theta should be in range [0, pi].")
        if (phiB < 0) | (phiB > np.pi / 2):
            RuntimeError("Phi should be in range [0, pi/2].")