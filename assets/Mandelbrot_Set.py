import numpy as np
import matplotlib.pyplot as plt


def mandelbrot_set_optimized(xmin, xmax, ymin, ymax, width, height, max_iter):
    # Crea la griglia dei punti complessi
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    # Inizializza Z e la matrice dei conteggi
    Z = np.zeros(C.shape, dtype=complex)
    M = np.full(C.shape, max_iter, dtype=int)

    # Crea una maschera per i punti ancora in calcolo
    mask = np.full(C.shape, True, dtype=bool)

    # Itera finché ci sono punti che non hanno divergito
    for i in range(max_iter):
        # Calcola la nuova Z solo per i punti in cui ancora non sono divergenti
        Z[mask] = Z[mask] ** 2 + C[mask]

        # Trova i punti che hanno appena superato il limite (divergenti)
        diverged = np.abs(Z) > 2
        newly_diverged = diverged & mask

        # Registra il numero di iterazioni per i punti che hanno divergito
        M[newly_diverged] = i

        # Aggiorna la maschera, escludendo i punti già divergenti
        mask[newly_diverged] = False

        # Se tutti i punti sono divergenti, interrompe il ciclo
        if not mask.any():
            break
    return M


# Coordinate e parametri per lo zoom ottimizzato
xmin, xmax = -0.74365, -0.74355
ymin, ymax = 0.13180, 0.13190
width, height = 1080, 1920
max_iter = 500

# Calcola il set di Mandelbrot con il metodo ottimizzato
M_instagram = mandelbrot_set_optimized(xmin, xmax, ymin, ymax, width, height, max_iter)

# Visualizza l'immagine in formato verticale (storia Instagram)
plt.figure(figsize=(9, 16))
<<<<<<<<<<<<<<  ✨ Codeium Command ⭐ >>>>>>>>>>>>>>>>
import numpy as np

def generate_brownian_motion(T, N, dt):
    """
    Generate Brownian motion (Wiener process).

    Parameters:
        T (float): Total time.
        N (int): Number of steps.
        dt (float): Time increment.

    Returns:
        np.ndarray: Brownian motion path.
    """
    dW = np.sqrt(dt) * np.random.randn(N)
    W = np.cumsum(dW)
    return np.insert(W, 0, 0)


def generate_geometric_brownian_motion(S0, mu, sigma, T, N):
    """
    Generate Geometric Brownian Motion.

    Parameters:
        S0 (float): Initial stock price.
        mu (float): Drift coefficient.
        sigma (float): Volatility coefficient.
        T (float): Total time.
        N (int): Number of steps.

    Returns:
        np.ndarray: Geometric Brownian Motion path.
    """
    dt = T / N
    t = np.linspace(0, T, N+1)
    W = generate_brownian_motion(T, N, dt)
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    return S0 * np.exp(X)

<<<<<<<  9bfaa587-e3cc-4aaa-9966-19c1988cd550  >>>>>>>
plt.imshow(M_instagram, extent=(xmin, xmax, ymin, ymax), cmap="inferno")
plt.axis("off")  # Rimuove gli assi per un aspetto pulito
plt.show()
