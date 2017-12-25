settings = {
    "learning_rate": 1e-6,
    "memory_size": 1000000,                 # 1 Million frames in memory
    "epsilon_start": 0.5,                         # Start of epsilon decent
    "epsilon_end": 0.0,                     # End of epsilon decent
    "epsilon_steps": 100000,                     # Epsilon steps
    "exploration_wins": 0,                  # Number of victories using random moves before starting epsilon phase
    "batch_size": 16,
    "discount_factor":  0.99,
    "grayscale": False,
    "load_latest_checkpoint": False,
}