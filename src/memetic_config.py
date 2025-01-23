# ==============================================
#               CONFIGURATION                  
# ==============================================

# Population parameters
POP_SIZE = 50             # Increased population size for greater diversity
MAX_DEPTH = 5                # Reduced maximum depth to mitigate bloat
N_GENERATIONS = 20           # Increased number of generations
TOURNAMENT_SIZE = 5          # Higher selective pressure
MUTATION_RATE = 0.6          # Reduced to balance the effect of crossover
CROSSOVER_RATE = 0.3         # Increased to encourage exploration
ELITISM = 5                  # Increased number of elite individuals

# Bloat control parameter
BLOAT_PENALTY = 0.2          # Increased penalty to favor smaller trees

# Partial Reinitialization parameters
PARTIAL_REINIT_EVERY = 100   # Increased reinitialization frequency (obsolete)
PARTIAL_REINIT_RATIO = 0.25  # Increased reinitialization proportion (obsolete)
DIVERSITY_THRESHOLD = 0.1      # Threshold to trigger reinitialization
REINIT_FRACTION = 0.7        # Fraction of the population to reinitialize

# New options
ENABLE_LOCAL_SEARCH = True   # Enable/disable local search
ADAPTIVE_STRATEGY = True     # Activate adaptive strategies

# Early stopping parameters
MAX_GENERATIONS_NO_IMPROVEMENT = 100  # Maximum generations without improvement
FITNESS_THRESHOLD = 1                 # Minimum threshold for best fitness

# Seed for reproducibility
SEED = 42
