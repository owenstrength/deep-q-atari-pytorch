class CONFIGS:
    def __init__(self):
        # AGENT
        self.BUFFER_SIZE = 30000 # Reduced from 1M to 30k

        self.GAMMA = 0.99

        self.EPSILON_START = 1.0
        self.EPSILON_END = 0.05
        self.EPSILON_DECAY_STEPS = 1000000  # 1M frames for linear decay
        
        self.TARGET_FREQ_UPDATE = 10000  # Changed from 10K to 1K

        # TRAIN_AGENT
        self.NUM_FRAMES = 5000000 # Reduced from 50M to 5M

config = CONFIGS()