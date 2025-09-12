
# Configuration for TrOCR Table Extraction Project


# Dataset config for Hugging Face Datasets
DATASET_NAME = "ibm/ibm-pubtabnet"
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
TEST_SPLIT = "test"

# Model and training hyperparameters
MODEL_NAME = "microsoft/trocr-base-printed"
VISION_ENCODER = "microsoft/deit-base-distilled-patch16-384"
IMAGE_SIZE = (384, 384)
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
EPOCHS = 20
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
NUM_BEAMS = 4
EARLY_STOPPING = True
NO_REPEAT_NGRAM_SIZE = 3
