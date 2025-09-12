import matplotlib.pyplot as plt
from transformers import TrOCRProcessor
from dataset import PubTabNetDataset
import config

def test_pubtabnet_loader():
    processor = TrOCRProcessor.from_pretrained(config.MODEL_NAME)
    dataset = PubTabNetDataset(split=config.TRAIN_SPLIT, processor=processor, max_samples=2)
    for i, batch in enumerate(dataset):
        print(f"Sample {i}:")
        print("Input IDs:", batch['input_ids'].shape)
        print("Pixel Values:", batch['pixel_values'].shape)
        # Visualize image
        img = batch['pixel_values'].numpy().transpose(1, 2, 0)
        plt.imshow(img)
        plt.title(f"Sample {i}")
        plt.show()
        if i >= 1:
            break

if __name__ == "__main__":
    test_pubtabnet_loader()
