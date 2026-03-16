from ..models.train import main as train_model


def run():
    print("Starting training pipeline...")

    train_model()

    print("Training finished.")


if __name__ == "__main__":
    run()