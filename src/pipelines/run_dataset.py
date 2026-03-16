from ..data.raw_to_interim import main as raw_to_interim
from ..data.interim_to_processed import main as interim_to_processed


def run():
    print("Building dataset pipeline...")

    raw_to_interim()
    interim_to_processed()

    print("Dataset pipeline finished.")


if __name__ == "__main__":
    run()