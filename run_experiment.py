# run_experiment.py
from src.complete_implementation import CreditCardFraudDetectionPipeline

if __name__ == "__main__":
    pipeline = CreditCardFraudDetectionPipeline()

    # -----------------------------
    # 1) Test pipeline using synthetic data
    # -----------------------------
    print("=== Running pipeline with synthetic data ===")
    pipeline.run_complete_pipeline(filepath=None, use_novel_sampling=True)

    # -----------------------------
    # 2) Test pipeline using real Kaggle dataset
    # -----------------------------
    try:
        print("\n=== Running pipeline with real data ===")
        pipeline.run_complete_pipeline(
            filepath="data/raw/creditcard.csv", use_novel_sampling=True)
    except FileNotFoundError:
        print("Kaggle dataset not found at 'data/raw/creditcard.csv'. Skipping real data test.")
