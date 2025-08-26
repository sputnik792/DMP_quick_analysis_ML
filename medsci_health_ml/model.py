import argparse
from trainer import get_datasets, load_dataset, train_models

def main():
    parser = argparse.ArgumentParser(description="Train ML models on a dataset")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset file inside datasets/ folder (e.g., diabetes.csv)")
    parser.add_argument("--features", nargs="+", required=True,
                        help="List of feature columns to use")
    parser.add_argument("--target", type=str, required=True,
                        help="Target column to predict")

    args = parser.parse_args()

    # Train models
    print(f"Training models on {args.dataset} with target={args.target} and features={args.features}")
    train_models(args.dataset, args.features, args.target)

if __name__ == "__main__":
    main()
