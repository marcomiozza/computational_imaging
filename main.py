import json
from src.evaluation import evaluate_batch, save_results_to_csv

def main():
    with open("config_test2.json", "r") as f:
        config = json.load(f)

    results = evaluate_batch(config)
    save_results_to_csv(results, "results/tv_metrics_summary.csv")

if __name__ == "__main__":
    main()
