import argparse
import yaml
from pathlib import Path
import datetime
import time

from ctf4science.data_module import (
    load_dataset,
    parse_pair_ids,
    get_applicable_plots,
    get_prediction_timesteps,
    get_metadata,
)
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization

from spatialTemporalModel import SpatialTemporalTrainer

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    pair_ids = parse_pair_ids(config["dataset"])

    model_name = f"{config['model']['name']}"
    batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    batch_results = {
        "batch_id": batch_id,
        "model": model_name,
        "dataset": dataset_name,
        "pairs": [],
    }

    viz = Visualization()
    applicable_plots = get_applicable_plots(dataset_name)

    for pair_id in pair_ids:
        print(f"\n=== Running pair {pair_id} ===")

        train_data, init_data = load_dataset(dataset_name, pair_id)
        prediction_timesteps = get_prediction_timesteps(dataset_name, pair_id)
        delta_t = get_metadata(dataset_name)["delta_t"]

        trainer = SpatialTemporalTrainer(
            pair_id=pair_id,
            config=config,
            train_data=train_data,
            init_data=init_data,
            prediction_timesteps=prediction_timesteps,
            delta_t=delta_t,
        )

        # --- TRAINING WITH EPOCH PROGRESS ---
        print(f"Training model for pair {pair_id}...")
        start_time = time.time()
        batch_size = config['model'].get('batch_size', 64)
        lr = config['model'].get('learning_rate', 1e-3)
        epochs = config['model'].get('epochs', 100)

        train_loader = trainer.train_dataset
        trainer.model.train()
        trainer.train()  # Uses the updated trainer with per-epoch printouts

        print(f"Training finished in {time.time() - start_time:.2f}s")

        # --- PREDICTION ---
        print("Running prediction...")
        predictions = trainer.predict()

        # --- EVALUATION ---
        results = evaluate(dataset_name, pair_id, predictions)

        results_directory = save_results(
            dataset_name,
            model_name,
            batch_id,
            pair_id,
            config,
            predictions,
            results,
        )

        batch_results["pairs"].append({
            "pair_id": pair_id, 
            "metrics": results
        })

        # --- PLOTS ---
        for plot_type in applicable_plots:
            try:
                fig = viz.plot_from_batch(
                    dataset_name, pair_id, results_directory, plot_type=plot_type
                )
                viz.save_figure_results(
                    fig, dataset_name, model_name, batch_id, pair_id, plot_type
                )
            except Exception as e:
                print(f"Error generating {plot_type} plot for pair_id {pair_id}: {e}")

    # --- SAVE BATCH RESULTS ---
    with open(results_directory.parent / "batch_results.yaml", "w") as f:
        yaml.dump(batch_results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)