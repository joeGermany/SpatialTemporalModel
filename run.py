import argparse
import yaml
from pathlib import Path
import datetime

from ctf4science.data_module import (
    load_dataset,
    parse_pair_ids,
    get_applicable_plots,
    get_prediction_timesteps,
    get_metadata,
)
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization

from spatial_temporal_trainer import SpatialTemporalTrainer


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
            config=config,
            train_data=train_data,
            init_data=init_data,
            prediction_timesteps=prediction_timesteps,
            delta_t=delta_t,
        )

        trainer.train()

        predictions = trainer.predict()

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

    # Save aggregated batch results
    with open(results_directory.parent / "batch_results.yaml", "w") as f:
        yaml.dump(batch_results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)