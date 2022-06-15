import json
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import joblib
import mlflow
import optuna
import pandas as pd
import typer
from optuna.integration.mlflow import MLflowCallback
from optuna.samplers import TPESampler

from config import config
from config.config import logger
from tagolym import predict, train, utils

app = typer.Typer()


@app.command()
def train_model(args_fp: Path, experiment_name: str, run_name: str) -> None:
    """Load and split the dataset into training, validation, and testing.
    Then, train the model on training split and predict on validation and testing split.
    Log artifacts, metrics, and parameters to MLflow.
    Save MLflow run ID and metrics to config for future use.

    Args:
        args_fp (Path): Filepath of arguments to be used during training
        experiment_name (str): Name of the experiment to be activated
        run_name (str): Name of MLflow run
    """

    # load labeled data
    projects_fp = Path(config.DATA_DIR, "math_problems.json")
    df = pd.read_json(projects_fp)

    # train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)

    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")

        # fit, predict, and evaluate
        artifacts = train.train(args=args, df=df)

        # log key metrics
        for split in ["train", "val", "test"]:
            metrics = artifacts[f"{split}_metrics"]["overall"]
            for score in ["precision", "recall", "f1"]:
                mlflow.log_metrics({f"{split}_{score}": metrics[f"{score}"]})

        # log artifacts
        with tempfile.TemporaryDirectory() as dp:
            joblib.dump(artifacts["label_encoder"], Path(dp, "label_encoder.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(
                artifacts["train_metrics"], Path(dp, "train_metrics.json"), cls=utils.NumpyEncoder
            )
            utils.save_dict(
                artifacts["val_metrics"], Path(dp, "val_metrics.json"), cls=utils.NumpyEncoder
            )
            utils.save_dict(
                artifacts["test_metrics"], Path(dp, "test_metrics.json"), cls=utils.NumpyEncoder
            )
            utils.save_dict({**args.__dict__}, Path(dp, "args.json"), cls=utils.NumpyEncoder)
            mlflow.log_artifacts(dp)

        # log parameters
        mlflow.log_params(vars(artifacts["args"]))

    # save to config
    open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
    utils.save_dict(
        artifacts["train_metrics"],
        Path(config.CONFIG_DIR, "train_metrics.json"),
        cls=utils.NumpyEncoder,
    )
    utils.save_dict(
        artifacts["val_metrics"],
        Path(config.CONFIG_DIR, "val_metrics.json"),
        cls=utils.NumpyEncoder,
    )
    utils.save_dict(
        artifacts["test_metrics"],
        Path(config.CONFIG_DIR, "test_metrics.json"),
        cls=utils.NumpyEncoder,
    )


@app.command()
def optimize(args_fp: Path, study_name: str, num_trials: int) -> None:
    """Two-step hyperparameter optimization including those in preprocessing.
    The first step is for all hyperparameters other than those used for incremental learning.
    The second step is for all hyperparameters for incremental learning.

    Args:
        args_fp (Path): Filepath of arguments to be used during training
        study_name (str): Name of optuna study
        num_trials (int): Number of trials for each study
    """

    # load labeled data
    projects_fp = Path(config.DATA_DIR, "math_problems.json")
    df = pd.read_json(projects_fp)
    args = Namespace(**utils.load_dict(filepath=args_fp))

    # optimize some args
    study = optuna.create_study(
        sampler=TPESampler(seed=config.SEED), study_name=study_name, direction="maximize"
    )
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(
        lambda trial: train.objective(args, df, trial, experiment=0),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # update args
    args = {**args.__dict__, **study.best_params}
    args = Namespace(**args)

    # optimize other args
    study.optimize(
        lambda trial: train.objective(args, df, trial, experiment=1),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # best trial
    args = {**args.__dict__, **study.best_params}
    utils.save_dict(args, Path(config.CONFIG_DIR, "args_opt.json"), cls=utils.NumpyEncoder)
    logger.info(f"Best value (f1): {study.best_value}")
    logger.info(f"Best hyperparameters: {json.dumps(args, indent=2)}")


def load_artifacts(run_id: str) -> Dict:
    """Load artifacts from MLflow run.

    Args:
        run_id (str): ID of MLflow run

    Returns:
        Dict: Artifacts including arguments, label encoder, model, and metrics
    """

    # load objects from run
    client = mlflow.tracking.MlflowClient()
    with tempfile.TemporaryDirectory() as dp:
        client.download_artifacts(run_id=run_id, path="", dst_path=dp)
        mlb = joblib.load(Path(dp, "label_encoder.pkl"))
        model = joblib.load(Path(dp, "model.pkl"))
        train_metrics = utils.load_dict(filepath=Path(dp, "train_metrics.json"))
        val_metrics = utils.load_dict(filepath=Path(dp, "val_metrics.json"))
        test_metrics = utils.load_dict(filepath=Path(dp, "test_metrics.json"))
        args = Namespace(**utils.load_dict(filepath=Path(dp, "args.json")))

    return {
        "args": args,
        "label_encoder": mlb,
        "model": model,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


@app.command()
def predict_tag(text: List, run_id: str = None) -> None:
    """Predict tags from a text or list of texts using the model from a specific MLflow run ID.

    Args:
        text (List): Math problem(s) in LaTeX format
        run_id (str, optional): ID of MLflow run, by default the latest run from `train_model`.
    """

    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()

    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=text, artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    app()
