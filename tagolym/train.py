from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from config import config
from tagolym import data, evaluate


def train(args, df):
    """Train model on data."""

    # setup
    df = data.preprocess(df, args.nocommand, args.stem)
    tags, mlb = data.binarize(df["tags"])
    classes = mlb.classes_
    X_train, X_val, X_test, y_train, y_val, y_test = data.split_data(
        df[["token"]], tags, random_state=config.SEED
    )

    # model
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, args.ngram_max_range))),
            (
                "stack",
                MultiOutputClassifier(
                    StackingClassifier(
                        [
                            (
                                "sgd",
                                SGDClassifier(
                                    penalty="elasticnet",
                                    random_state=config.SEED,
                                    early_stopping=True,
                                    class_weight="balanced",
                                    loss=args.loss,
                                    alpha=args.alpha,
                                    l1_ratio=args.l1_ratio,
                                    learning_rate=args.learning_rate,
                                    eta0=args.eta0,
                                    power_t=args.power_t,
                                ),
                            )
                        ]
                    ),
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # fit, predict, and evaluate
    model.fit(X_train["token"], y_train)
    metrics = []
    for X, y in zip((X_train, X_val, X_test), (y_train, y_val, y_test)):
        y_pred = model.predict(X["token"])
        metrics.append(evaluate.get_metrics(y_true=y, y_pred=y_pred, classes=classes, df=X))

    return {
        "args": args,
        "label_encoder": mlb,
        "model": model,
        "train_metrics": metrics[0],
        "val_metrics": metrics[1],
        "test_metrics": metrics[2],
    }


def objective(args, df, trial, experiment=0):
    """Objective function for optimization trials."""

    # parameters to tune
    if experiment == 0:
        args.nocommand = trial.suggest_categorical("nocommand", [True, False])
        args.stem = trial.suggest_categorical("stem", [True, False])
        args.ngram_max_range = trial.suggest_int("ngram_max_range", 2, 4)
        args.loss = trial.suggest_categorical("loss", ["hinge", "log", "modified_huber"])
        args.l1_ratio = trial.suggest_uniform("l1_ratio", 0.0, 1.0)
        args.alpha = trial.suggest_loguniform("alpha", 1e-5, 1e-2)
    elif experiment == 1:
        args.learning_rate = trial.suggest_categorical(
            "learning_rate", ["constant", "optimal", "invscaling", "adaptive"]
        )
        if args.learning_rate != "optimal":
            args.eta0 = trial.suggest_loguniform("eta0", 1e-2, 1e-0)
        if args.learning_rate == "invscaling":
            args.power_t = trial.suggest_uniform("power_t", 0.1, 0.5)
    else:
        raise ValueError("Experiment not recognized. Try 0 or 1.")

    # train & evaluate
    artifacts = train(args=args, df=df)

    # set additional attributes
    for split in ["train", "val", "test"]:
        metrics = artifacts[f"{split}_metrics"]["overall"]
        for score in ["precision", "recall", "f1"]:
            trial.set_user_attr(f"{split}_{score}", metrics[f"{score}"])

    return artifacts["val_metrics"]["overall"]["f1"]
