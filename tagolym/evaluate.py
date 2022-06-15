from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier, SlicingFunction, slicing_function


@slicing_function()
def short_problem(x):
    return len(x["token"].split()) < 5


# keyword-based SFs
def keyword_lookup(x, keywords):
    return all(word in x["token"].split() for word in keywords)


def make_keyword_sf(keywords):
    return SlicingFunction(
        name="keyword_number",
        f=keyword_lookup,
        resources=dict(keywords=keywords),
    )


keyword_number = make_keyword_sf(keywords=["integ", "real"])


def average_performance(y_true, y_pred, average="weighted"):
    """Generate metrics using an averaging method."""

    metrics = precision_recall_fscore_support(y_true, y_pred, average=average)
    return {
        "precision": metrics[0],
        "recall": metrics[1],
        "f1": metrics[2],
        "num_samples": len(y_true),
    }


def get_slice_metrics(y_true, y_pred, slices):
    """Generate metrics for slices of data."""

    slice_metrics = {}
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)
        if sum(mask):
            slice_metrics[slice_name] = average_performance(y_true[mask], y_pred[mask], "micro")

    return slice_metrics


def get_metrics(y_true, y_pred, classes, df=None):
    """Performance metrics using ground truths and predictions."""

    # performance
    performance = {"overall": {}, "class": {}}

    # overall performance
    performance["overall"] = average_performance(y_true, y_pred, "weighted")

    # per-class performance
    metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i in range(len(classes)):
        performance["class"][classes[i]] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "num_samples": metrics[3][i],
        }

    # per-slice performance
    if df is not None:  # pragma: no cover, slicing template
        slices = PandasSFApplier([short_problem, keyword_number]).apply(df)
        performance["slices"] = get_slice_metrics(y_true, y_pred, slices)

    return performance
