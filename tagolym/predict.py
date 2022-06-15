from tagolym import data


def predict(texts, artifacts):
    """Predict tags for given texts."""

    args = artifacts["args"]
    mlb = artifacts["label_encoder"]
    model = artifacts["model"]
    x = [data.preprocess_problem(txt, args.nocommand, args.stem) for txt in texts]
    y = model.predict(x)
    tags = mlb.inverse_transform(y)
    predictions = [
        {
            "input_text": texts[i],
            "predicted_tags": tags[i],
        }
        for i in range(len(texts))
    ]
    return predictions
