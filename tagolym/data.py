import regex as re
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from config import config

porter = PorterStemmer()


def extract_features(equation_pattern, p):
    """Extract LaTeX syntax."""

    pattern = re.findall(equation_pattern, p)
    ptn_len = [len(ptn) for ptn in pattern]
    pattern = ["".join(ptn) for ptn in pattern]
    syntax = [" ".join(re.findall(r"\\(?:[^a-zA-Z]|[a-zA-Z]+[*=']?)", ptn)) for ptn in pattern]
    split = ["" if s is None else s for s in re.split(equation_pattern, p)]

    i = 0
    for ptn, length, cmd in zip(pattern, ptn_len, syntax):
        while "".join(split[i : i + length]) != ptn:
            i += 1
        split[i : i + length] = [cmd]

    return " ".join(split)


def preprocess_problem(x, nocommand=False, stem=False):
    """Preprocess a problem."""

    x = x.lower()  # lowercase all
    x = re.sub(r"http\S+", "", x)  # remove URLs
    x = x.replace("$$$", "$$ $")  # separate triple dollars
    x = x.replace("\n", " ")  # remove new lines
    x = extract_features(config.EQUATION_PATTERN, x)  # extract latex
    x = re.sub(config.ASYMPTOTE_PATTERN, "", x)  # remove asymptote

    # remove stopwords
    x = x.replace("\\", " \\")
    x = " ".join(word for word in x.split() if word not in config.STOPWORDS)

    x = re.sub(r"([-;.,!?<=>])", r" \1 ", x)  # separate filters from words
    x = re.sub("[^A-Za-z0-9]+", " ", x)  # remove non-alphanumeric chars

    # clean command words
    if nocommand:
        x = " ".join(word for word in x.split() if word not in config.COMMANDS)

    # stemming
    if stem:
        x = " ".join(porter.stem(word) for word in x.split())
    
    x = x.strip()

    return x


def preprocess(df, nocommand, stem):
    """Preprocess whole data."""

    df["token"] = df["problem"].apply(preprocess_problem, args=(nocommand, stem))
    df = df[df["token"] != ""].reset_index(drop=True)
    return df


def binarize(tags):
    """Binarize tags."""

    mlb = MultiLabelBinarizer()
    tags = mlb.fit_transform(tags)
    return tags, mlb


def split_data(X, y, train_size=0.7, random_state=None):
    """Split predictors and targets into training, validation, and testing."""

    X_train, X_, y_train, y_ = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=0.5, stratify=y_, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
