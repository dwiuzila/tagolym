# tagolym

Tag Olympiad math problems into algebra, combinatorics, geometry, or number theory.

## Organization
```bash
config/
├── config.py        - configuration setup
├── args.json        - training parameters
└── args_opt.json    - optimized training parameters
tagolym/
├── data.py          - data processing components
├── evaluate.py      - evaluation components
├── main.py          - training/optimization pipelines
├── predict.py       - inference components
├── train.py         - training components
└── utils.py         - supplementary utilities
```

## Operations
```python linenums="1"
from pathlib import Path
from config import config
from tagolym import main

# Train model
args_fp = Path(config.CONFIG_DIR, 'args.json')
main.train_model(args_fp, experiment_name='baselines', run_name='sgd')

# Optimize model
main.optimize(args_fp, study_name='optimization', num_trials=20)
args_fp = Path(config.CONFIG_DIR, 'args_opt.json')
main.train_model(args_fp, experiment_name='best', run_name='sgd')

# Inference
text = [
    "Let $n \geqslant 100$ be an integer. Ivan writes the numbers $n, n+1, \ldots, 2 n$ each on different cards. He then shuffles these $n+1$ cards, and divides them into two piles. Prove that at least one of the piles contains two cards such that the sum of their numbers is a perfect square.",
    "Show that the inequality\[\sum_{i=1}^n \sum_{j=1}^n \sqrt{|x_i-x_j|}\leqslant \sum_{i=1}^n \sum_{j=1}^n \sqrt{|x_i+x_j|}\]holds for all real numbers $x_1,\ldots x_n.$",
    "Let $D$ be an interior point of the acute triangle $ABC$ with $AB > AC$ so that $\angle DAB = \angle CAD.$ The point $E$ on the segment $AC$ satisfies $\angle ADE =\angle BCD,$ the point $F$ on the segment $AB$ satisfies $\angle FDA =\angle DBC,$ and the point $X$ on the line $AC$ satisfies $CX = BX.$ Let $O_1$ and $O_2$ be the circumcenters of the triangles $ADC$ and $EXD,$ respectively. Prove that the lines $BC, EF,$ and $O_1O_2$ are concurrent.",
]
run_id = open(Path(config.CONFIG_DIR, 'run_id.txt')).read()
main.predict_tag(text=text, run_id=run_id)
```

## Documentation
```
python -m mkdocs serve -a localhost:8000
```

## Styling
```
black .
flask8 --exit-zero
isort .
```

## Makefile
```bash
make help
```

## API
```bash
# Update RUN_ID inside config/run_id.txt
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir tagolym --reload-dir app  # dev
```
