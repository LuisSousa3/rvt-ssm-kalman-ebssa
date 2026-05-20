# EBSSA RVT project

Use `config.py` to choose what the pipeline does.

```python
pipeline_stages = ("evaluate",)
```

Common options:

```python
pipeline_stages = ("convert", "preprocess", "train", "evaluate")
pipeline_stages = ("evaluate",)
```

Run:

```bash
python main.py
```

Evaluation writes the test metrics plus one MP4, tracking CSV, and tracking PNG
for each test sequence used by the evaluation.

