# identity_recognition
Face verification system that can identify the set user using a binary classification problem.


se propone un desarrollo mediante notebooks en donde se recopile la información y se estructure apropiadamente y se entrene al modelo de forma optima, luego se construirá una simple API mediante fastApi exponiendo un endpoint midiendo desempeño y registrando cada interacción.

### Requerimientos

uv
python3.10+



### install dependencies

dev
```
uv sync --group dev
```

```
uv add [dep] --group dev
```

prod
```
uv sync --group default
```

---
### run

```
uv run uvicorn app.main:app --reload --port 33001
```


## Training Process

uv run .\scripts\crop_faces.py
uv run .\scripts\embeddings.py
uv run .\scripts\train.py
uv run .\scripts\evaluate.py