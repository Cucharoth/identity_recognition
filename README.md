# identity_recognition
Face verification system that can identify the set user using a binary classification problem.


se propone un desarrollo mediante notebooks en donde se recopile la información y se estructure apropiadamente y se entrene al modelo de forma optima, luego se construirá una simple API mediante fastApi exponiendo un endpoint midiendo desempeño y registrando cada interacción.



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