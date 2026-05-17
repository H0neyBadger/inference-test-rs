
Create venv kernel for diy-alexa (python3.12)
```
uv venv --python 3.12 ./.venv/diy-alexa
uv pip install --python ./.venv/diy-alexa ipykernel -r diy-alexa/model/requirements.txt
./.venv/diy-alexa/bin/ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv/diy-alexa --name=tensorflow
```

```
uv run --with jupyter jupyter lab --notebook-dir=./diy-alexa/model/
```
