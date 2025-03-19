

### Install uv


Windows 

```
powershell -c "irm https://astral.sh/uv/install.ps1 | more"
```

macOS and Linux

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or 

```
wget -qO- https://astral.sh/uv/install.sh | sh
```

### Install Python 

```
uv python install 3.10
```

or any other version that you might like

installing packages is as easy as:

```
uv add numpy
```

and so is removing 

```
uv remove numpy
```


### Running Python 

```
uv run python test.py
```