---
title: Space Invaders
logo: /assets/space-invaders/favicon.svg
description: Variant of the famous game written in Processing language.
buttons:
    - name: Downloads
      link: https://github.com/sylvanld-enssat/space-invader/releases
    - name: View on GitHub
      link: https://github.com/sylvanld-enssat/space-invader
---

## Table of contents

<style>
.toc ul{
    padding-left: 1.5rem;
}
    </style>

[TOC]

## Quickstart

### Install dependencies

```bash
pip install -r flask
```

### Use the framework

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "hello world"

app.run(port=8080)
```

## Supported platforms

=== "Windows"
    1. [Download](https://github.com/ENSSAT/space-invader/releases/) the latest version for windows.
    2. Extract content of the zip where you want to install it.
    3. (optional) Right click on `space_invaders.exe` and create a desktop shortcut.
    4. To play, double click on `space_invaders.exe` or previously created shortcut.

=== "Linux"
    1. [Download](https://github.com/ENSSAT/space-invader/releases/) the latest version for linux.
    2. Extract content of the zip where you want to install it.
    3. To play, run `./space_invaders` in your terminal

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
