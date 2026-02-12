# Mesa-LLM: Generative Agent-Based Modeling with Large Language Models Empowered Agents

| | |
| --- | --- |
| CI/CD | [![GitHub CI](https://github.com/mesa/mesa-llm/workflows/build/badge.svg)](https://github.com/mesa/mesa-llm/actions) [![Read the Docs](https://readthedocs.org/projects/mesa-llm/badge/?version=stable)](https://mesa-llm.readthedocs.io/stable) [![Codecov](https://codecov.io/gh/projectmesa/mesa-llm/branch/main/graph/badge.svg)](https://codecov.io/gh/projectmesa/mesa-llm) |
| Package | [![PyPI](https://img.shields.io/pypi/v/mesa-llm.svg)](https://pypi.org/project/mesa-llm) [![PyPI - License](https://img.shields.io/pypi/l/mesa-llm)](https://pypi.org/project/mesa-llm/) [![PyPI - Downloads](https://img.shields.io/pypi/dw/mesa-llm)](https://pypistats.org/packages/mesa-llm) |
| Meta | [![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch) |
| Chat | [![chat](https://img.shields.io/matrix/mesa-llm:matrix.org?label=chat&logo=Matrix)](https://matrix.to/#/#mesa-llm:matrix.org) |

Mesa-LLM integrates large language models (LLMs) as decision-making agents into the Mesa agent-based modeling (ABM) framework. It enables sophisticated, language-driven agent behaviors, allowing researchers to model scenarios involving communication, negotiation, and decision-making influenced by natural language.

> **⚠️ WARNING ⚠️**
> This repository is currently under active development. The API and functionality may change significantly. Please stay tuned for our first release coming soon!

## Using Mesa-LLM

To install Mesa-LLM, run:
```bash
pip install -U mesa-llm
```

Mesa-LLM pre-releases can be installed with:
```bash
pip install -U --pre mesa-llm
```

You can also use `pip` to install the GitHub version:
```bash
pip install -U -e git+https://github.com/mesa/mesa-llm.git#egg=mesa-llm
```

Or any other (development) branch on this repo or your own fork:
``` bash
pip install -U -e git+https://github.com/YOUR_FORK/mesa-llm@YOUR_BRANCH#egg=mesa-llm
```

For more help on using Mesa-LLM, check out the following resources:

- [Introductory Tutorial](http://mesa-llm.readthedocs.io/stable/tutorials/intro_tutorial.html)
- [Docs](http://mesa-llm.readthedocs.io/stable/)
- [Mesa-LLM Discussions](https://github.com/mesa/mesa-llm/discussions)
- [PyPI](https://pypi.org/project/mesa-llm/)

## Contributing to Mesa-LLM

Want to join the team or just curious about what is happening with Mesa & Mesa-LLM? You can...

  * Join our [Matrix chat room](https://matrix.to/#/#mesa-llm:matrix.org) in which questions, issues, and ideas can be (informally) discussed.
  * Come to a monthly dev session (you can find dev session times, agendas and notes at [Mesa discussions](https://github.com/mesa/mesa/discussions).
  * Just check out the code at [GitHub](https://github.com/mesa/mesa-llm/).

If you run into an issue, please file a [ticket](https://github.com/mesa/mesa-llm/issues) for us to discuss. If possible, follow up with a pull request.

If you would like to add a feature, please reach out via [ticket](https://github.com/mesa/mesa-llm/issues) or join a dev session (see [Mesa discussions](https://github.com/mesa/mesa/discussions)).

A feature is most likely to be added if you build it!

Don't forget to check out the [Contributors guide](https://github.com/mesa/mesa-llm/blob/main/CONTRIBUTING.md).



```{toctree}
---
maxdepth: 2
hidden: true
---
Introduction <self>
Examples <examples>
API Documentation <apis/api_main>
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`