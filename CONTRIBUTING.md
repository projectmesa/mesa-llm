Contributing
============

As an open source project, Mesa-LLM welcomes contributions of many forms, and from beginners to experts. If you are curious or just want to see what is happening, we post our development session agendas and development session notes on [Mesa-LLM discussions].

In no particular order, examples include:

- Code patches
- Bug reports and patch reviews
- New features
- Documentation improvements
- Tutorials

[Mesa-LLM discussions]: https://github.com/mesa/mesa-llm/discussions
[issue]: https://github.com/mesa/mesa-llm/issues

**To submit a contribution**

- Create a ticket for the item that you are working on.
- Fork the Mesa-LLM repository.
- [Clone your repository] from GitHub to your machine.
- Create a new branch in your fork: `git checkout -b BRANCH_NAME`
- Run `git config pull.rebase true`. This prevents messy merge commits when updating your branch on top of Mesa-LLM main branch.
- Install an editable version with developer requirements locally: `pip install -e ".[dev]"`
- Edit the code. Save.
- Git add the new files and files with changes: `git add FILE_NAME`
- Git commit your changes with a meaningful message: `git commit -m "Fix issue X"`
- If implementing a new feature, include some documentation in docs folder.
- Make sure that your submission passes the [GH Actions build]. See "Testing and Standards below" to be able to run these locally.
- Make sure that your code is formatted according to the [black] standard (you can do it via [pre-commit]).
- Push your changes to your fork on Github: `git push origin NAME_OF_BRANCH`.
- [Create a pull request].
- Describe the change w/ ticket number(s) that the code fixes.

[Clone your repository]: https://help.github.com/articles/cloning-a-repository/
[GH Actions build]: https://github.com/mesa/mesa-llm/actions/workflows/ci.yml
[Create a pull request]: https://help.github.com/articles/creating-a-pull-request/
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

Testing and Code Standards
--------------------------

[![](https://codecov.io/gh/projectmesa/mesa-llm/branch/main/graph/badge.svg)](https://codecov.io/gh/projectmesa/mesa-llm) [![](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

As part of our contribution process, we practice continuous integration and use GH Actions to help enforce best practices.

If you're changing previous Mesa-LLM features, please make sure of the following:

- Your changes pass the current tests.
- Your changes pass our style standards.
- Your changes don't break the models or your changes include updated models.
- Additional features or rewrites of current features are accompanied by tests.
- New features are demonstrated in a model, so folks can understand more easily.

We test by implementing simple models and through traditional unit tests in the tests/ folder. The following only covers unit tests coverage. Ensure that your test coverage has not gone down. If it has and you need help, we will offer advice on how to structure tests for the contribution.

```bash
pytest --cov=mesa_llm tests/
```

With respect to code standards, we follow [PEP8] and the [Google Style Guide]. We recommend to use [black] as an automated code formatter. You can automatically format your code using [pre-commit], which will prevent `git commit` of unstyled code and will automatically apply black style so you can immediately re-run `git commit`. To set up pre-commit run the following commands:

```bash
pre-commit install
```

You should no longer have to worry about code formatting. If still in doubt you may run the following command. If the command generates errors, fix all errors that are returned.

```bash
pre-commit run --all-files
```

[PEP8]: https://www.python.org/dev/peps/pep-0008
[Google Style Guide]: https://google.github.io/styleguide/pyguide.html
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

Licensing
---------

The license of this project is located in [LICENSE]. By submitting a contribution to this project, you are agreeing that your contribution will be released under the terms of this license.

[LICENSE]: https://github.com/mesa/mesa-llm/blob/main/LICENSE

Special Thanks
--------------

A special thanks to the following projects who offered inspiration for this contributing file.

- [Mesa](https://github.com/mesa/mesa/blob/main/CONTRIBUTING.md)
- [Django](https://github.com/django/django/blob/master/CONTRIBUTING.rst)
- [18F's FOIA](https://github.com/18F/foia-hub/blob/master/CONTRIBUTING.md)
- [18F's Midas](https://github.com/18F/midas/blob/devel/CONTRIBUTING.md)