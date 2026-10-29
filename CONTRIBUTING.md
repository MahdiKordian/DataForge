# Contributing to DataForge

Thank you for wanting to improve DataForge. Pull requests are welcome.

## What you need to know first

- DataForge is released under the [MIT License](LICENSE). By sending a pull request, you agree that your contribution is released under the same license.
- You may fork the repository and use your fork freely, as the license allows.
- The author decides what is merged, and may decline a change without giving a reason.

## Before you start

1. For anything bigger than a small fix, [open an issue](https://github.com/MahdiKordian/DataForge/issues/new/choose) first and describe the need. This saves you from writing something that does not fit.
2. Report security problems privately, not as an issue or pull request. See [docs/SECURITY.md](docs/SECURITY.md).

## How to send a change

1. Fork the repository on GitHub, and make a branch.
2. Make the change. Keep it small and focused: one fix or one feature per pull request.
3. Add or update tests in `tests/`, and run them:
   ```
   pip install -r requirements.txt pytest
   python -m pytest -q
   ```
4. Keep the style of the surrounding code. New Python files start with the copyright line used in the other files.
5. Write a clear title and describe what changed and why. Fill in the pull request template.
6. Do not include personal or confidential data. Code or data that you did not write must have a license that allows this use, and must be listed in `NOTICE`.

## Commit messages

Use a short prefix that says the kind of change, for example `fix(prepare): ...`, `feat(lib): ...`, `docs: ...`, `test: ...`.
