# Releasing ExoSim 2

Full guide: `docs/source/contributing/releasing.rst`
(rendered at <https://exosim2-public.readthedocs.io/> → Developer guide → Releasing).

## TL;DR

- **Versioning** is automatic (`setuptools-scm`, from git tags). Nothing to bump
  by hand. Every commit on `develop` is `X.Y.(Z+1).devN`.
- **Changelog**: when a piece of work is done, run `nox -s changelog-fragment`
  and describe it. Fragments live in `changelog.d/`.
- **Cut a release** from a clean, up-to-date `develop`:

  ```console
  nox -s release
  ```

  It asks major/minor/patch, folds the changelog fragments in, updates
  `CITATION.cff` / `codemeta.json`, commits, fast-forwards `main`, and pushes the
  `vX.Y.Z` tag.

- The tag triggers `.github/workflows/release.yml`:
  build → TestPyPI → verify install → PyPI → GitHub Release.
  Then `sync-to-public.yml` mirrors `main` to `ExoSim2-public` and copies the
  release there.

## One-time setup

- Add a **PyPI trusted publisher** to the `exosim` project on both
  <https://pypi.org> and <https://test.pypi.org>:
  owner `arielmission-space`, repo `ExoSim2.0`, workflow `release.yml`,
  environment `pypi` / `testpypi`.
- Create the GitHub environments `pypi` and `testpypi` in the `ExoSim2.0` repo.
- Make sure the release author can fast-forward `main` (branch protection).
