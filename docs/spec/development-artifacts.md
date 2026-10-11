# Development artifact delivery contract

Scope: REQ-014/017/020 installation and CI/CD evidence, development packages only.

- DIST-01: build both a Python wheel and source distribution from the reviewed source commit. Retain package version 0.1.0 but label delivery experimental/development; no PyPI, GitHub release or stable H3 capability publication.
- DIST-02: install the built wheel in a fresh virtual environment, run outside the repository with PYTHONPATH/PYTHONHOME removed, verify installed module locations, packaged JSON schemas and both console/module CLI help entry points. This must not pass by importing checkout files or a preexisting editable install.
- DIST-03: build a wheel from the produced sdist in a separate extraction/build directory. This verifies the sdist actually includes the package/schema sources needed for installation.
- DIST-04: inspect distributions for absolute/traversal archive paths and forbidden credential/runtime directories; produce SHA256/size records tied to the complete 40-character commit SHA. The manifest describes build/install checks, not hardware execution or model support.
- DIST-05: GitHub CI builds/tests development artifacts on Linux, Windows and macOS and uploads only the wheel, sdist and manifest under a name containing full commit SHA and platform, after successful smoke checks. No secret-bearing logs, model weights or local runtime directories are uploaded.
- DIST-06: workflow has contents:read and no publication credentials; failures stop upload. No automated release approval based solely on self-reported hardware JSON is implemented.

The default unit suite tests the delivery verifier without network/build dependencies. The dedicated CI job performs actual packaging and fresh-environment smoke tests. G1 installability does not prove H3, iPhone, distinct physical devices or distributed capacity.

Build sources come from a checked HEAD's exact `git archive`, not dirty tracked or untracked checkout files. The archive rejects links/traversal paths before extraction. Commit mismatch fails before building; the CI checkout explicitly selects the same SHA used in artifact names and manifest.
