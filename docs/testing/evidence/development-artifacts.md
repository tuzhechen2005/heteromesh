# Development delivery TDD evidence

Author: `/root/product`. No stable release, publishing credential or H3 support claim.

- RED: verifier tests could not import the missing delivery module.
- GREEN: eight tests cover development-only SHA manifests, traversal/credential/weight archive rejection and source symlink rejection.
- RED: committed-source test failed before archive_source existed.
- GREEN: nine tests verify exact committed bytes despite dirty/untracked checkout files and reject mismatched commit identity.
- Initial local packaging probe built wheel/sdist, installed wheel into a new venv outside the repository, verified installed module paths and all three JSON schemas, ran console/module CLI help, rebuilt the sdist into another wheel and repeated installation smoke. This exploratory dirty-tree probe is not a release artifact; the final workflow uses exact git-archived source.
- Dedicated CI repeats actual build/install smoke on Linux, Windows and macOS and uploads only verified development distributions plus checksums under commit-specific names. Hardware fields remain NOT_RUN; this manifest is not a hardware approval gate.
