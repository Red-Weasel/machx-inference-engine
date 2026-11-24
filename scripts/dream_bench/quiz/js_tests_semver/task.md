A teammate wrote `compareSemver(a, b)` in `solution.mjs` (an ES module, `export function compareSemver`). It compares two Semantic Versioning 2.0.0 strings and returns -1, 0 or 1:

- `MAJOR.MINOR.PATCH` are compared numerically (`1.10.0` > `1.9.0`). No leading zeros, no `v` prefix.
- A pre-release (`1.0.0-alpha.1`) is lower than the same version without one (`1.0.0-alpha` < `1.0.0`).
- Pre-release identifiers are compared one dot-separated field at a time: numeric fields numerically (`alpha.2` < `alpha.10`), other fields by ASCII order, a numeric field is lower than a non-numeric one, and when all shared fields are equal the shorter list is lower (`1.0.0-alpha` < `1.0.0-alpha.1`).
- Build metadata after `+` is ignored (`1.0.0+build.5` equals `1.0.0+build.9`).
- An invalid version string (`"1.2"`, `"01.2.3"`, `"v1.2.3"`, `"1.2.3-"`, or a numeric pre-release field with a leading zero such as `"1.0.0-01"`) makes it throw an `Error`.

Write `test_solution.mjs`: a test script that does `import { compareSemver } from './solution.mjs'`, checks the behaviour above with `node:assert`, and exits non-zero if anything is wrong. It must pass on a correct implementation and fail on implementations with bugs. Do not write the implementation itself.
