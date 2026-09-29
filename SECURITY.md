# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately, not in a public issue.

- Preferred: GitHub private vulnerability reporting, via the "Report a
  vulnerability" button on the repository's Security tab
  (<https://github.com/heykav/microprice-rust/security/advisories/new>).
- Alternative: email heykavofficial@gmail.com.

Include the affected version or commit, a description of the problem, and
steps to reproduce it if you have them.

## What to expect

This is a single-maintainer research project. Reports are handled on a
best-effort basis, with no guaranteed response or fix time. There is no bug
bounty.

## Supported versions

Only the latest commit on `main` is supported. The project has not made a
stable release.

## Scope

The code is research software and is not intended to be used for trading or
in any setting where a defect could cause financial loss. Reports about the
parsing of untrusted input files (CSV, Parquet, serialized models), the
Python bindings, the helper scripts in `scripts/`, and the GitHub Actions
workflows are all in scope.
