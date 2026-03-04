# Logical Commit Workflow

This repository can enforce small, reversible, meaningful commits via local git hooks.

## Enable Once Per Clone

```bash
cd /workspaces/walberla-dev/walberla
bash utilities/git/setup-logical-commits.sh
```

## What Is Enforced

- `commit-msg` hook:
  - Subject length must be `<= 72` characters.
  - Subject must use `<area>: <meaningful change>`.
  - Generic subjects like `wip`, `update`, `misc` are rejected.
- `pre-commit` hook:
  - Rejects oversized staged commits by default:
    - max files: `20`
    - max line delta (`added + deleted`): `600`

## Tune Limits

```bash
LOGICAL_COMMIT_MAX_FILES=30 LOGICAL_COMMIT_MAX_LINES=1000 git commit
```

## Exceptional Bypass

Use only when there is a clear reason to keep one larger commit.

```bash
LOGICAL_COMMIT_BYPASS=1 git commit
```

## Commit Template

The setup script configures `.gitmessage-logical-unit.txt`:

```text
<area>: <meaningful change>

Why:
- <problem or motivation>

What:
- <key implementation detail>

Validation:
- <command and result>
```
