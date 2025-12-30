# Repository Identity Check

## Commands

```
$ gh auth status
github.com
  ✓ Logged in to github.com account DiogoRibeiro7 (keyring)
  - Active account: true
  - Git operations protocol: ssh
  - Token: gho_************************************
  - Token scopes: 'admin:public_key', 'gist', 'read:org', 'repo'

$ gh repo view diogoribeiro7/condensite-cde --json nameWithOwner,url,sshUrl,defaultBranchRef
{"defaultBranchRef":{"name":"master"},"nameWithOwner":"DiogoRibeiro7/condensite-cde","sshUrl":"git@github.com:DiogoRibeiro7/condensite-cde.git","url":"https://github.com/DiogoRibeiro7/condensite-cde"}

$ git remote -v
origin	git@github.com:DiogoRibeiro7/condensite-cde.git (fetch)
origin	git@github.com:DiogoRibeiro7/condensite-cde.git (push)
```

## Result

PASS — Remote `origin` points to `git@github.com:DiogoRibeiro7/condensite-cde.git` and the GitHub repo `DiogoRibeiro7/condensite-cde` exists with default branch `master`.

## Metadata Alignment (2025-12-22)

- Updated `pyproject.toml` so `[project].name` is `condensite-cde`, added `Diogo Ribeiro <dfr@esmad.ipp.pt>` to the authors list, and pointed all URLs to `https://github.com/diogoribeiro7/condensite-cde`.
- Added Diogo to `CITATION.cff` and ensured the repository link matches the GitHub project.
- Adjusted `README.md` heading to `# condensite-cde` and replaced the corrupted “Condensit‚” text with “Condensite”.
- Ran `poetry check` after the edits; it completed with only the existing Poetry warnings about license syntax.

- Verified CITATION.cff lists Diogo Ribeiro and the correct repository URL; updated AUTHORS.md to include Diogo Ribeiro <dfr@esmad.ipp.pt>.
