# audit/issues — checked-in issue stubs

The Cursor agent that authored this directory **does not have GitHub credentials**
(no `gh` CLI installed, no `GITHUB_TOKEN` / `GH_TOKEN` in env, no `.github/`
configuration in this repo). The brief in chat asked for two GitHub issues to
be opened so out-of-Phase-1 finds don't get lost. Until the real issues are
opened, the same content lives here, version-controlled, and is therefore
impossible to lose.

## How to convert these stubs into real GitHub issues

Once `gh` is installed and authenticated (`gh auth login`), from the repo root:

```powershell
gh issue create `
  --title "[Phase 2.5] auto_restart regression in bot-edit POST handler (worker_api.py:5444)" `
  --body-file audit/issues/phase-2-5-auto-restart-regression.md

gh issue create `
  --title "[Phase 3.2] hard_sl_pct defaults to 0.0 server-side (worker_api.py:5459)" `
  --body-file audit/issues/phase-3-2-hard-sl-pct-default.md
```

Then add the resulting URLs back into `audit/db_writers.md` §6 and delete
the corresponding files in this directory in the same commit.
