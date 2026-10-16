# Sky Qiu's Quarto website

The site source is in the root `.qmd` files. Quarto renders the website to `docs/`.
The TMLE paper catalog is maintained from Semantic Scholar by standard-library
Python scripts and reviewed through pull requests.

## Refresh the TMLE catalog locally

Use Python 3.12 and [Quarto 1.6.42](https://github.com/quarto-dev/quarto-cli/releases/tag/v1.6.42):

```sh
python3 -m unittest discover -s tests -v
python3 scripts/fetch_semantic_scholar_papers.py \
  --mode bulk --max-results-per-keyword 0 --request-interval-s 2 \
  --report /tmp/tmle-discovery-report.json
python3 scripts/generate_tmle_papers_qmd.py
quarto render tmle-papers.qmd
```

Run these commands in order, stopping if any command fails. The fetcher uses the
unauthenticated Semantic Scholar bulk API, requests newest papers first, and
follows pagination until every configured query is exhausted. The value `0`
means there is no local per-keyword result cap. A positive cap is only suitable
for bounded diagnostics: if more results remain, the run is incomplete and must
not replace the catalog. API errors and truncated queries produce a nonzero exit
status, leave the previous catalog intact, and are recorded in the report.

The default queries cover TMLE, targeted maximum likelihood estimation, targeted
minimum loss based estimation, super learner, super learning, and highly adaptive
lasso. Exact title/abstract matching, the minimum publication year, deduplication,
and `data/blocked_tmle_papers.json` still determine inclusion. Existing records are
retained and enriched by default; absence from a later search does not delete a
paper. Complete pagination does not guarantee that Semantic Scholar has indexed
every relevant paper, so human relevance review remains necessary.

## Weekly discovery and review

[`.github/workflows/tmle-papers.yml`](.github/workflows/tmle-papers.yml) runs on
Mondays at **00:17 UTC / 08:17 China Standard Time**. It also supports **Actions →
Review weekly TMLE papers → Run workflow** on the default branch. Manual runs
selected from another branch are skipped. GitHub schedules can be delayed and
run only when the workflow exists on the default branch; public repositories may
have schedules disabled after prolonged inactivity. See GitHub's
[schedule documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).

Each run:

1. Checks out the default branch, runs regression tests, and fetches every bulk
   result page without an arbitrary result cap.
2. Saves a diagnostic artifact containing the complete run report, query counts,
   errors, truncation status, newly added papers, and the fetch/generation/render
   logs when available. The artifact
   is retained for 30 days, including for failed or unchanged discovery runs when
   those files were produced.
3. Compares the paper records and keyword list with the default-branch catalog.
   With no content changes, it skips rendering and opening a PR; run timestamps
   and statistics remain in the artifact. When the same proposed catalog is
   already on the automation branch, its existing catalog metadata is reused to
   avoid timestamp-only updates to the open PR. An unchanged run never closes or
   rewrites an existing PR. If the older automation branch differs from the
   now-current catalog (for example, a proposed paper was since blocklisted), a
   prominent workflow warning and diagnostic summary flag the potentially
   obsolete proposal for manual review or closure before merging.
4. Generates `tmle-papers.qmd` and renders it with Quarto 1.6.42 when content
   changes. Unexpected changes outside the generated-file allowlist fail the run.
5. Opens or updates a **draft** PR from `automation/tmle-papers`, with counts,
   **every newly added paper title, all available author names, and paper links**.
   The list compares the final proposed catalog against the default-branch
   baseline, so enriched existing papers are not mislabeled as new. Author names
   are never shortened to “et al.”; missing source metadata is explicitly marked.
   A complete `newly-added-papers.md` is also saved in the diagnostic artifact.
   If the list would exceed the PR-description size budget, the PR prominently
   links to the run's artifacts and names this complete file instead of silently
   truncating papers or authors. Artifacts are retained for 30 days.
   Only the catalog JSON, TMLE QMD,
   rendered TMLE HTML, search index, and generated CSS under `docs/site_libs/` are
   eligible for the PR. The branch is reserved for automation and may be rewritten
   when the next successful run updates the PR.

The build job has read-only repository access. A separate PR job receives only
`contents: write` and `pull-requests: write`, using GitHub's ephemeral `GITHUB_TOKEN`.
Checkout does not persist credentials. The PR job consumes generated files from
that same successful run; it does not execute fetched paper content, generated
QMD, or repository scripts. It refuses the update if the default branch moved
between discovery and the PR job. Retry the workflow in that case.

All third-party actions are pinned to full commit SHAs. Python is fixed to the
3.12 series (security patch releases may change), and Quarto is fixed to 1.6.42.
The workflow requires no personal access token, bot account token, or Semantic
Scholar API secret. Action pin updates should be reviewed like other code changes.

### Repository setup and review

Before the first run, a repository administrator must ensure Actions is enabled
and **Settings → Actions → General → Workflow permissions → Allow GitHub Actions
to create and approve pull requests** is permitted by repository/organization
policy. The workflow only creates/updates draft PRs; it never approves them.
No settings are changed by the workflow. See the action's
[permissions documentation](https://github.com/peter-evans/create-pull-request#workflow-permissions).

Merging the workflow onto the default branch makes its schedule eligible to run.
Keep it on a review branch until ready to enable it. Creating or editing this file
locally does not activate a schedule.

Review the query diagnostics, newly added paper list, publication dates,
duplicates, blocklist exclusions, generated diff, and rendered page before
merging a refresh PR. A failed or incomplete discovery run cannot create or
update the review PR. The workflow does not merge changes, write generated files
directly to the default branch, or publish the site. Merging a PR may affect any
separately configured site deployment, which must be reviewed independently.

GitHub restricts downstream events created with `GITHUB_TOKEN`: do not assume
that a bot-created PR has run every other CI check. Check its status and approve
any pending workflow runs or start the appropriate existing manual checks before
merging. See GitHub's current
[workflow-trigger behavior](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow#triggering-a-workflow-from-a-workflow).
