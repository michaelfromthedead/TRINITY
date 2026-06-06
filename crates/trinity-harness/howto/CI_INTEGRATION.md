# How To: CI Integration

> **Note:** This applies to **v1** (rusqlite with `.harness/state.db`). V2 will use SuperSQLite's `brain.db`.

Set up GitHub Actions to run only affected tests.

## Generated Workflow

The harness can generate a workflow file:

```rust
use trinity_harness::ci::{WorkflowConfig, generate_yaml};

let config = WorkflowConfig::new("harness")
    .on_push("main")
    .on_pull_request("main");

let yaml = generate_yaml(&config);
std::fs::write(".github/workflows/harness.yml", yaml)?;
```

## Manual Setup

Create `.github/workflows/harness.yml`:

```yaml
name: Trinity Harness

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

jobs:
  smart-test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Need full history for diff
      
      - name: Setup Rust
        uses: dtolnay/rust-action@stable
      
      - name: Build harness
        run: cargo build -p trinity-harness --release
      
      - name: Query stale tests
        id: query
        run: |
          ./target/release/trinity-harness query needs-testing --format json > stale.json
          echo "count=$(jq '.count' stale.json)" >> $GITHUB_OUTPUT
      
      - name: Run stale tests
        if: steps.query.outputs.count > 0
        run: ./target/release/trinity-harness run-stale
      
      - name: Update state
        run: ./target/release/trinity-harness update
      
      - name: Upload state
        uses: actions/upload-artifact@v4
        with:
          name: harness-state
          path: .harness/state.db
```

## Caching State Between Runs

Add state caching for faster subsequent runs:

```yaml
      - name: Restore harness state
        uses: actions/cache@v4
        with:
          path: .harness/state.db
          key: harness-${{ runner.os }}-${{ github.sha }}
          restore-keys: |
            harness-${{ runner.os }}-
      
      # ... run tests ...
      
      - name: Save harness state
        uses: actions/cache/save@v4
        with:
          path: .harness/state.db
          key: harness-${{ runner.os }}-${{ github.sha }}
```

## Matrix Testing

Test across multiple configurations:

```yaml
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, nightly]
    
    runs-on: ${{ matrix.os }}
    
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-action@${{ matrix.rust }}
      - run: cargo build -p trinity-harness
      - run: ./target/debug/trinity-harness run-stale
```

## PR Comments

Post test results as PR comments:

```yaml
      - name: Generate report
        run: |
          ./target/release/trinity-harness query needs-testing --format markdown > report.md
      
      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('report.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## Harness Report\n\n${report}`
            });
```

## Integration with Existing Test Suites

If you already have `cargo test` and `pytest` workflows:

```yaml
      # Run harness first to identify what needs testing
      - name: Query affected
        run: |
          trinity-harness query needs-testing --output affected.txt
      
      # Pass to cargo test
      - name: Run Rust tests
        run: |
          if [ -s affected.txt ]; then
            cargo test $(cat affected.txt | grep '\.rs' | xargs -I{} echo "--test {}")
          fi
      
      # Pass to pytest
      - name: Run Python tests
        run: |
          if [ -s affected.txt ]; then
            pytest $(cat affected.txt | grep '\.py' | tr '\n' ' ')
          fi
```

## Notifications

Send Slack/Discord notifications on failure:

```yaml
      - name: Notify on failure
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "Harness tests failed on ${{ github.ref }}",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*<${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}|View Run>*"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```
