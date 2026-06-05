# PHASE 5: Workflow Activation — Architecture

**Duration:** 1 day
**Depends On:** Phase 4 (Baseline)

---

## Overview

Start the daemon, enable file watching, integrate with CI. The system is now live.

## Components

### 5.1 HarnessDaemon

```rust
pub struct HarnessDaemon {
    db: HarnessDb,
    watcher: FileWatcher,
    processor: EventProcessor,
    notifier: NotificationService,
}

impl HarnessDaemon {
    pub fn run(&self) -> ! {
        // Start file watcher thread
        let watcher_handle = self.watcher.start();
        
        // Process events forever
        loop {
            let events = self.db.poll_events()?;
            for event in events {
                self.processor.process(event)?;
            }
            self.notifier.flush()?;
        }
    }
}
```

### 5.2 CI Integration

```yaml
# .github/workflows/harness.yml
name: V2 Harness
on: [push, pull_request]

jobs:
  test-stale:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Get stale tests
        run: trinity-harness query needs-testing --format=json > stale.json
      - name: Run stale tests only
        run: trinity-harness run-stale --input=stale.json
      - name: Update state
        run: trinity-harness update-from-results
```

### 5.3 Notification Service

- Pub/sub for state changes
- Webhook integration
- UI updates (future)

## Acceptance Criteria

- [ ] Daemon starts and watches filesystem
- [ ] File changes trigger events
- [ ] Events processed, states updated
- [ ] CI workflow runs stale tests only
- [ ] CI updates state after test run
