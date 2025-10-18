# Repository Guidelines

## Project Structure & Module Organization
The entry points live in `cmd/api` (REST + SSE + static hosting) and `cmd/trainer` (training loop that publishes NATS events). Shared logic sits in `internal/`, with subpackages for config parsing (`internal/config`), data loading (`internal/data`), event payloads (`internal/events`), the JetStream wrapper (`internal/natsbus`), and network primitives (`internal/snn`). `web/index.html` provides the lightweight dashboard served by the API, while `config.yaml` defines NATS, training, and UI settings in JSON form. Keep generated datasets under `.data/` and out of version control.

## Build, Test, and Development Commands
- `go build ./cmd/api` — verify the UI/API binary compiles against Go 1.22.
- `go build ./cmd/trainer` — check the training agent and event emitters.
- `go run ./cmd/api` and `go run ./cmd/trainer` — iterate locally once NATS (default `nats://127.0.0.1:4222`) is running.
- `go test ./...` — run the suite; currently returns immediately because no `_test.go` files exist, so add coverage as you extend packages.

## Coding Style & Naming Conventions
Format Go with `go fmt ./...` or your editor’s gofmt integration; do not hand-format. Follow standard Go naming: exported API types/functions are PascalCase, intra-package helpers stay camelCase. Keep package-level variables minimal and prefer dependency injection via function parameters. Maintain concise, contextual comments only where logic is non-obvious (see the SSE handler in `cmd/api/main.go` for tone).

## Testing Guidelines
Use table-driven tests under the matching package directory (e.g., `internal/snn/net_test.go`). Aim to cover data loaders, event serialization, and NATS helper behaviour. When dealing with JetStream, abstract connections to allow deterministic fakes, and clean up temporary directories during tests.

## Commit & Pull Request Guidelines
Commit history is short and message style informal; adopt a clearer convention going forward: `area: succinct imperative` (e.g., `trainer: cap residual jitter`). Reference issue IDs when available. Each PR should include a summary of behaviour changes, testing evidence (`go test ./...`, manual API checks), and screenshots of the UI if visuals shift. Ensure `config.yaml` diffs explain why overrides are safe for other environments.

## Configuration & Runtime Notes
`config.yaml` must remain valid JSON; inline comments will break `internal/config.Load`. Provide sample overrides via README snippets instead of editing the default file. When introducing new NATS subjects or UI routes, update both the config structure and downstream handlers to keep the API, trainer, and dashboard in sync.
