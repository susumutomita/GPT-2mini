.PHONY: install
install:
	bun install

.PHONY: install_ci
install_ci:
	bun install --frozen-lockfile

.PHONY: clean
clean:
	bun run clean

.PHONY: lint_text
lint_text:
	bun run lint:text

.PHONY: before-commit
before-commit: lint_text
