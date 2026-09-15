PROJECT_DIR := $(CURDIR)
SCRIPTS_DIR := $(PROJECT_DIR)/scripts

RUN_SESSION := preg-comp
DEV_SESSION := preg-comp-dev
MONITOR_SESSION := preg-comp-monitor

.PHONY: help check run dev monitor stop clean

help:
	@echo "  make check    - check project structure and code"
	@echo "  make run      - run Streamlit application"
	@echo "  make dev      - start development environment"
	@echo "  make monitor  - start system monitoring"
	@echo "  make stop     - stop tmux sessions"
	@echo "  make clean    - remove temporary files"

run:
	@$(SCRIPTS_DIR)/run.sh

dev:
	@$(SCRIPTS_DIR)/dev.sh $(PROJECT_DIR)

monitor:
	@$(SCRIPTS_DIR)/monitor.sh $(PROJECT_DIR)

stop:
	@tmux kill-session -t $(DEV_SESSION) 2>/dev/null || true
	@tmux kill-session -t $(MONITOR_SESSION) 2>/dev/null || true
	@tmux kill-session -t $(RUN_SESSION) 2>/dev/null || true
	@echo "tmux sessions stopped"

clean:
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@echo "Temporary Python files removed"
