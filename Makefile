SHELL := /usr/bin/env bash
PROJECT_ROOT := $(CURDIR)
SCRIPT := bash scripts/project.sh
PORT ?= 8501

.PHONY: help init run start stop status find grep analyze system tmux clean

help:
	@$(SCRIPT) --help

init:
	@$(SCRIPT) init

run: init
	@PORT=$(PORT) $(SCRIPT) run

start: init
	@PORT=$(PORT) $(SCRIPT) start

stop:
	@$(SCRIPT) stop

status:
	@$(SCRIPT) status

find:
	@$(SCRIPT) find '$(or $(PATTERN),*.py)'

grep:
	@test -n "$(PATTERN)" || (echo 'Usage: make grep PATTERN=parser'; exit 2)
	@$(SCRIPT) grep "$(PATTERN)"

analyze: init
	@$(SCRIPT) analyze

system:
	@$(SCRIPT) system

tmux:
	@tmux new-session -d -s preg-comp 'cd $(PROJECT_ROOT) && PORT=$(PORT) $(SCRIPT) run' 2>/dev/null || true
	@echo 'tmux session: preg-comp'
	@tmux ls | grep '^preg-comp:' || true

clean:
	@$(SCRIPT) clean
