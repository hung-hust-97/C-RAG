SHELL := /bin/bash
SETUP_SCRIPT := scripts/setup/setup.sh
SETUP_BASH ?= $(or $(firstword $(wildcard /opt/homebrew/bin/bash /usr/local/bin/bash /opt/local/bin/bash)),$(shell command -v bash 2>/dev/null),bash)
SETUP_OPTS ?=
COMPOSE_FILE ?= docker-compose.yml
DOCKER_COMPOSE ?= docker compose
SERVICE ?=
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_BLUE := \033[34m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m

ifeq ($(NO_COLOR),1)
COLOR_RESET :=
COLOR_BOLD :=
COLOR_BLUE :=
COLOR_GREEN :=
COLOR_YELLOW :=
endif

.PHONY: help configure env-base env-storage env-server env-validate env-backup env-security-check env-base-rewrite env-storage-rewrite env base storage server validate backup security security-check base-rewrite storage-rewrite compose-dev compose-dev-up compose-dev-down compose-dev-restart compose-dev-logs compose-dev-ps compose-prod compose-prod-up compose-prod-down compose-prod-restart compose-prod-logs compose-prod-ps

help:
	@printf "$(COLOR_BOLD)Interactive setup targets$(COLOR_RESET)\n"
	@printf "  $(COLOR_GREEN)make env-base$(COLOR_RESET)               Configure LLM, embedding, and reranker (run first)\n"
	@printf "  $(COLOR_GREEN)make env-storage$(COLOR_RESET)            Configure storage backends and databases\n"
	@printf "  $(COLOR_GREEN)make env-server$(COLOR_RESET)             Configure server, security, and SSL\n"
	@printf "  $(COLOR_GREEN)make env-validate$(COLOR_RESET)           Validate existing .env\n"
	@printf "  $(COLOR_GREEN)make env-security-check$(COLOR_RESET)     Audit existing .env for security risks\n"
	@printf "  $(COLOR_GREEN)make env-backup$(COLOR_RESET)             Backup current .env\n"
	@printf "  $(COLOR_GREEN)make env-base-rewrite$(COLOR_RESET)       Force-regenerate wizard-managed compose services during base setup\n"
	@printf "  $(COLOR_GREEN)make env-storage-rewrite$(COLOR_RESET)    Force-regenerate wizard-managed compose services during storage setup\n"
	@printf "  $(COLOR_GREEN)make base$(COLOR_RESET)                   Short form of make env-base (all env prefix can be stripped)\n"
	@printf "  $(COLOR_GREEN)make compose-dev$(COLOR_RESET)            Start full stack using compose dev profile\n"
	@printf "  $(COLOR_GREEN)make compose-prod$(COLOR_RESET)           Start full stack using compose prod profile\n"
	@printf "  $(COLOR_GREEN)make compose-dev-down$(COLOR_RESET)       Stop dev profile stack\n"
	@printf "  $(COLOR_GREEN)make compose-prod-down$(COLOR_RESET)      Stop prod profile stack\n"
	@printf "  $(COLOR_GREEN)make compose-dev-logs SERVICE=lightrag-dev$(COLOR_RESET)  Tail logs in dev profile\n"
	@printf "  $(COLOR_GREEN)make compose-prod-logs SERVICE=lightrag$(COLOR_RESET)      Tail logs in prod profile\n"
	@printf "\n"
	@printf "$(COLOR_BOLD)Typical workflow$(COLOR_RESET)\n"
	@printf "  1. make env-base       # set LLM/embedding/reranker\n"
	@printf "  2. make env-storage    # set storage backends (optional)\n"
	@printf "  3. make env-server     # set port/security/SSL (optional)\n\n"
	@printf "$(COLOR_BOLD)Examples$(COLOR_RESET)\n"
	@printf "  make env-base\n"
	@printf "  make env-storage SETUP_OPTS=--debug\n"
	@printf "  make env-server\n\n"
	@printf "  make env-storage-rewrite\n\n"
	@printf "  make env-security-check\n\n"
	@printf "$(COLOR_BOLD)Compose Output$(COLOR_RESET)\n"
	@printf "  Bundled service images are defined in scripts/setup/templates/*.yml.\n"
	@printf "  Compose file output: docker-compose.final.yml\n"

env-base env base configure:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --base $(SETUP_OPTS)

env-storage storage:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --storage $(SETUP_OPTS)

env-base-rewrite base-rewrite:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --base --rewrite-compose $(SETUP_OPTS)

env-storage-rewrite storage-rewrite:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --storage --rewrite-compose $(SETUP_OPTS)

env-server server:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --server $(SETUP_OPTS)

env-validate validate:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --validate $(SETUP_OPTS)

env-security-check security security-check:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --security-check $(SETUP_OPTS)

env-backup backup:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --backup $(SETUP_OPTS)

compose-dev compose-dev-up:
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile dev up -d --build

compose-dev-down:
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile dev down

compose-dev-restart:
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile dev restart

compose-dev-logs:
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile dev logs -f --tail=200 $(SERVICE)

compose-dev-ps:
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile dev ps

compose-prod compose-prod-up:
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile prod up -d --build

compose-prod-down:
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile prod down

compose-prod-restart:
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile prod restart

compose-prod-logs:
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile prod logs -f --tail=200 $(SERVICE)

compose-prod-ps:
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile prod ps
