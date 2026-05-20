.PHONY: setup validate migrate tables report test smoke article m5 lint plot server-bench

HW ?= Mac M3
ARTICLE ?= 1

setup:
	./scripts/setup_env.sh

validate:
	python scripts/validate_results.py --hardware "$(HW)"

migrate:
	python scripts/migrate_results.py --apply --hardware "$(HW)" --delete-legacy-files

migrate-dry:
	python scripts/migrate_results.py --dry-run --hardware "$(HW)"

tables:
	python scripts/generate_article_tables.py --hardware "$(HW)" --article $(ARTICLE) \
		-o docs/articles/_generated/tables_article$(ARTICLE)_$(shell echo "$(HW)" | tr ' /' '__').md

report:
	python scripts/report.py --hardware "$(HW)" --migrate-dry-run \
		-o docs/articles/_generated/report_$(shell echo "$(HW)" | tr ' /' '__').md

test:
	pytest tests/ -q

lint:
	ruff check scripts/benchmark scripts/benchmark_schema.py \
		scripts/run_benchmark.py scripts/plot_results.py \
		scripts/benchmark_server.py tests

plot:
	python scripts/plot_results.py --hardware "$(HW)" --preset llama3-8b
	@if [ -d results/Mac_M5_Max ]; then \
	  python scripts/plot_results.py --hardware "$(HW)" --compare-hardware "Mac M5 Max" --config w4+kv_cache+prefill; \
	fi

server-bench:
	python scripts/benchmark_server.py --hardware "$(HW)" --preset llama3-8b --config w4

smoke:
	python scripts/run_benchmark.py --dry-run --preset llama3-8b --config w4
	python scripts/compare_runtimes.py --dry-run --hardware "$(HW)"

article:
	./scripts/run_article.sh $(ARTICLE) "$(HW)"

m5:
	./scripts/run_m5_ladder.sh "Mac M5 Max"
