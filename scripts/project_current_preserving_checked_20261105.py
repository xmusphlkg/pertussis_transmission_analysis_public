"""Run the original projection producer with explicit archived-source revisions.

The three prototype sources were replaced by import shims after these historical
inputs were produced. Their exact original bytes are retained as source evidence.
The original producer records each allowed revision and still performs the full
saved-current trajectory replay. No fitted/transmission equations are changed.
"""
from scripts import run_current_preserving_strategies_20261103 as producer
_RUN_COUNTRY = producer.run_country


def configure():
    # Historical compatibility modules are selected by their paths.
    return None


def run_country(country, output_dir):
    # Also executes in a spawned worker; do not depend on forked global state.
    configure()
    return _RUN_COUNTRY(country, output_dir)


def main():
    configure()
    producer.run_country = run_country
    producer.main()


if __name__ == '__main__': main()
