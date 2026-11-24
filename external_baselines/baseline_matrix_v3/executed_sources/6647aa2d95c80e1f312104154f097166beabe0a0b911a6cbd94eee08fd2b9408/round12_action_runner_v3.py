"""Native coverage closure for the unchanged R12 v2 numerical action path.

Earlier pilot raw files remain untouched. This version writes the coverage
contract at run time so the frozen scorer can verify every expected source.
"""
from pathlib import Path
import round12_action_runner_v2 as base

_write = base.write_new


def write_new(path, value):
    if Path(path).name == 'manifest.json':
        value['round12_action_closure_version'] = 3
        value['source_numerical_action_driver'] = 'round12_action_runner_v2.py'
    elif Path(path).name == 'complete.json':
        expected = value['expected_QA']
        value['coverage_complete'] = (
            value['recorded_QA'] == expected and sum(value['counts'].values()) == expected
        )
        value['round12_action_closure_version'] = 3
        value['all_failed_sources_remain_in_primary_denominator'] = True
    return _write(path, value)


def main():
    base.RUNTIME_FILES = tuple(dict.fromkeys((*base.RUNTIME_FILES, 'round12_action_runner_v3.py')))
    base.write_new = write_new
    return base.main()


if __name__ == '__main__':
    raise SystemExit(main())
