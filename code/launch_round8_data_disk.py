"""Use the unchanged data-disk launcher against an isolated Round8 source tree.

The approved connection route and prior launchers remain bytewise unchanged.
All model/data/result paths in job argv/configs remain explicit.
"""
import json
import posixpath
import sys
from pathlib import Path

import launch_remote_data_disk as launcher
from remote import REMOTE_ROOT


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: python code/launch_round8_data_disk.py JOB.json')
    job_path = Path(sys.argv[1])
    job = json.loads(job_path.read_text(encoding='utf-8-sig'))
    source = posixpath.normpath(job['round8_source_root'])
    allowed = posixpath.normpath(REMOTE_ROOT) + '/stages/round8_'
    if not source.startswith(allowed) or not source[len(allowed):].replace('_', '').isalnum():
        raise ValueError('Round8 sources must be isolated under the project data disk')
    launcher.REMOTE_ROOT = source
    # The launch control body uses only the standard library. Keep it outside
    # the relocated Conda tree; the job's argv declares its data-volume runtime.
    launcher.PYTHON = '/usr/bin/python3'
    launcher.main()


if __name__ == '__main__':
    main()
