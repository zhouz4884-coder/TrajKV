"""R12 exact reviewed scanner-source handling, with all other scans intact.

Only the immutable scanner source itself has four reviewed regex-definition
lines masked while scanning. Any body change disables this narrow exception.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import export_public as frozen

SCANNER_RELATIVE = 'ops/export_public.py'
SCANNER_SHA256 = 'ae711282f004b6704d76e2f09901ef840480d505706b50bf44a3fd8a02a030ef'
# Hashes cover exact lines including their original line endings. No regex
# patterns or connection examples are repeated in this helper.
REVIEWED_REGEX_LINES = {
    128: '880147fee7f883c5509856a0d33531a625a7c8a6e78b6dd1c0890c064f2a7356',
    186: '50e832a9cedc0b88a190689e313fdb17bd8dfb8f67e1c04f0adc4a7af796b158',
    187: '09ec401f43aff4b39d19de9441aee315b742624fdffe320be356baa8c16d6f86',
    189: 'ed930b42e2ffee898398d58580e5c6f465b4eb12d7b1bc1152cfdb2cd5c55e50',
}


def digest(body):
    return hashlib.sha256(body).hexdigest()


def verify_frozen_scanner():
    if digest(Path(frozen.__file__).read_bytes()) != SCANNER_SHA256:
        raise ValueError('R12 reviewed frozen scanner implementation changed')


def reviewed_scanner(relative, body):
    return relative == SCANNER_RELATIVE and digest(body) == SCANNER_SHA256


def findings(relative, body):
    verify_frozen_scanner()
    if reviewed_scanner(relative, body):
        lines = body.splitlines(keepends=True)
        for number, expected in REVIEWED_REGEX_LINES.items():
            if digest(lines[number - 1]) != expected:
                raise ValueError('R12 reviewed regex-definition line changed')
            lines[number - 1] = b'\n'
        body = b''.join(lines)
    return frozen.security_findings(relative, body)


def file_findings(relative, path):
    """Keep whole-text regex semantics; stream only validated JSONL records.

    Non-JSONL may contain whitespace-spanning patterns and is scanned whole.
    JSONL strings cannot contain literal newlines; complete JSON validity and
    row identity are independently required by the semantic verifier.
    """
    verify_frozen_scanner()
    path = Path(path)
    if path.suffix.lower() != '.jsonl':
        return findings(relative, path.read_bytes())
    result = set()
    with path.open('rb') as stream:
        for line in stream:
            result.update(frozen.security_findings(relative, line))
    return sorted(result)
