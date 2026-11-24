"""Read a sealed compact Java material journal; no simulator or model calls."""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct

from .events import EVENT_FIELDS, Event, KINDS, ContractError

MAGIC = b'GMEMJ1\n'
TOKENIZATION = 'material-journal-v1-causal-aggregate'
HEADER_FIELDS = ('tokenization_version', 'source_kind', 'recording_id', 'episode_id',
                 'config_sha256', 'run_index', 'seed', 'shift_seconds', 'window_seconds',
                 'target_grade', 'grade_lower', 'grade_upper')
STR_FIELDS = {'batch_id', 'request_id'}
LONG_FIELDS = {'seq', 'decision_sequence'}
INT_FIELDS = {'window', 'source_id', 'truck_id'}
BOOL_FIELDS = {'grade_valid', 'window_grade_valid'}


def require(condition, message):
    if not condition:
        raise ContractError(message)


@dataclass(frozen=True)
class Journal:
    header: dict
    events: tuple
    raw_process_calls: int
    file_sha256: str
    logical_sha256: str


class Reader:
    def __init__(self, data): self.data, self.offset = data, 0

    def take(self, count):
        require(0 <= count <= len(self.data) - self.offset, 'truncated material journal')
        result = self.data[self.offset:self.offset+count]
        self.offset += count
        return result

    def number(self, fmt): return struct.unpack('>' + fmt, self.take(struct.calcsize('>' + fmt)))[0]

    def string(self):
        count = self.number('i')
        require(0 <= count <= 4096, 'invalid journal UTF-8 length')
        try:
            return self.take(count).decode('utf-8', errors='strict')
        except UnicodeError as exc:
            raise ContractError('invalid journal UTF-8') from exc


def validate_header(header):
    require(set(header) == set(HEADER_FIELDS), 'material journal header fields differ')
    require(header['tokenization_version'] == TOKENIZATION, 'unknown tokenization contract')
    require(header['source_kind'] in ('actual_anylogic', 'synthetic'), 'unknown material source kind')
    for key in ('recording_id', 'episode_id'):
        value = header[key]
        require(type(value) is str and value and value.strip() == value and
                not any(c in value for c in ('\x00', '\n', '\r')), 'invalid journal identity: ' + key)
    digest = header['config_sha256']
    require(type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
            'invalid configuration byte SHA256')
    for key in ('run_index', 'seed'):
        require(type(header[key]) is int and 0 < header[key] < 2**63, 'invalid journal run/seed')
    expected = dict(shift_seconds=28800., window_seconds=7200., target_grade=.27, grade_lower=.26, grade_upper=.28)
    for key, value in expected.items():
        require(type(header[key]) in (int, float) and math.isfinite(header[key]) and header[key] == value,
                'journal physical configuration differs: ' + key)
    return header


def read_journal(path, *, max_bytes=128 * 1024 * 1024, max_events=1000000):
    path = Path(path)
    require(path.suffix != '.partial' and path.is_file(), 'sealed journal required; partial files refused')
    require(path.stat().st_size <= max_bytes, 'material journal exceeds configured byte limit')
    data = path.read_bytes()
    reader = Reader(data)
    require(reader.take(len(MAGIC)) == MAGIC and reader.number('i') == 1, 'material journal magic/version differs')
    header = {key: reader.string() for key in HEADER_FIELDS[:5]}
    header.update({key: reader.number('q') for key in HEADER_FIELDS[5:7]})
    header.update({key: reader.number('d') for key in HEADER_FIELDS[7:]})
    validate_header(header)
    events = []
    while True:
        prefix_end = reader.offset
        kind = reader.number('B')
        if kind == 255:
            count, raw_process = reader.number('q'), reader.number('q')
            checksum_end = reader.offset
            digest = reader.take(32)
            require(reader.offset == len(data), 'bytes after material journal footer')
            require(count == len(events) and count > 0, 'material journal event count differs')
            require(raw_process >= sum(e.kind == 'PROCESS' for e in events), 'invalid raw PROCESS count')
            require(hashlib.sha256(data[:checksum_end]).digest() == digest, 'material journal logical SHA256 differs')
            break
        require(1 <= kind <= len(KINDS) and len(events) < max_events, 'unknown event tag or event limit exceeded')
        values = {'kind': KINDS[kind-1]}
        for field in EVENT_FIELDS:
            if field == 'kind': continue
            if field in STR_FIELDS: value = reader.string()
            elif field in LONG_FIELDS: value = reader.number('q')
            elif field in INT_FIELDS: value = reader.number('i')
            elif field in BOOL_FIELDS:
                value = reader.number('B')
                require(value in (0, 1), 'journal boolean must be 0 or 1')
                value = bool(value)
            else: value = reader.number('d')
            values[field] = value
        events.append(Event.from_dict(values))
    require(events[-1].kind == 'END', 'journal lacks true terminal END')
    return Journal(header, tuple(events), raw_process,
                   hashlib.sha256(data).hexdigest(), digest.hex())


def synthetic_journal_bytes(header, events, *, raw_process_calls=None):
    """Deterministic engineering fixture writer; never labels synthetic data real."""
    validate_header(header)
    require(header['source_kind'] == 'synthetic', 'fixture writer cannot create actual_anylogic journals')
    payload = bytearray(MAGIC)
    def number(fmt, value): payload.extend(struct.pack('>' + fmt, value))
    def string(value):
        encoded = value.encode('utf-8'); number('i', len(encoded)); payload.extend(encoded)
    number('i', 1)
    for field in HEADER_FIELDS[:5]: string(header[field])
    for field in HEADER_FIELDS[5:7]: number('q', header[field])
    for field in HEADER_FIELDS[7:]: number('d', header[field])
    events = tuple(events)
    for event in events:
        event = event if isinstance(event, Event) else Event.from_dict(event)
        number('B', KINDS.index(event.kind) + 1)
        for field in EVENT_FIELDS:
            if field == 'kind': continue
            value = getattr(event, field)
            if field in STR_FIELDS: string(value)
            elif field in LONG_FIELDS: number('q', value)
            elif field in INT_FIELDS: number('i', value)
            elif field in BOOL_FIELDS: number('B', int(value))
            else: number('d', value)
    number('B', 255); number('q', len(events))
    number('q', sum(e.kind == 'PROCESS' for e in events) if raw_process_calls is None else raw_process_calls)
    digest = hashlib.sha256(payload).digest()
    payload.extend(digest)
    return bytes(payload)
