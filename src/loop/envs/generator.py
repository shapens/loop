from __future__ import annotations

import random

from loop.types import Split, TaskSpec

FAMILIES: tuple[str, ...] = (
    "keyset_paging",
    "ledger_invariants",
    "rate_limiter_window",
)


def _split_offset(split: Split) -> int:
    # Strictly disjoint seed namespaces for train vs held-out eval.
    return 10_000 if split == "train" else 900_000


def _make_keyset_paging(seed: int, split: Split) -> TaskSpec:
    rng = random.Random(seed)
    prefix = rng.choice(["cur", "pg", "tok", "ks"])
    tie_desc = rng.choice([True, False])
    tie_word = "highest id first" if tie_desc else "lowest id first"
    default_limit = rng.choice([3, 4, 5])

    sort_bug = (
        "def sort_key(record: dict[str, int]) -> tuple[int, int]:\n"
        '    """Order records by timestamp ascending, breaking ties by id."""\n'
        "    # BUG: missing tie-breaker on id\n"
        "    return (record['ts'], 0)\n"
    )
    tie_expr = '-record["id"]' if tie_desc else 'record["id"]'
    sort_fix = (
        "def sort_key(record: dict[str, int]) -> tuple[int, int]:\n"
        f'    """Order records by timestamp ascending, breaking ties ({tie_word})."""\n'
        f"    return (record['ts'], {tie_expr})\n"
    )

    cursor_bug = (
        f'PREFIX = "{prefix}:"\n\n'
        "def encode_cursor(ts: int, item_id: int) -> str:\n"
        '    return f"{PREFIX}{ts}:{item_id}"\n\n'
        "def decode_cursor(raw: str) -> tuple[int, int]:\n"
        "    # BUG: accepts arbitrary trailing segments or non-matching prefix\n"
        '    parts = raw.split(":")\n'
        "    return int(parts[1]), int(parts[2])\n"
    )
    cursor_fix = (
        f'PREFIX = "{prefix}:"\n\n'
        "def encode_cursor(ts: int, item_id: int) -> str:\n"
        '    return f"{PREFIX}{ts}:{item_id}"\n\n'
        "def decode_cursor(raw: str) -> tuple[int, int]:\n"
        "    if not isinstance(raw, str) or not raw.startswith(PREFIX):\n"
        '        raise ValueError("invalid cursor prefix")\n'
        "    body = raw[len(PREFIX):]\n"
        '    parts = body.split(":")\n'
        "    if len(parts) != 2:\n"
        '        raise ValueError("invalid cursor format")\n'
        "    if not all(p and (p.isdigit() or (p[0] == '-' and p[1:].isdigit())) for p in parts):\n"
        '        raise ValueError("invalid cursor integers")\n'
        "    return int(parts[0]), int(parts[1])\n"
    )

    after_key_expr = f"(after_ts, {'-after_id' if tie_desc else 'after_id'})"
    sig = (
        f"def list_page(\n"
        f"    records: list[dict[str, int]],\n"
        f"    limit: int = {default_limit},\n"
        f"    cursor: str | None = None,\n"
        f") -> tuple[list[dict[str, int]], str | None]:\n"
    )

    api_bug = (
        "from src.cursor import decode_cursor, encode_cursor\n"
        "from src.sort_key import sort_key\n\n"
        + sig
        + "    if limit <= 0:\n"
        + '        raise ValueError("limit must be positive")\n'
        + "    ordered = sorted(records, key=sort_key)\n"
        + "    if cursor is not None:\n"
        + "        after_ts, after_id = decode_cursor(cursor)\n"
        + f"        after_key = {after_key_expr}\n"
        + "        ordered = [r for r in ordered if sort_key(r) > after_key]\n"
        + "    page = ordered[:limit]\n"
        + "    # BUG: emits next_cursor even when page is exact end of list\n"
        + '    nxt = encode_cursor(page[-1]["ts"], page[-1]["id"])\n'
        + "    next_cursor = nxt if len(page) == limit else None\n"
        + "    return page, next_cursor\n"
    )
    api_fix = (
        "from src.cursor import decode_cursor, encode_cursor\n"
        "from src.sort_key import sort_key\n\n"
        + sig
        + "    if limit <= 0:\n"
        + '        raise ValueError("limit must be positive")\n'
        + "    ordered = sorted(records, key=sort_key)\n"
        + "    if cursor is not None:\n"
        + "        after_ts, after_id = decode_cursor(cursor)\n"
        + f"        after_key = {after_key_expr}\n"
        + "        ordered = [r for r in ordered if sort_key(r) > after_key]\n"
        + "    page = ordered[:limit]\n"
        + "    has_more = len(ordered) > limit\n"
        + '    nxt = encode_cursor(page[-1]["ts"], page[-1]["id"]) if page else None\n'
        + "    next_cursor = nxt if has_more else None\n"
        + "    return page, next_cursor\n"
    )

    visible_test = (
        "import unittest\n"
        "from src.api import list_page\n\n"
        "class TestVisiblePaging(unittest.TestCase):\n"
        "    def test_basic_ordering(self) -> None:\n"
        '        rows = [{"ts": 20, "id": 1}, {"ts": 10, "id": 2}, {"ts": 30, "id": 3}]\n'
        "        page, nxt = list_page(rows, limit=2)\n"
        '        self.assertEqual([r["id"] for r in page], [2, 1])\n'
        "        self.assertIsNotNone(nxt)\n"
        "        page2, nxt2 = list_page(rows, limit=2, cursor=nxt)\n"
        '        self.assertEqual([r["id"] for r in page2], [3])\n'
        "        self.assertIsNone(nxt2)\n"
    )

    expected_tie = [30, 20, 10] if tie_desc else [10, 20, 30]
    hidden_test = (
        "import unittest\n"
        "from src.api import list_page\n"
        "from src.cursor import decode_cursor\n\n"
        "class TestHiddenPaging(unittest.TestCase):\n"
        "    def test_same_timestamp_tie_break_across_pages(self) -> None:\n"
        '        rows = [{"ts": 5, "id": 20}, {"ts": 5, "id": 10}, {"ts": 5, "id": 30}]\n'
        "        seen = []\n"
        "        cur = None\n"
        "        for _ in range(5):\n"
        "            page, cur = list_page(rows, limit=2, cursor=cur)\n"
        '            seen.extend(r["id"] for r in page)\n'
        "            if cur is None:\n"
        "                break\n"
        f"        self.assertEqual(seen, {expected_tie})\n\n"
        "    def test_exact_multiple_has_no_next_cursor(self) -> None:\n"
        '        rows = [{"ts": i, "id": i} for i in range(4)]\n'
        "        page, nxt = list_page(rows, limit=2)\n"
        "        self.assertIsNotNone(nxt)\n"
        "        page2, nxt2 = list_page(rows, limit=2, cursor=nxt)\n"
        "        self.assertEqual(len(page2), 2)\n"
        "        self.assertIsNone(nxt2)\n\n"
        "    def test_cursor_validation(self) -> None:\n"
        f'        for bad in ["wrong:1:2", "{prefix}:1:2:3", "{prefix}:1", "{prefix}:a:2", ""]:\n'
        "            with self.assertRaises(ValueError):\n"
        "                decode_cursor(bad)\n"
    )

    prompt = (
        "Fix keyset pagination bugs across `src/sort_key.py`, `src/cursor.py`, `src/api.py`:\n"
        f"1. Records must sort by `ts` ascending, breaking ties by `{tie_word}`.\n"
        "2. `list_page` must return `next_cursor=None` when no items follow a full page.\n"
        "3. `decode_cursor` must strictly accept only strings produced by `encode_cursor` "
        f"(prefix `{prefix}:` followed by exactly two integers) and raise `ValueError`.\n"
        "Do not modify test files."
    )

    repo_files = {
        "src/__init__.py": "",
        "src/sort_key.py": sort_bug,
        "src/cursor.py": cursor_bug,
        "src/api.py": api_bug,
        "tests/__init__.py": "",
        "tests/test_visible.py": visible_test,
    }
    solution_files = {
        "src/sort_key.py": sort_fix,
        "src/cursor.py": cursor_fix,
        "src/api.py": api_fix,
    }
    return TaskSpec(
        task_id=f"{split}-keyset_paging-{seed}",
        family="keyset_paging",
        split=split,
        seed=seed,
        prompt=prompt,
        repo_files=repo_files,
        hidden_tests={"test_hidden_paging.py": hidden_test},
        solution_files=solution_files,
        allowed_edit_globs=("src/*",),
    )


def _make_ledger_invariants(seed: int, split: Split) -> TaskSpec:
    rng = random.Random(seed)
    scale = rng.choice([100, 1000])
    decimals = 2 if scale == 100 else 3
    inclusive = rng.choice([True, False])
    cutoff_word = "inclusive (`ts <= as_of`)" if inclusive else "exclusive (`ts < as_of`)"

    money_bug = (
        f"SCALE = {scale}\n"
        f"DECIMALS = {decimals}\n\n"
        "def format_minor(amount: int) -> str:\n"
        "    # BUG: loses negative sign when -SCALE < amount < 0\n"
        "    whole = int(amount / SCALE)\n"
        "    frac = abs(amount) % SCALE\n"
        '    return f"{whole}.{frac:0{DECIMALS}d}"\n'
    )
    money_fix = (
        f"SCALE = {scale}\n"
        f"DECIMALS = {decimals}\n\n"
        "def format_minor(amount: int) -> str:\n"
        '    sign = "-" if amount < 0 else ""\n'
        "    mag = abs(amount)\n"
        "    whole = mag // SCALE\n"
        "    frac = mag % SCALE\n"
        '    return f"{sign}{whole}.{frac:0{DECIMALS}d}"\n'
    )

    cmp_op = "<=" if inclusive else "<"
    wrong_op = "<" if inclusive else "<="

    store_bug = (
        "class Ledger:\n"
        "    def __init__(self) -> None:\n"
        "        self._entries: list[tuple[int, str, int]] = []\n"
        "        self._version = 0\n"
        "        self._cache: dict[tuple[int, str, int], int] = {}\n\n"
        "    def post(self, ts: int, account: str, minor: int) -> None:\n"
        "        self._entries.append((ts, account, minor))\n"
        "        self._version += 1\n\n"
        "    def rename_account(self, old: str, new: str) -> None:\n"
        "        # BUG: mutates entries without bumping _version, leaving stale cache hits\n"
        "        self._entries = [\n"
        "            (t, new if a == old else a, m) for (t, a, m) in self._entries\n"
        "        ]\n\n"
        "    def balance(self, account: str, as_of: int) -> int:\n"
        "        key = (self._version, account, as_of)\n"
        "        if key in self._cache:\n"
        "            return self._cache[key]\n"
        "        total = sum(\n"
        f"            m for (t, a, m) in self._entries if a == account and t {wrong_op} as_of\n"
        "        )\n"
        "        self._cache[key] = total\n"
        "        return total\n"
    )
    store_fix = (
        "class Ledger:\n"
        "    def __init__(self) -> None:\n"
        "        self._entries: list[tuple[int, str, int]] = []\n"
        "        self._version = 0\n"
        "        self._cache: dict[tuple[int, str, int], int] = {}\n\n"
        "    def post(self, ts: int, account: str, minor: int) -> None:\n"
        "        self._entries.append((ts, account, minor))\n"
        "        self._version += 1\n\n"
        "    def rename_account(self, old: str, new: str) -> None:\n"
        "        self._entries = [\n"
        "            (t, new if a == old else a, m) for (t, a, m) in self._entries\n"
        "        ]\n"
        "        self._version += 1\n\n"
        "    def balance(self, account: str, as_of: int) -> int:\n"
        "        key = (self._version, account, as_of)\n"
        "        if key in self._cache:\n"
        "            return self._cache[key]\n"
        "        total = sum(\n"
        f"            m for (t, a, m) in self._entries if a == account and t {cmp_op} as_of\n"
        "        )\n"
        "        self._cache[key] = total\n"
        "        return total\n"
    )

    visible_test = (
        "import unittest\n"
        "from src.money import format_minor\n"
        "from src.store import Ledger\n\n"
        "class TestVisibleLedger(unittest.TestCase):\n"
        "    def test_positive_formatting(self) -> None:\n"
        f'        self.assertEqual(format_minor({scale + 5}), "1.{"0" * (decimals - 1)}5")\n\n'
        "    def test_basic_post(self) -> None:\n"
        "        led = Ledger()\n"
        '        led.post(1, "cash", 500)\n'
        '        self.assertEqual(led.balance("cash", as_of=10), 500)\n'
    )

    expected_at_boundary = 500 if inclusive else 0
    neg_val = -(scale // 2)
    expected_neg = f"-0.{abs(neg_val):0{decimals}d}"
    hidden_test = (
        "import unittest\n"
        "from src.money import format_minor\n"
        "from src.store import Ledger\n\n"
        "class TestHiddenLedger(unittest.TestCase):\n"
        "    def test_sub_unit_negative_keeps_sign(self) -> None:\n"
        f'        self.assertEqual(format_minor({neg_val}), "{expected_neg}")\n'
        f'        self.assertEqual(format_minor(0), "0.{"0" * decimals}")\n\n'
        "    def test_cutoff_boundary(self) -> None:\n"
        "        led = Ledger()\n"
        '        led.post(10, "cash", 500)\n'
        f'        self.assertEqual(led.balance("cash", as_of=10), {expected_at_boundary})\n\n'
        "    def test_rename_invalidates_balance_cache(self) -> None:\n"
        "        led = Ledger()\n"
        '        led.post(1, "old", 300)\n'
        '        self.assertEqual(led.balance("old", as_of=5), 300)\n'
        '        self.assertEqual(led.balance("new", as_of=5), 0)\n'
        '        led.rename_account("old", "new")\n'
        '        self.assertEqual(led.balance("old", as_of=5), 0)\n'
        '        self.assertEqual(led.balance("new", as_of=5), 300)\n'
    )

    prompt = (
        "Fix the bugs in `src/money.py` and `src/store.py`:\n"
        f"1. `format_minor` must keep the minus sign for negative sub-unit values "
        f"(e.g. `{neg_val}` -> `{expected_neg}`).\n"
        f"2. `Ledger.balance(account, as_of)` must use a {cutoff_word} timestamp filter.\n"
        "3. `Ledger.rename_account` must keep cached balances coherent (no stale hits).\n"
        "Do not modify test files."
    )

    return TaskSpec(
        task_id=f"{split}-ledger_invariants-{seed}",
        family="ledger_invariants",
        split=split,
        seed=seed,
        prompt=prompt,
        repo_files={
            "src/__init__.py": "",
            "src/money.py": money_bug,
            "src/store.py": store_bug,
            "tests/__init__.py": "",
            "tests/test_visible.py": visible_test,
        },
        hidden_tests={"test_hidden_ledger.py": hidden_test},
        solution_files={"src/money.py": money_fix, "src/store.py": store_fix},
        allowed_edit_globs=("src/*",),
    )


def _make_rate_limiter_window(seed: int, split: Split) -> TaskSpec:
    rng = random.Random(seed)
    window_ms = rng.choice([1000, 2000, 5000])
    capacity = rng.choice([5, 8, 10])

    init_sig = (
        "    def __init__(\n"
        f"        self, capacity: int = {capacity}, window_ms: int = {window_ms}\n"
        "    ) -> None:\n"
    )
    limiter_bug = (
        "class SlidingWindowLimiter:\n"
        + init_sig
        + "        self.capacity = capacity\n"
        + "        self.window_ms = window_ms\n"
        + "        self._events: dict[str, list[tuple[int, int]]] = {}\n\n"
        + "    def allow(self, key: str, now_ms: int, cost: int = 1) -> bool:\n"
        + "        if cost <= 0:\n"
        + '            raise ValueError("cost must be positive")\n'
        + "        cutoff = now_ms - self.window_ms\n"
        + "        # BUG 1: keeps events at exact cutoff (should expire when ts <= cutoff)\n"
        + "        kept = [(t, c) for (t, c) in self._events.get(key, []) if t >= cutoff]\n"
        + "        # BUG 2: counts number of events instead of total cost\n"
        + "        used = len(kept)\n"
        + "        if used + cost > self.capacity:\n"
        + "            self._events[key] = kept\n"
        + "            return False\n"
        + "        kept.append((now_ms, cost))\n"
        + "        self._events[key] = kept\n"
        + "        return True\n"
    )
    limiter_fix = (
        "class SlidingWindowLimiter:\n"
        + init_sig
        + "        self.capacity = capacity\n"
        + "        self.window_ms = window_ms\n"
        + "        self._events: dict[str, list[tuple[int, int]]] = {}\n\n"
        + "    def allow(self, key: str, now_ms: int, cost: int = 1) -> bool:\n"
        + "        if cost <= 0:\n"
        + '            raise ValueError("cost must be positive")\n'
        + "        cutoff = now_ms - self.window_ms\n"
        + "        kept = [(t, c) for (t, c) in self._events.get(key, []) if t > cutoff]\n"
        + "        used = sum(c for (_, c) in kept)\n"
        + "        if used + cost > self.capacity:\n"
        + "            self._events[key] = kept\n"
        + "            return False\n"
        + "        kept.append((now_ms, cost))\n"
        + "        self._events[key] = kept\n"
        + "        return True\n"
    )

    visible_test = (
        "import unittest\n"
        "from src.limiter import SlidingWindowLimiter\n\n"
        "class TestVisibleLimiter(unittest.TestCase):\n"
        "    def test_unit_cost_limit(self) -> None:\n"
        f"        lim = SlidingWindowLimiter(capacity=2, window_ms={window_ms})\n"
        '        self.assertTrue(lim.allow("a", 100))\n'
        '        self.assertTrue(lim.allow("a", 200))\n'
        '        self.assertFalse(lim.allow("a", 300))\n'
    )

    hidden_test = (
        "import unittest\n"
        "from src.limiter import SlidingWindowLimiter\n\n"
        "class TestHiddenLimiter(unittest.TestCase):\n"
        "    def test_exact_window_expiry(self) -> None:\n"
        f"        lim = SlidingWindowLimiter(capacity=2, window_ms={window_ms})\n"
        '        self.assertTrue(lim.allow("k", 100, cost=2))\n'
        f'        self.assertFalse(lim.allow("k", {100 + window_ms - 1}, cost=1))\n'
        f'        self.assertTrue(lim.allow("k", {100 + window_ms}, cost=2))\n\n'
        "    def test_weighted_cost_accounting(self) -> None:\n"
        f"        lim = SlidingWindowLimiter(capacity={capacity}, window_ms={window_ms})\n"
        f'        self.assertTrue(lim.allow("k", 10, cost={capacity - 1}))\n'
        '        self.assertFalse(lim.allow("k", 20, cost=2))\n'
        '        self.assertTrue(lim.allow("k", 20, cost=1))\n'
    )

    prompt = (
        "Fix the sliding-window rate limiter in `src/limiter.py`:\n"
        "1. Expire events once `now_ms - t >= window_ms` (keep `t > now_ms - window_ms`).\n"
        "2. Capacity accounting must sum weighted `cost` values, not event count.\n"
        "3. Rejected requests must not consume capacity.\n"
        "Do not modify test files."
    )

    return TaskSpec(
        task_id=f"{split}-rate_limiter_window-{seed}",
        family="rate_limiter_window",
        split=split,
        seed=seed,
        prompt=prompt,
        repo_files={
            "src/__init__.py": "",
            "src/limiter.py": limiter_bug,
            "tests/__init__.py": "",
            "tests/test_visible.py": visible_test,
        },
        hidden_tests={"test_hidden_limiter.py": hidden_test},
        solution_files={"src/limiter.py": limiter_fix},
        allowed_edit_globs=("src/*",),
    )


def generate_task(family: str, seed: int, split: Split = "train") -> TaskSpec:
    if family == "keyset_paging":
        return _make_keyset_paging(seed, split)
    if family == "ledger_invariants":
        return _make_ledger_invariants(seed, split)
    if family == "rate_limiter_window":
        return _make_rate_limiter_window(seed, split)
    raise ValueError(f"Unknown task family: {family}")


def generate_split(split: Split, count: int) -> list[TaskSpec]:
    base = _split_offset(split)
    tasks: list[TaskSpec] = []
    for i in range(count):
        family = FAMILIES[i % len(FAMILIES)]
        tasks.append(generate_task(family=family, seed=base + i, split=split))
    return tasks
