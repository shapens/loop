from __future__ import annotations

import random

from loop.types import Split, TaskSpec

FAMILIES: tuple[str, ...] = (
    "keyset_paging",
    "ledger_invariants",
    "rate_limiter_window",
    "async_keyed_queue",
    "spec_invoice_calc",
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


def _make_async_keyed_queue(seed: int, split: Split) -> TaskSpec:
    rng = random.Random(seed)
    max_wait = rng.choice([2, 3, 4])

    queue_bug = (
        "from __future__ import annotations\n"
        "import asyncio\n"
        "from collections.abc import Awaitable, Callable\n"
        "from typing import TypeVar\n\n"
        'T = TypeVar("T")\n\n'
        "class KeyedSerializer:\n"
        '    """Serialize async jobs per key while allowing different keys in parallel."""\n\n'
        "    def __init__(self) -> None:\n"
        "        self._locks: dict[str, asyncio.Lock] = {}\n"
        "        self._active = 0\n\n"
        "    async def run(self, key: str, fn: Callable[[], Awaitable[T]]) -> T:\n"
        "        # BUG 1: naive lock eviction races when another waiter is queued\n"
        "        lock = self._locks.setdefault(key, asyncio.Lock())\n"
        "        await lock.acquire()\n"
        "        self._active += 1\n"
        "        try:\n"
        "            return await fn()\n"
        "        finally:\n"
        "            self._active -= 1\n"
        "            lock.release()\n"
        "            # BUG 2: deletes lock even if another task is waiting on it!\n"
        "            if not lock.locked():\n"
        "                self._locks.pop(key, None)\n\n"
        "    @property\n"
        "    def pending_keys(self) -> int:\n"
        "        return len(self._locks)\n"
    )
    queue_fix = (
        "from __future__ import annotations\n"
        "import asyncio\n"
        "from collections.abc import Awaitable, Callable\n"
        "from typing import TypeVar\n\n"
        'T = TypeVar("T")\n\n'
        "class KeyedSerializer:\n"
        '    """Serialize async jobs per key while allowing different keys in parallel."""\n\n'
        "    def __init__(self) -> None:\n"
        "        self._locks: dict[str, asyncio.Lock] = {}\n"
        "        self._refcounts: dict[str, int] = {}\n"
        "        self._active = 0\n\n"
        "    async def run(self, key: str, fn: Callable[[], Awaitable[T]]) -> T:\n"
        "        lock = self._locks.get(key)\n"
        "        if lock is None:\n"
        "            lock = asyncio.Lock()\n"
        "            self._locks[key] = lock\n"
        "            self._refcounts[key] = 0\n"
        "        self._refcounts[key] += 1\n"
        "        acquired = False\n"
        "        try:\n"
        "            await lock.acquire()\n"
        "            acquired = True\n"
        "            self._active += 1\n"
        "            try:\n"
        "                return await fn()\n"
        "            finally:\n"
        "                self._active -= 1\n"
        "        finally:\n"
        "            if acquired:\n"
        "                lock.release()\n"
        "            self._refcounts[key] -= 1\n"
        "            if self._refcounts[key] == 0:\n"
        "                del self._refcounts[key]\n"
        "                del self._locks[key]\n\n"
        "    @property\n"
        "    def pending_keys(self) -> int:\n"
        "        return len(self._locks)\n"
    )

    visible_test = (
        "import asyncio\n"
        "import unittest\n"
        "from src.queue import KeyedSerializer\n\n"
        "class TestVisibleQueue(unittest.TestCase):\n"
        "    def test_single_key_serial(self) -> None:\n"
        "        async def go() -> list[str]:\n"
        "            ser = KeyedSerializer()\n"
        "            log: list[str] = []\n"
        "            async def job(tag: str) -> None:\n"
        '                log.append(f"start:{tag}")\n'
        "                await asyncio.sleep(0.01)\n"
        '                log.append(f"end:{tag}")\n'
        "            await asyncio.gather(\n"
        '                ser.run("k", lambda: job("a")),\n'
        '                ser.run("k", lambda: job("b")),\n'
        "            )\n"
        "            return log\n"
        '        self.assertEqual(asyncio.run(go()), ["start:a", "end:a", "start:b", "end:b"])\n'
    )

    hidden_test = (
        "import asyncio\n"
        "import unittest\n"
        "from src.queue import KeyedSerializer\n\n"
        "class TestHiddenQueue(unittest.TestCase):\n"
        "    def test_handover_no_overlap_under_wave(self) -> None:\n"
        "        async def go() -> tuple[int, int]:\n"
        "            ser = KeyedSerializer()\n"
        "            in_flight = 0\n"
        "            peak = 0\n"
        "            gate1 = asyncio.Event()\n"
        "            async def holder1() -> None:\n"
        "                nonlocal in_flight, peak\n"
        "                in_flight += 1\n"
        "                peak = max(peak, in_flight)\n"
        "                await gate1.wait()\n"
        "                in_flight -= 1\n"
        "            async def slow_job() -> None:\n"
        "                nonlocal in_flight, peak\n"
        "                in_flight += 1\n"
        "                peak = max(peak, in_flight)\n"
        "                await asyncio.sleep(0.02)\n"
        "                in_flight -= 1\n"
        '            t1 = asyncio.create_task(ser.run("k", holder1))\n'
        "            await asyncio.sleep(0.002)\n"
        '            t2 = asyncio.create_task(ser.run("k", slow_job))\n'
        "            await asyncio.sleep(0.002)\n"
        "            gate1.set()\n"
        "            await t1\n"
        "            wave = [\n"
        f'                asyncio.create_task(ser.run("k", slow_job)) for _ in range({max_wait})\n'
        "            ]\n"
        "            await asyncio.gather(t2, *wave)\n"
        "            return peak, ser.pending_keys\n"
        "        peak, rem = asyncio.run(go())\n"
        "        self.assertEqual(peak, 1)\n"
        "        self.assertEqual(rem, 0)\n\n"
        "    def test_cancelled_waiter_does_not_leak_or_break_next(self) -> None:\n"
        "        async def go() -> tuple[str, int]:\n"
        "            ser = KeyedSerializer()\n"
        "            gate = asyncio.Event()\n"
        "            async def holder() -> str:\n"
        "                await gate.wait()\n"
        '                return "held"\n'
        "            async def quick() -> str:\n"
        '                return "ok"\n'
        '            t_hold = asyncio.create_task(ser.run("k", holder))\n'
        "            await asyncio.sleep(0.002)\n"
        '            t_wait = asyncio.create_task(ser.run("k", quick))\n'
        "            await asyncio.sleep(0.002)\n"
        "            t_wait.cancel()\n"
        "            with self.assertRaises(asyncio.CancelledError):\n"
        "                await t_wait\n"
        '            t_next = asyncio.create_task(ser.run("k", quick))\n'
        "            gate.set()\n"
        "            await t_hold\n"
        "            res = await t_next\n"
        "            return res, ser.pending_keys\n"
        "        res, rem = asyncio.run(go())\n"
        '        self.assertEqual(res, "ok")\n'
        "        self.assertEqual(rem, 0)\n"
    )

    prompt = (
        "Fix the concurrency and cleanup bugs in `src/queue.py` (`KeyedSerializer`):\n"
        "1. Concurrent `run(key, fn)` calls for the same key must never overlap, even when "
        "new callers arrive right as a holder releases the lock.\n"
        "2. Cancelling a waiting caller must not leak locks or corrupt subsequent callers.\n"
        "3. `pending_keys` must return `0` once all callers for a key finish or cancel.\n"
        "Do not modify test files."
    )

    return TaskSpec(
        task_id=f"{split}-async_keyed_queue-{seed}",
        family="async_keyed_queue",
        split=split,
        seed=seed,
        prompt=prompt,
        repo_files={
            "src/__init__.py": "",
            "src/queue.py": queue_bug,
            "tests/__init__.py": "",
            "tests/test_visible.py": visible_test,
        },
        hidden_tests={"test_hidden_queue.py": hidden_test},
        solution_files={"src/queue.py": queue_fix},
        allowed_edit_globs=("src/*",),
    )


def _make_spec_invoice_calc(seed: int, split: Split) -> TaskSpec:
    rng = random.Random(seed)
    tax_bps = rng.choice([500, 750, 825, 1000])  # 5%, 7.5%, 8.25%, 10%
    discount_cents = rng.choice([10, 17, 25])

    calc_bug = (
        "from __future__ import annotations\n\n"
        "def round_half_away(num: int, den: int) -> int:\n"
        "    # BUG 1: uses Python banker's rounding (half-to-even) instead of half-away-from-zero\n"
        "    return round(num / den)\n\n"
        "def allocate_discount(line_cents: list[int], discount: int) -> list[int]:\n"
        "    # BUG 2: integer division drops remainder cents instead of largest-remainder method\n"
        "    pos_total = sum(x for x in line_cents if x > 0)\n"
        "    if pos_total == 0 or discount <= 0:\n"
        "        return [0] * len(line_cents)\n"
        "    return [(x * discount) // pos_total if x > 0 else 0 for x in line_cents]\n\n"
        f"def compute_total(\n"
        f"    line_cents: list[int], discount: int = {discount_cents}, tax_bps: int = {tax_bps}\n"
        ") -> dict[str, int]:\n"
        "    # BUG 3: does not validate int inputs or cap discount to positive subtotal\n"
        "    alloc = allocate_discount(line_cents, discount)\n"
        "    net = [x - d for x, d in zip(line_cents, alloc)]\n"
        "    subtotal = sum(net)\n"
        "    tax = round_half_away(subtotal * tax_bps, 10_000)\n"
        "    return {\n"
        '        "subtotal": subtotal,\n'
        '        "discount": sum(alloc),\n'
        '        "tax": tax,\n'
        '        "total": subtotal + tax,\n'
        "    }\n"
    )

    calc_fix = (
        "from __future__ import annotations\n\n"
        "def round_half_away(num: int, den: int) -> int:\n"
        "    if den <= 0:\n"
        '        raise ValueError("den must be positive")\n'
        "    sign = -1 if num < 0 else 1\n"
        "    mag = abs(num)\n"
        "    q, r = divmod(mag, den)\n"
        "    if r * 2 >= den:\n"
        "        q += 1\n"
        "    return sign * q\n\n"
        "def allocate_discount(line_cents: list[int], discount: int) -> list[int]:\n"
        "    pos_indices = [i for i, x in enumerate(line_cents) if x > 0]\n"
        "    pos_total = sum(line_cents[i] for i in pos_indices)\n"
        "    if pos_total == 0 or discount <= 0:\n"
        "        return [0] * len(line_cents)\n"
        "    capped = min(discount, pos_total)\n"
        "    alloc = [0] * len(line_cents)\n"
        "    remainders: list[tuple[int, int]] = []\n"
        "    used = 0\n"
        "    for i in pos_indices:\n"
        "        num = line_cents[i] * capped\n"
        "        base = num // pos_total\n"
        "        alloc[i] = base\n"
        "        used += base\n"
        "        remainders.append((num % pos_total, -i))\n"
        "    rem = capped - used\n"
        "    remainders.sort(reverse=True)\n"
        "    for k in range(rem):\n"
        "        idx = -remainders[k][1]\n"
        "        alloc[idx] += 1\n"
        "    return alloc\n\n"
        f"def compute_total(\n"
        f"    line_cents: list[int], discount: int = {discount_cents}, tax_bps: int = {tax_bps}\n"
        ") -> dict[str, int]:\n"
        "    if not isinstance(discount, int) or isinstance(discount, bool):\n"
        '        raise TypeError("discount must be int")\n'
        "    if not isinstance(tax_bps, int) or isinstance(tax_bps, bool):\n"
        '        raise TypeError("tax_bps must be int")\n'
        "    for x in line_cents:\n"
        "        if not isinstance(x, int) or isinstance(x, bool):\n"
        '            raise TypeError("line_cents must contain ints")\n'
        "    alloc = allocate_discount(line_cents, discount)\n"
        "    net = [x - d for x, d in zip(line_cents, alloc)]\n"
        "    subtotal = sum(net)\n"
        "    tax = round_half_away(subtotal * tax_bps, 10_000)\n"
        "    return {\n"
        '        "subtotal": subtotal,\n'
        '        "discount": sum(alloc),\n'
        '        "tax": tax,\n'
        '        "total": subtotal + tax,\n'
        "    }\n"
    )

    visible_test = (
        "import unittest\n"
        "from src.invoice import compute_total\n\n"
        "class TestVisibleInvoice(unittest.TestCase):\n"
        "    def test_zero_discount(self) -> None:\n"
        "        res = compute_total([1000, 1000], discount=0, tax_bps=1000)\n"
        '        self.assertEqual(res["subtotal"], 2000)\n'
        '        self.assertEqual(res["tax"], 200)\n'
        '        self.assertEqual(res["total"], 2200)\n'
    )

    hidden_test = (
        "import unittest\n"
        "from src.invoice import allocate_discount, compute_total, round_half_away\n\n"
        "class TestHiddenInvoice(unittest.TestCase):\n"
        "    def test_half_away_from_zero(self) -> None:\n"
        "        self.assertEqual(round_half_away(5, 2), 3)\n"
        "        self.assertEqual(round_half_away(1, 2), 1)\n"
        "        self.assertEqual(round_half_away(-1, 2), -1)\n"
        "        self.assertEqual(round_half_away(-5, 2), -3)\n\n"
        "    def test_largest_remainder_exact_sum_and_positive_only(self) -> None:\n"
        f"        alloc = allocate_discount([100, 100, 100, -50], {discount_cents})\n"
        f"        self.assertEqual(sum(alloc), {discount_cents})\n"
        "        self.assertEqual(alloc[3], 0)\n"
        "        self.assertEqual(max(alloc[:3]) - min(alloc[:3]), 1 if "
        f"{discount_cents} % 3 else 0)\n\n"
        "    def test_discount_cap_and_type_checks(self) -> None:\n"
        "        res = compute_total([50, -10], discount=999, tax_bps=1000)\n"
        '        self.assertEqual(res["discount"], 50)\n'
        '        self.assertEqual(res["subtotal"], -10)\n'
        "        with self.assertRaises(TypeError):\n"
        "            compute_total([100], discount=True)\n"
        "        with self.assertRaises(TypeError):\n"
        "            compute_total([10.5], discount=0)\n"
    )

    prompt = (
        "Fix the invoice calculation bugs in `src/invoice.py`:\n"
        "1. `round_half_away(num, den)` must round exact halves away from zero for both "
        "positive and negative numbers (not banker's rounding).\n"
        "2. `allocate_discount(line_cents, discount)` must apply discount only to positive lines, "
        "cap discount at the positive subtotal, and distribute remainder cents via "
        "largest-remainder (tie-breaking to earliest index) so "
        "`sum(alloc) == min(discount, pos_total)`.\n"
        "3. `compute_total` must reject non-int or `bool` values with `TypeError`.\n"
        "Do not modify test files."
    )

    return TaskSpec(
        task_id=f"{split}-spec_invoice_calc-{seed}",
        family="spec_invoice_calc",
        split=split,
        seed=seed,
        prompt=prompt,
        repo_files={
            "src/__init__.py": "",
            "src/invoice.py": calc_bug,
            "tests/__init__.py": "",
            "tests/test_visible.py": visible_test,
        },
        hidden_tests={"test_hidden_invoice.py": hidden_test},
        solution_files={"src/invoice.py": calc_fix},
        allowed_edit_globs=("src/*",),
    )


def generate_task(family: str, seed: int, split: Split = "train") -> TaskSpec:
    if family == "keyset_paging":
        return _make_keyset_paging(seed, split)
    if family == "ledger_invariants":
        return _make_ledger_invariants(seed, split)
    if family == "rate_limiter_window":
        return _make_rate_limiter_window(seed, split)
    if family == "async_keyed_queue":
        return _make_async_keyed_queue(seed, split)
    if family == "spec_invoice_calc":
        return _make_spec_invoice_calc(seed, split)
    raise ValueError(f"Unknown task family: {family}")


def generate_split(split: Split, count: int) -> list[TaskSpec]:
    base = _split_offset(split)
    tasks: list[TaskSpec] = []
    for i in range(count):
        family = FAMILIES[i % len(FAMILIES)]
        tasks.append(generate_task(family=family, seed=base + i, split=split))
    return tasks
