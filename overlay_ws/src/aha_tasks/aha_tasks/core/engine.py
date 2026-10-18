"""Subgoal execution with partial runs, checkpoints and rollback (ROS-independent).

Task memory is checkpointed before every subgoal. Subgoals outside the selection
are not executed; their expected/mock state is assumed instead. Physical state
(e.g. ``robot.location``) is brought in line with memory by restorers just before
the next executed subgoal, so a partial run starts from the state its first
subgoal expects. A failed subgoal rolls memory back to its checkpoint, i.e. the
state after the previous subgoal, and the failure policy decides what follows.
"""

from collections import deque
import copy
import threading
import time

from aha_tasks.core.skill import SkillContext, SkillResult, normalize
from aha_tasks.core.spec import select

PENDING = "pending"
BYPASSED = "bypassed"  # not selected; mock state assumed
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
UNSUPPORTED = "unsupported"  # selected, but no implementation; mock state assumed
SKIPPED = "skipped"  # failed, then continued with mock state
CANCELLED = "cancelled"

POLICIES = ("pause", "retry", "skip", "abort")
STARTABLE = ("idle", "completed", "failed", "aborted")


class _Abort(Exception):
    pass


class TaskEngine:
    def __init__(
        self,
        spec,
        skills,
        restorers=None,
        publish=None,
        logger=None,
        default_selection=None,
        default_policy="pause",
        max_retries=1,
    ):
        """``skills`` maps names to instances (None: not implemented).

        ``restorers`` maps memory keys to ``fn(value, cancelled) -> result`` that
        make the physical state match a remembered value.
        """
        if default_policy not in POLICIES:
            raise ValueError(f"Unknown failure policy: {default_policy}")
        self.spec = spec
        self.skills = skills
        self.restorers = restorers or {}
        self._publish = publish
        self._logger = logger
        self.default_selection = select(spec, default_selection)
        self.default_policy = default_policy
        self.max_retries = max_retries
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._cancel = threading.Event()
        self._thread = None
        self._decision = None
        self._cursor = None
        self._selected = set()
        self._run_count = 0
        self.state = "preparing"
        self.phase = None
        self.message = ""
        self.current = None
        self.run = None
        self.log = deque(maxlen=200)
        self.memory = copy.deepcopy(spec.initial_state)
        # Physical state assumed at startup, e.g. the spawn location.
        self.realized = {
            k: copy.deepcopy(v)
            for k, v in spec.initial_state.items()
            if k in self.restorers
        }
        self.records = {}
        self._reset_records()

    # ---- external API (any thread) -------------------------------------------

    def notify(self, message):
        with self._lock:
            self.message = message
            self._log(message)

    def set_ready(self, message="Ready"):
        with self._lock:
            self.state, self.message = "idle", message
            self._log(message)

    def set_error(self, message):
        with self._lock:
            self.state, self.message = "error", message
            self._log(message, level="error")

    def start(self, selection=None, on_failure=None):
        with self._lock:
            if self.state not in STARTABLE:
                return False, f"Cannot start while {self.state}"
            policy = on_failure or self.default_policy
            if policy not in POLICIES:
                return False, f"Unknown failure policy: {policy}"
            try:
                ids = select(self.spec, selection or self.default_selection)
            except ValueError as error:
                return False, str(error)
            self._run_count += 1
            self.run = {
                "number": self._run_count,
                "selection": ids,
                "on_failure": policy,
                "max_retries": self.max_retries,
                "started_at": time.time(),
                "finished_at": None,
            }
            self._selected = set(ids)
            self.memory = copy.deepcopy(self.spec.initial_state)
            self._reset_records()
            self._cancel.clear()
            self._decision = None
            self.state, self.message = "running", ""
            self._log(
                f"Run {self._run_count}: {len(ids)} subgoal(s), on failure {policy}"
            )
            self._thread = threading.Thread(
                target=self._worker, args=(policy,), daemon=True
            )
            self._thread.start()
        return True, f"Started {len(ids)} subgoal(s)"

    def command(self, name, target=None):
        with self._lock:
            if name == "abort":
                if self.state not in ("running", "paused"):
                    return False, "Nothing to abort"
                self._cancel.set()
                self._decision = ("abort", None)
                self._changed.notify_all()
                self._log("Abort requested", level="warn")
                return True, "Abort requested"
            if name not in ("retry", "skip", "rewind"):
                return False, f"Unknown command: {name}"
            if self.state != "paused":
                return False, f"{name} is only available while paused"
            index = None
            if name == "rewind":
                try:
                    index = self._rewind_index(target)
                except ValueError as error:
                    return False, str(error)
            self._decision = (name, index)
            self._changed.notify_all()
            suffix = f" to {self.spec.subgoals[index].id}" if name == "rewind" else ""
            return True, f"{name}{suffix} accepted"

    def finished(self):
        thread = self._thread
        return thread is None or not thread.is_alive()

    def shutdown(self, timeout=None):
        self.command("abort")
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return self.finished()

    def status(self):
        with self._lock:
            selected = self._selected if self.run else set(self.default_selection)
            subgoals = []
            for subgoal in self.spec.subgoals:
                item = {
                    "id": subgoal.id,
                    "title": subgoal.title,
                    "group": subgoal.group,
                    "skill": subgoal.skill,
                    "available": self.skills.get(subgoal.skill) is not None,
                    "points": subgoal.points,
                    "note": subgoal.note,
                    "selected": subgoal.id in selected,
                }
                item.update(copy.deepcopy(self.records[subgoal.id]))
                subgoals.append(item)
            return {
                "schema_version": 1,
                "task": {
                    "id": self.spec.task,
                    "title": self.spec.title,
                    "rulebook": self.spec.rulebook,
                    "time_limit": self.spec.time_limit,
                },
                "state": self.state,
                "phase": self.phase,
                "message": self.message,
                "current": self.current,
                "run": copy.deepcopy(self.run),
                "defaults": {
                    "selection": list(self.default_selection),
                    "on_failure": self.default_policy,
                    "max_retries": self.max_retries,
                },
                "policies": list(POLICIES),
                "subgoals": subgoals,
                "memory": copy.deepcopy(self.memory),
                "realized": copy.deepcopy(self.realized),
                "log": list(self.log),
            }

    # ---- worker ---------------------------------------------------------------

    def _worker(self, policy):
        subgoals = self.spec.subgoals
        last = max(i for i, s in enumerate(subgoals) if s.id in self._selected)
        checkpoints, cursor, retries, outcome = {}, 0, 0, "completed"
        try:
            while cursor <= last:
                subgoal = subgoals[cursor]
                with self._lock:
                    self._cursor = cursor
                    checkpoints[cursor] = copy.deepcopy(self.memory)
                if self._cancel.is_set():
                    raise _Abort
                skill = self.skills.get(subgoal.skill)
                if subgoal.id not in self._selected:
                    self._assume(subgoal, BYPASSED, "Not selected; mock state assumed")
                    cursor += 1
                    continue
                if skill is None:
                    self._assume(
                        subgoal,
                        UNSUPPORTED,
                        f"Skill '{subgoal.skill}' is not implemented; mock state assumed",
                    )
                    cursor += 1
                    continue
                if self._attempt(subgoal, skill):
                    cursor, retries = cursor + 1, 0
                    continue
                if self._cancel.is_set():
                    raise _Abort
                self._rollback(subgoal, checkpoints[cursor])
                decision, target = policy, None
                if policy == "retry":
                    if retries < self.max_retries:
                        retries += 1
                    else:
                        decision = "abort"
                elif policy == "pause":
                    self._sync()  # physically return to the previous subgoal's state
                    decision, target = self._await_decision(subgoal)
                if decision == "retry":
                    self._log(f"Retrying {subgoal.id}")
                    continue
                if decision == "skip":
                    self._assume(subgoal, SKIPPED, None)
                    cursor, retries = cursor + 1, 0
                    continue
                if decision == "rewind":
                    with self._lock:
                        self.memory = copy.deepcopy(checkpoints[target])
                        for earlier in subgoals[target : cursor + 1]:
                            # Keep attempts/history for the analysis of retries.
                            self.records[earlier.id].update(status=PENDING, message="")
                        self._log(f"Rewound to {subgoals[target].id}")
                    cursor, retries = target, 0
                    continue
                outcome = "failed"
                break
        except _Abort:
            outcome = "aborted"
            with self._lock:
                if self._cursor in checkpoints:
                    self.memory = copy.deepcopy(checkpoints[self._cursor])
                for key in self.restorers:
                    self.realized.pop(key, None)
        except Exception as error:  # restorer/engine bug: report, never hang
            outcome = "failed"
            self._log(f"Internal error: {type(error).__name__}: {error}", level="error")
        finally:
            with self._lock:
                counts = {}
                for record in self.records.values():
                    counts[record["status"]] = counts.get(record["status"], 0) + 1
                if outcome == "completed" and counts.get(SKIPPED):
                    outcome = "failed"
                self.state, self.phase, self.current = outcome, None, None
                self.run["finished_at"] = time.time()
                self.message = ", ".join(
                    f"{counts[s]} {s}"
                    for s in (SUCCEEDED, FAILED, SKIPPED, UNSUPPORTED, CANCELLED)
                    if counts.get(s)
                )
                self._log(
                    f"Run {outcome.upper()}: {self.message}",
                    level="info" if outcome == "completed" else "error",
                )

    def _attempt(self, subgoal, skill):
        record = self.records[subgoal.id]
        with self._lock:
            record["attempts"] += 1
            record.update(status=RUNNING, message="", duration=None)
            self.current = subgoal.id
            self._log(f"Starting {subgoal.id} (attempt {record['attempts']})")
        ok, message = self._sync()
        result, duration = SkillResult(False, message), 0.0
        if ok:
            with self._lock:
                self.phase = "executing"
                context = SkillContext(
                    subgoal=subgoal.id,
                    params=copy.deepcopy(subgoal.params),
                    memory=copy.deepcopy(self.memory),
                    cancelled=self._cancel,
                    log=lambda text: self._log(text, subgoal=subgoal.id),
                )
                self._emit()
            started = time.monotonic()
            try:
                result = normalize(skill.execute(context))
            except Exception as error:
                result = SkillResult(False, f"{type(error).__name__}: {error}")
            duration = time.monotonic() - started
        success = result.success and not self._cancel.is_set()
        with self._lock:
            if success:
                self.memory.update(copy.deepcopy(skill.expected(subgoal.params)))
                self.memory.update(copy.deepcopy(result.outputs))
                for key in self.restorers:
                    if self.memory.get(key) is None:
                        self.realized.pop(key, None)
                    else:
                        self.realized[key] = copy.deepcopy(self.memory[key])
                status = SUCCEEDED
            else:
                status = CANCELLED if self._cancel.is_set() else FAILED
            text = result.message or ("" if success else status.capitalize())
            record.update(status=status, message=text, duration=round(duration, 3))
            record["history"].append(
                {"status": status, "message": text, "duration": round(duration, 3)}
            )
            self.phase = None
            self._log(
                f"{status.upper()} {subgoal.id}" + (f": {text}" if text else ""),
                level="info" if success else "error",
                subgoal=subgoal.id,
            )
        return success

    def _sync(self):
        """Make physical state match memory; returns (ok, message)."""
        for key, restore in self.restorers.items():
            with self._lock:
                target = copy.deepcopy(self.memory.get(key))
                if target is None or self.realized.get(key) == target:
                    continue
                self.phase = "restoring"
                self._log(f"Restoring {key} = {target}")
            try:
                result = normalize(restore(target, self._cancel))
            except Exception as error:
                result = SkillResult(False, f"{type(error).__name__}: {error}")
            with self._lock:
                self.phase = None
                if self._cancel.is_set():
                    self.realized.pop(key, None)
                    return False, f"Cancelled while restoring {key} = {target}"
                if not result.success:
                    self.realized.pop(key, None)
                    message = f"Could not restore {key} = {target}: {result.message}"
                    self._log(message, level="error")
                    return False, message
                self.realized[key] = target
                self._emit()
        return True, ""

    def _rollback(self, subgoal, checkpoint):
        with self._lock:
            self.memory = copy.deepcopy(checkpoint)
            # A failed attempt may have moved the robot partially.
            for key in self.restorers:
                self.realized.pop(key, None)
            self._log(f"Rolled back to the state before {subgoal.id}", level="warn")

    def _await_decision(self, subgoal):
        with self._lock:
            self.state, self.phase = "paused", None
            self.message = f"{subgoal.id} failed: choose retry, rewind, skip or abort"
            self._decision = None
            self._emit()
            while self._decision is None:
                self._changed.wait()
            decision, self._decision = self._decision, None
            if decision[0] == "abort":
                raise _Abort
            self.state, self.message = "running", ""
            self._emit()
            return decision

    def _assume(self, subgoal, status, message):
        with self._lock:
            skill = self.skills.get(subgoal.skill)
            if skill is not None:
                self.memory.update(copy.deepcopy(skill.expected(subgoal.params)))
            self.memory.update(copy.deepcopy(subgoal.mock))
            record = self.records[subgoal.id]
            if status == SKIPPED:
                message = f"Skipped after failure: {record['message']}"
            record.update(status=status, message=message)
            if status == BYPASSED:
                self._emit()
            else:
                self._log(f"{status.upper()} {subgoal.id}: {message}", level="warn")

    def _rewind_index(self, target):
        if target:
            index = self.spec.index(target)
            if index > self._cursor:
                raise ValueError(f"Cannot rewind forward to {target}")
            return index
        # Unsupported/bypassed ones would only re-apply their mock: skip over them.
        for index in range(self._cursor - 1, -1, -1):
            if self.records[self.spec.subgoals[index].id]["status"] == SUCCEEDED:
                return index
        raise ValueError("No earlier executed subgoal; use retry")

    # ---- helpers (call with the lock held) ------------------------------------

    def _reset_records(self):
        for subgoal in self.spec.subgoals:
            self.records[subgoal.id] = {
                "status": PENDING,
                "attempts": 0,
                "message": "",
                "duration": None,
                "history": [],
            }

    def _log(self, message, level="info", subgoal=None):
        with self._lock:
            self.log.append(
                {
                    "time": time.time(),
                    "level": level,
                    "subgoal": subgoal,
                    "message": message,
                }
            )
            if self._logger is not None:
                self._logger(level, message)
            self._emit()

    def _emit(self):
        if self._publish is not None:
            self._publish(self.status())
