from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# Optional imports for hardware backends.
try:
    import boto3
    from braket.aws import AwsDevice, AwsSession
    from braket.devices import Devices

    HAS_BRAKET = True
except Exception:
    HAS_BRAKET = False

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    HAS_IBM_RUNTIME = True
except Exception:
    HAS_IBM_RUNTIME = False

try:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    try:
        from qiskit.primitives import Sampler as LocalSampler

        LOCAL_SAMPLER_KIND = "v1"
    except Exception:
        from qiskit.primitives import StatevectorSampler as LocalSampler

        LOCAL_SAMPLER_KIND = "v2"
    HAS_QISKIT = True
except Exception:
    HAS_QISKIT = False
    LocalSampler = None
    LOCAL_SAMPLER_KIND = ""


SGT = timezone(timedelta(hours=8))
MIN_WINDOW_REMAINING_MINUTES = 10.0
MIN_AWS_WINDOW_REMAINING_MINUTES = 30.0
IBM_MIN_RUNTIME_SECONDS = 15.0

LOGGER = logging.getLogger("qobench.hardware_manager")


class QPUWindowExpiredError(RuntimeError):
    """Raised when a QPU's availability window has closed.

    Callers should catch this to attempt failover to another QPU.
    """
    def __init__(self, qpu_id: str, message: str | None = None) -> None:
        self.qpu_id = qpu_id
        super().__init__(
            message or f"QPU '{qpu_id}' availability window has closed."
        )


@dataclass(slots=True)
class QPUConfig:
    name: str
    provider: str
    max_qubits: int
    region: str = "us-west-1"
    priority: int = 10
    backend_name: str = ""
    is_available: bool = False
    last_error: str = ""
    total_jobs: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    total_time_sec: float = 0.0
    availability_windows_sgt: Optional[list[tuple[int, int] | tuple[int, int, int, int]]] = None
    weekdays_only: bool = False


# IDs used for iterating over AWS Braket QPUs in initialize / refresh.
_BRAKET_QPU_IDS = ("rigetti_ankaa3", "iqm_emerald", "iqm_garnet", "amazon_sv1")


def _default_qpus() -> dict[str, QPUConfig]:
    return {
        "rigetti_ankaa3": QPUConfig(
            name="Rigetti Ankaa-3",
            provider="braket_rigetti",
            max_qubits=82,
            region="us-west-1",
            priority=1,
            availability_windows_sgt=[(8, 15), (17, 24), (0, 3), (5, 8)],
        ),
        "iqm_emerald": QPUConfig(
            name="IQM Emerald",
            provider="braket_iqm_emerald",
            max_qubits=54,
            region="eu-north-1",
            priority=2,
            availability_windows_sgt=[(8, 0, 11, 30), (15, 0, 0, 30), (4, 0, 8, 0)],
            weekdays_only=True,
        ),
        "iqm_garnet": QPUConfig(
            name="IQM Garnet",
            provider="braket_iqm_garnet",
            max_qubits=20,
            region="eu-north-1",
            priority=3,
            availability_windows_sgt=[(8, 0, 9, 30), (11, 15, 23, 30), (1, 15, 8, 0)],
            weekdays_only=True,
        ),
        "amazon_sv1": QPUConfig(
            name="Amazon SV1",
            provider="braket_sim_sv1",
            max_qubits=34,
            region="us-west-1",
            priority=8,
            availability_windows_sgt=None,
        ),
        "ibm_quantum": QPUConfig(
            name="IBM Quantum",
            provider="ibm",
            max_qubits=156,
            region="global",
            priority=4,
            availability_windows_sgt=None,
        ),
        "local_qiskit": QPUConfig(
            name="Local Qiskit Statevector Simulator",
            provider="local_qiskit",
            max_qubits=28,
            region="local",
            priority=20,
            availability_windows_sgt=None,
        ),
    }


def _time_in_windows(
    now_dt: datetime,
    windows: list[tuple[int, int] | tuple[int, int, int, int]],
) -> bool:
    now_minutes = now_dt.hour * 60 + now_dt.minute
    for window in windows:
        if len(window) == 2:
            start_h, end_h = window
            start_m = start_h * 60
            end_m = end_h * 60
        else:
            start_h, start_minute, end_h, end_minute = window
            start_m = start_h * 60 + start_minute
            end_m = end_h * 60 + end_minute

        if start_m <= end_m:
            if start_m <= now_minutes < end_m:
                return True
        elif now_minutes >= start_m or now_minutes < end_m:
            return True
    return False


def _remaining_minutes_in_windows(
    now_dt: datetime,
    windows: list[tuple[int, int] | tuple[int, int, int, int]],
) -> float:
    now_minutes = now_dt.hour * 60 + now_dt.minute
    for window in windows:
        if len(window) == 2:
            start_h, end_h = window
            start_m = start_h * 60
            end_m = end_h * 60
        else:
            start_h, start_minute, end_h, end_minute = window
            start_m = start_h * 60 + start_minute
            end_m = end_h * 60 + end_minute

        if start_m <= end_m:
            if start_m <= now_minutes < end_m:
                return float(end_m - now_minutes)
        else:
            if now_minutes >= start_m:
                return float((24 * 60 - now_minutes) + end_m)
            if now_minutes < end_m:
                return float(end_m - now_minutes)
    return 0.0


def _get_ibm_usage_remaining_seconds(service: Any) -> Optional[float]:
    if not hasattr(service, "usage"):
        return None
    try:
        usage = service.usage() if callable(service.usage) else service.usage
    except Exception:
        return None
    if not isinstance(usage, dict):
        return None
    remaining = usage.get("usage_remaining_seconds")
    try:
        return float(remaining)
    except Exception:
        return None


def normalize_counts(
    raw_counts: dict[Any, Any],
    num_qubits: int,
    reverse_bits: bool,
) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for key, value in raw_counts.items():
        try:
            count = int(value)
        except Exception:
            continue
        if count <= 0:
            continue

        if isinstance(key, int):
            bits = format(key, f"0{num_qubits}b")
        else:
            bits = str(key).replace(" ", "")
            if bits.startswith("0b"):
                bits = bits[2:]
            if bits.isdigit() is False:
                continue
            if len(bits) < num_qubits:
                bits = bits.zfill(num_qubits)
            elif len(bits) > num_qubits:
                bits = bits[-num_qubits:]

        if reverse_bits:
            bits = bits[::-1]
        normalized[bits] = normalized.get(bits, 0) + count
    return normalized


class QuantumHardwareManager:
    def __init__(
        self,
        aws_profile: str | None = None,
        ibm_token: str | None = None,
        ibm_instance: str | None = None,
        ibm_credentials_json: str | None = None,
        use_aws: bool = True,
        use_ibm: bool = True,
        allow_simulators: bool = True,
        enabled_qpu_ids: Optional[set[str]] = None,
        qiskit_optimization_level: int = 3,
    ) -> None:
        self.aws_profile = aws_profile
        self.ibm_token = ibm_token
        self.ibm_instance = ibm_instance
        self.ibm_credentials_json = (
            Path(ibm_credentials_json).expanduser().resolve()
            if ibm_credentials_json
            else None
        )
        self._ibm_credential_pool: list[dict[str, str]] = []
        self._ibm_credential_file_mtime_ns: int | None = None
        self._ibm_credential_next_index = 0
        self._ibm_active_credential_key: str | None = None
        self._ibm_credential_missing_warned = False
        self._ibm_depleted_credential_keys: set[str] = set()  # permanently exhausted credentials
        if self.ibm_token and self.ibm_instance:
            self._ibm_active_credential_key = (
                f"{self.ibm_token.strip()}||{self.ibm_instance.strip()}"
            )
        self.use_aws = use_aws
        self.use_ibm = use_ibm
        self.allow_simulators = allow_simulators
        self.enabled_qpu_ids = enabled_qpu_ids
        try:
            level = int(qiskit_optimization_level)
        except Exception:
            level = 3
        self.qiskit_optimization_level = max(0, min(3, level))

        self.qpus = _default_qpus()
        self.devices: dict[str, Any] = {}
        self.sessions: dict[str, Any] = {}
        self.ibm_backends: list[dict[str, Any]] = []
        self.ibm_backends_preferred: list[dict[str, Any]] = []
        self._ibm_rr_index = 0

        self._lock = threading.Lock()
        self._qpu_locks: dict[str, threading.Lock] = {
            qpu_id: threading.Lock() for qpu_id in self.qpus.keys()
        }
        self._transpile_cache: dict[tuple[str, str], Any] = {}

        self.last_credential_refresh = time.time()
        self.credential_refresh_interval = 2700.0
        self._ibm_usage_last_check = 0.0
        self._ibm_usage_check_interval = 30.0
        self._ibm_last_reconnect_attempt = 0.0
        self._ibm_reconnect_interval = 120.0
        if self.ibm_credentials_json is not None:
            # Rotate quickly when using a live-updated credential file.
            self._ibm_reconnect_interval = 10.0
        self._ibm_last_backend_refresh = 0.0
        self.ibm_backend_refresh_interval = 300.0
        self.min_window_remaining_minutes = MIN_WINDOW_REMAINING_MINUTES
        self.min_aws_window_remaining_minutes = MIN_AWS_WINDOW_REMAINING_MINUTES
        self.ibm_min_runtime_seconds = IBM_MIN_RUNTIME_SECONDS
        self.job_status_log_interval = 30.0
        LOGGER.debug(
            "QuantumHardwareManager initialized | use_aws=%s use_ibm=%s allow_simulators=%s aws_profile=%s enabled_qpus=%s qiskit_optimization_level=%s ibm_credentials_json=%s",
            self.use_aws,
            self.use_ibm,
            self.allow_simulators,
            self.aws_profile,
            sorted(self.enabled_qpu_ids) if self.enabled_qpu_ids is not None else "all",
            self.qiskit_optimization_level,
            str(self.ibm_credentials_json) if self.ibm_credentials_json is not None else "(none)",
        )

    @staticmethod
    def _credential_key(token: str, instance: str) -> str:
        return f"{token.strip()}||{instance.strip()}"

    @staticmethod
    def _mask_token(token: str) -> str:
        t = str(token or "").strip()
        if len(t) <= 8:
            return "***"
        return f"{t[:4]}...{t[-4:]}"

    @staticmethod
    def _extract_credential_entries(data: Any) -> list[Any]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("credentials", "accounts", "ibm_credentials", "items"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
            if any(k in data for k in ("token", "api_key", "apikey")):
                return [data]
        return []

    @staticmethod
    def _parse_credential_entry(entry: Any) -> Optional[dict[str, str]]:
        if not isinstance(entry, dict):
            return None
        enabled = entry.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() not in {"0", "false", "no"}
        if enabled is False:
            return None

        token = (
            entry.get("token")
            or entry.get("api_key")
            or entry.get("apikey")
            or entry.get("ibm_token")
        )
        instance = (
            entry.get("instance")
            or entry.get("crn")
            or entry.get("ibm_instance")
        )
        if token is None or instance is None:
            return None
        token_s = str(token).strip()
        instance_s = str(instance).strip()
        if token_s == "" or instance_s == "":
            return None
        label = str(entry.get("label") or entry.get("name") or "").strip()
        return {
            "token": token_s,
            "instance": instance_s,
            "label": label,
        }

    def _load_ibm_credential_pool(self, force: bool = False) -> bool:
        path = self.ibm_credentials_json
        if path is None:
            return False
        try:
            stat = path.stat()
        except FileNotFoundError:
            if not self._ibm_credential_missing_warned:
                LOGGER.warning(
                    "[ibm_quantum] Credential JSON not found: %s",
                    path,
                )
                self._ibm_credential_missing_warned = True
            return False
        except Exception as exc:
            LOGGER.warning(
                "[ibm_quantum] Could not stat credential JSON %s: %s",
                path,
                str(exc)[:200],
            )
            return False
        self._ibm_credential_missing_warned = False

        if (
            not force
            and self._ibm_credential_file_mtime_ns is not None
            and stat.st_mtime_ns == self._ibm_credential_file_mtime_ns
        ):
            return False

        try:
            raw = path.read_text(encoding="utf-8")
        except Exception as exc:
            LOGGER.warning(
                "[ibm_quantum] Could not read credential JSON %s: %s",
                path,
                str(exc)[:200],
            )
            return False

        try:
            payload = json.loads(raw)
        except Exception as exc:
            LOGGER.warning(
                "[ibm_quantum] Credential JSON parse failed (%s). Keeping previous credential pool.",
                str(exc)[:200],
            )
            return False

        entries = self._extract_credential_entries(payload)
        parsed: list[dict[str, str]] = []
        seen: set[str] = set()
        for entry in entries:
            cred = self._parse_credential_entry(entry)
            if cred is None:
                continue
            key = self._credential_key(cred["token"], cred["instance"])
            if key in seen:
                continue
            # Skip credentials that were permanently depleted during this session.
            if key in self._ibm_depleted_credential_keys:
                LOGGER.debug(
                    "[ibm_quantum] Skipping depleted credential (label=%s) during pool reload.",
                    cred.get("label", "?"),
                )
                continue
            seen.add(key)
            parsed.append(cred)

        self._ibm_credential_pool = parsed
        self._ibm_credential_file_mtime_ns = stat.st_mtime_ns
        if self._ibm_active_credential_key is not None and parsed:
            for idx, cred in enumerate(parsed):
                key = self._credential_key(cred["token"], cred["instance"])
                if key == self._ibm_active_credential_key:
                    self._ibm_credential_next_index = (idx + 1) % len(parsed)
                    break
            else:
                self._ibm_credential_next_index = 0
        else:
            self._ibm_credential_next_index = 0

        LOGGER.info(
            "[ibm_quantum] Credential pool reloaded from %s (%d usable entries, %d depleted)",
            path,
            len(parsed),
            len(self._ibm_depleted_credential_keys),
        )
        return True

    def _persist_ibm_account(self, token: str, instance: str) -> None:
        if not HAS_IBM_RUNTIME:
            return
        save_kwargs = {
            "token": token,
            "instance": instance,
            "set_as_default": True,
            "overwrite": True,
        }
        for channel in ("ibm_cloud", "ibm_quantum"):
            try:
                QiskitRuntimeService.save_account(channel=channel, **save_kwargs)
                LOGGER.info(
                    "[ibm_quantum] Saved IBM account credentials via channel=%s",
                    channel,
                )
                return
            except TypeError:
                # Older qiskit-ibm-runtime versions may not accept channel in save_account.
                try:
                    QiskitRuntimeService.save_account(**save_kwargs)
                    LOGGER.info("[ibm_quantum] Saved IBM account credentials")
                    return
                except Exception:
                    continue
            except Exception:
                continue

    def _set_ibm_credential(
        self,
        *,
        token: str,
        instance: str,
        persist_account: bool,
        reason: str,
    ) -> None:
        self.ibm_token = token
        self.ibm_instance = instance
        self._ibm_active_credential_key = self._credential_key(token, instance)
        LOGGER.info(
            "[ibm_quantum] Switched credentials (%s) token=%s instance=%s",
            reason,
            self._mask_token(token),
            "set",
        )
        if persist_account:
            try:
                self._persist_ibm_account(token, instance)
            except Exception as exc:
                LOGGER.warning(
                    "[ibm_quantum] Failed to persist IBM account credentials: %s",
                    str(exc)[:200],
                )

    def _rotate_ibm_credential_from_pool(
        self,
        *,
        force_reload: bool,
        allow_current: bool,
        reason: str,
    ) -> bool:
        self._load_ibm_credential_pool(force=force_reload)
        if not self._ibm_credential_pool:
            return False

        attempts = len(self._ibm_credential_pool)
        for _ in range(attempts):
            idx = self._ibm_credential_next_index % len(self._ibm_credential_pool)
            self._ibm_credential_next_index = (idx + 1) % len(self._ibm_credential_pool)
            cred = self._ibm_credential_pool[idx]
            key = self._credential_key(cred["token"], cred["instance"])
            if not allow_current and key == self._ibm_active_credential_key and len(self._ibm_credential_pool) > 1:
                continue
            self._set_ibm_credential(
                token=cred["token"],
                instance=cred["instance"],
                persist_account=True,
                reason=reason,
            )
            return True
        return False

    def _is_enabled(self, qpu_id: str) -> bool:
        if self.enabled_qpu_ids is None:
            return True
        return qpu_id in self.enabled_qpu_ids

    def _mark_disabled(self, qpu_id: str) -> None:
        self.qpus[qpu_id].is_available = False
        self.qpus[qpu_id].last_error = "Disabled by --only-qpu filter"
        LOGGER.debug("QPU disabled by filter | qpu_id=%s", qpu_id)

    @staticmethod
    def _is_simulator_provider(provider: str) -> bool:
        return provider.startswith("braket_sim_") or provider.startswith("local_")

    def is_qpu_in_time_window(self, qpu: QPUConfig) -> bool:
        now_sgt = datetime.now(SGT)
        if qpu.availability_windows_sgt is None:
            return True
        if qpu.weekdays_only and now_sgt.weekday() >= 5:
            return False
        return _time_in_windows(now_sgt, qpu.availability_windows_sgt)

    def _remaining_window_minutes_sgt(self, qpu: QPUConfig) -> Optional[float]:
        if qpu.availability_windows_sgt is None:
            return None
        now_sgt = datetime.now(SGT)
        return _remaining_minutes_in_windows(now_sgt, qpu.availability_windows_sgt)

    def _has_sufficient_window(self, qpu: QPUConfig) -> bool:
        remaining = self._remaining_window_minutes_sgt(qpu)
        if remaining is None:
            return True
        required = self.min_window_remaining_minutes
        if qpu.provider.startswith("braket_") and not self._is_simulator_provider(qpu.provider):
            required = max(required, self.min_aws_window_remaining_minutes)
            # Adaptive guard: if this backend is slow, require extra runway.
            if qpu.successful_jobs > 0:
                avg_job_minutes = (qpu.total_time_sec / max(1, qpu.successful_jobs)) / 60.0
                adaptive_required = min(180.0, (2.0 * avg_job_minutes) + 5.0)
                required = max(required, adaptive_required)
        return remaining >= required

    @staticmethod
    def _extract_queue_position(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            for key in (
                "queuePosition",
                "queue_position",
                "positionInQueue",
                "position_in_queue",
                "position",
            ):
                if key in value and value[key] is not None:
                    return value[key]
            for key in ("queueInfo", "queue_info"):
                if key in value:
                    nested = QuantumHardwareManager._extract_queue_position(value[key])
                    if nested is not None:
                        return nested
            return None
        for attr in (
            "queuePosition",
            "queue_position",
            "positionInQueue",
            "position_in_queue",
            "position",
        ):
            if hasattr(value, attr):
                try:
                    candidate = getattr(value, attr)
                    if callable(candidate):
                        candidate = candidate()
                    if candidate is not None:
                        return candidate
                except Exception:
                    continue
        return None

    def _get_runtime_handle_status(self, runtime_handle: Any) -> str:
        for attr in ("status", "state"):
            if not hasattr(runtime_handle, attr):
                continue
            try:
                value = getattr(runtime_handle, attr)
                if callable(value):
                    value = value()
                if value is not None:
                    return str(value).upper()
            except Exception:
                continue
        return "UNKNOWN"

    def _get_runtime_handle_queue_position(self, runtime_handle: Any) -> Any:
        for attr in ("queue_position", "queuePosition", "queue_info", "queueInfo"):
            if not hasattr(runtime_handle, attr):
                continue
            try:
                value = getattr(runtime_handle, attr)
                if callable(value):
                    value = value()
                extracted = self._extract_queue_position(value)
                if extracted is not None:
                    return extracted
            except Exception:
                continue
        if hasattr(runtime_handle, "metadata"):
            try:
                metadata = runtime_handle.metadata()
                extracted = self._extract_queue_position(metadata)
                if extracted is not None:
                    return extracted
            except Exception:
                pass
        return None

    def _wait_for_terminal_status(
        self,
        *,
        runtime_handle: Any,
        label: str,
        timeout_sec: float | None,
        terminal_states: set[str],
        success_states: set[str],
        min_log_interval_sec: float | None = None,
    ) -> str:
        start = time.time()
        last_log = 0.0
        sleep_sec = max(1.0, min(5.0, float(self.job_status_log_interval)))
        effective_log_interval = float(self.job_status_log_interval)
        if min_log_interval_sec is not None:
            effective_log_interval = max(effective_log_interval, float(min_log_interval_sec))
        while True:
            state = self._get_runtime_handle_status(runtime_handle)
            queue_position = self._get_runtime_handle_queue_position(runtime_handle)
            now = time.time()
            if (now - last_log) >= effective_log_interval or state in terminal_states:
                LOGGER.info(
                    "%s status | state=%s queue_position=%s elapsed_sec=%.1f",
                    label,
                    state,
                    "n/a" if queue_position is None else queue_position,
                    now - start,
                )
                last_log = now
            if state in terminal_states:
                if state not in success_states:
                    raise RuntimeError(f"{label} ended in terminal state '{state}'")
                return state
            if timeout_sec is not None and (now - start) > float(timeout_sec):
                raise TimeoutError(f"{label} timed out after {timeout_sec}s")
            time.sleep(sleep_sec)

    def initialize(self) -> dict[str, bool]:
        """Initialize all available QPUs. Returns dict of QPU name -> success."""
        results: dict[str, bool] = {}
        LOGGER.info(
            "Initializing backends | use_aws=%s use_ibm=%s allow_simulators=%s",
            self.use_aws,
            self.use_ibm,
            self.allow_simulators,
        )

        # ------- AWS Braket QPUs -------
        if self.use_aws and not HAS_BRAKET:
            LOGGER.warning(
                "[AWS] Braket SDK not installed - all AWS backends will be unavailable. "
                "Install with: pip install amazon-braket-sdk"
            )

        if self.use_aws and HAS_BRAKET:
            for qpu_id in _BRAKET_QPU_IDS:
                if qpu_id not in self.qpus:
                    continue
                if not self._is_enabled(qpu_id):
                    self._mark_disabled(qpu_id)
                    results[qpu_id] = False
                    continue
                if not self.allow_simulators and self._is_simulator_provider(
                    self.qpus[qpu_id].provider
                ):
                    self.qpus[qpu_id].is_available = False
                    self.qpus[qpu_id].last_error = "Simulators disabled by configuration"
                    LOGGER.info("[%s] Skipped (simulators disabled)", qpu_id)
                    results[qpu_id] = False
                    continue
                success = self._init_braket_device(qpu_id)
                results[qpu_id] = success
        else:
            for qpu_id in _BRAKET_QPU_IDS:
                if qpu_id not in self.qpus:
                    continue
                if not self._is_enabled(qpu_id):
                    self._mark_disabled(qpu_id)
                else:
                    reason = "Braket SDK not installed" if not HAS_BRAKET else "AWS disabled"
                    self.qpus[qpu_id].is_available = False
                    self.qpus[qpu_id].last_error = reason
                    LOGGER.info("[%s] Unavailable - %s", qpu_id, reason)
                results[qpu_id] = False

        # ------- IBM Quantum -------
        if not self._is_enabled("ibm_quantum"):
            self._mark_disabled("ibm_quantum")
            results["ibm_quantum"] = False
        elif self.use_ibm and HAS_IBM_RUNTIME:
            success = self._init_ibm_device()
            results["ibm_quantum"] = success
        else:
            reason = "IBM Runtime SDK not installed" if not HAS_IBM_RUNTIME else "IBM disabled"
            self.qpus["ibm_quantum"].is_available = False
            self.qpus["ibm_quantum"].last_error = reason
            LOGGER.info("[ibm_quantum] Unavailable - %s", reason)
            results["ibm_quantum"] = False

        # ------- Local Qiskit Simulator -------
        if not self._is_enabled("local_qiskit"):
            self._mark_disabled("local_qiskit")
            results["local_qiskit"] = False
        elif self.allow_simulators and HAS_QISKIT:
            self.qpus["local_qiskit"].is_available = True
            self.qpus["local_qiskit"].last_error = ""
            LOGGER.info("[local_qiskit] Ready (max %d qubits)", self.qpus["local_qiskit"].max_qubits)
            results["local_qiskit"] = True
        else:
            self.qpus["local_qiskit"].is_available = False
            self.qpus["local_qiskit"].last_error = "Local simulator unavailable or disabled"
            LOGGER.info("[local_qiskit] Unavailable")
            results["local_qiskit"] = False

        return results

    def _init_braket_device(self, qpu_id: str) -> bool:
        qpu = self.qpus[qpu_id]
        LOGGER.info(
            "[%s] Connecting to %s (region=%s, profile=%s)...",
            qpu_id, qpu.name, qpu.region, self.aws_profile or "default",
        )
        try:
            if self.aws_profile:
                boto_session = boto3.Session(
                    profile_name=self.aws_profile,
                    region_name=qpu.region,
                )
            else:
                boto_session = boto3.Session(region_name=qpu.region)
            aws_session = AwsSession(boto_session=boto_session)

            if qpu.provider == "braket_rigetti":
                device = AwsDevice(Devices.Rigetti.Ankaa3, aws_session=aws_session)
            elif qpu.provider == "braket_iqm_emerald":
                device = AwsDevice(Devices.IQM.Emerald, aws_session=aws_session)
            elif qpu.provider == "braket_iqm_garnet":
                device = AwsDevice(Devices.IQM.Garnet, aws_session=aws_session)
            elif qpu.provider == "braket_sim_sv1":
                device = AwsDevice(
                    "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                    aws_session=aws_session,
                )
            else:
                raise ValueError(f"Unknown Braket provider: {qpu.provider}")

            status_str = str(device.status).upper()
            if status_str != "ONLINE":
                qpu.is_available = False
                qpu.last_error = f"Device status: {status_str}"
                LOGGER.warning("[%s] OFFLINE (status=%s)", qpu_id, status_str)
                return False

            try:
                props = device.properties
                if hasattr(props, "paradigm") and hasattr(props.paradigm, "qubitCount"):
                    qpu.max_qubits = int(props.paradigm.qubitCount)
            except Exception:
                pass

            self.devices[qpu_id] = device
            self.sessions[qpu_id] = {"aws_session": aws_session}
            qpu.is_available = True
            qpu.last_error = ""
            remaining_window = self._remaining_window_minutes_sgt(qpu)
            window_str = f"{remaining_window:.0f}m remaining" if remaining_window is not None else "always-on"
            LOGGER.info(
                "[%s] Ready - %s (%d qubits, window: %s)",
                qpu_id, qpu.name, qpu.max_qubits, window_str,
            )
            return True
        except Exception as exc:
            qpu.is_available = False
            qpu.last_error = str(exc)[:120]
            LOGGER.warning("[%s] Init failed - %s", qpu_id, qpu.last_error)
            return False

    def _connect_ibm_service(
        self,
        *,
        token: str | None,
        instance: str | None,
        allow_saved_fallback: bool,
    ) -> Any | None:
        service = None

        if token and instance:
            for channel in ("ibm_cloud", "ibm_quantum"):
                try:
                    service = QiskitRuntimeService(
                        channel=channel,
                        token=token,
                        instance=instance,
                    )
                    if service.active_account() is not None:
                        LOGGER.info("[ibm_quantum] Connected via %s channel", channel)
                        return service
                except Exception:
                    service = None

        if allow_saved_fallback:
            try:
                service = QiskitRuntimeService()
                if service.active_account() is not None:
                    LOGGER.info("[ibm_quantum] Connected via saved credentials")
                    return service
                service = None
            except Exception:
                service = None

        if allow_saved_fallback and token:
            try:
                service = QiskitRuntimeService(
                    channel="ibm_quantum",
                    token=token,
                )
                if service.active_account() is not None:
                    LOGGER.info("[ibm_quantum] Connected via token")
                    return service
            except Exception:
                service = None

        return None

    def _configure_ibm_session(
        self,
        *,
        service: Any,
        qpu: QPUConfig,
    ) -> tuple[bool, Optional[float]]:
        # Check runtime budget.
        remaining = _get_ibm_usage_remaining_seconds(service)
        threshold = float(self.ibm_min_runtime_seconds)
        if remaining is not None:
            LOGGER.info(
                "[ibm_quantum] Runtime budget: %.0fs remaining (threshold: %.0fs)",
                remaining,
                threshold,
            )
            if remaining < threshold:
                qpu.is_available = False
                qpu.last_error = f"Runtime budget low: {remaining:.0f}s < {threshold:.0f}s"
                LOGGER.warning("[ibm_quantum] %s", qpu.last_error)
                return False, remaining
        else:
            LOGGER.info("[ibm_quantum] Runtime budget: unavailable from API")

        # Discover backends.
        self.ibm_backends = []
        try:
            backends = service.backends(operational=True, simulator=False)
        except Exception:
            backends = []

        for backend in backends:
            try:
                pending_jobs = None
                try:
                    status = backend.status()
                    pending_jobs = getattr(status, "pending_jobs", None)
                except Exception:
                    pending_jobs = None
                self.ibm_backends.append(
                    {
                        "backend": backend,
                        "name": backend.name,
                        "num_qubits": int(backend.num_qubits),
                        "pending_jobs": pending_jobs,
                    }
                )
            except Exception:
                continue

        self.ibm_backends.sort(key=lambda item: item["num_qubits"], reverse=True)

        candidates = [
            item for item in self.ibm_backends if item.get("pending_jobs") is not None
        ]
        if candidates:
            candidates.sort(
                key=lambda item: (
                    int(item.get("pending_jobs", 10**9)),
                    -int(item.get("num_qubits", 0)),
                )
            )
            self.ibm_backends_preferred = candidates[:2]
        else:
            self.ibm_backends_preferred = self.ibm_backends[:2]

        if self.ibm_backends_preferred:
            qpu.backend_name = self.ibm_backends_preferred[0]["name"]
        # max_qubits should reflect the largest backend, not just the preferred one.
        if self.ibm_backends:
            qpu.max_qubits = max(int(b["num_qubits"]) for b in self.ibm_backends)

        self.sessions["ibm_quantum"] = service
        qpu.is_available = len(self.ibm_backends) > 0
        qpu.last_error = "" if qpu.is_available else "No operational IBM backend found"
        self._ibm_last_backend_refresh = time.time()

        if self.ibm_backends:
            backend_strs = [
                f"{b['name']}({b['num_qubits']}q, pending={b.get('pending_jobs', 'n/a')})"
                for b in self.ibm_backends
            ]
            LOGGER.info(
                "[ibm_quantum] %d backends found: %s",
                len(self.ibm_backends),
                ", ".join(backend_strs),
            )
            if self.ibm_backends_preferred:
                pref_strs = [b["name"] for b in self.ibm_backends_preferred]
                LOGGER.info("[ibm_quantum] Preferred (least busy): %s", ", ".join(pref_strs))
        else:
            LOGGER.warning("[ibm_quantum] No operational backends found")

        return qpu.is_available, remaining

    def _remove_depleted_credential(self, cred: dict[str, str]) -> None:
        """Permanently remove a depleted credential from the pool."""
        key = self._credential_key(cred["token"], cred["instance"])
        self._ibm_depleted_credential_keys.add(key)
        try:
            self._ibm_credential_pool.remove(cred)
        except ValueError:
            pass
        # Fix the next_index after removal.
        if self._ibm_credential_pool:
            self._ibm_credential_next_index = (
                self._ibm_credential_next_index % len(self._ibm_credential_pool)
            )
        else:
            self._ibm_credential_next_index = 0
        LOGGER.info(
            "[ibm_quantum] Credential (label=%s) permanently removed — depleted. "
            "%d credentials remaining in pool.",
            cred.get("label", "?"),
            len(self._ibm_credential_pool),
        )

    def _init_ibm_device_from_pool(self) -> bool:
        qpu = self.qpus["ibm_quantum"]
        if not self._ibm_credential_pool:
            return False

        # At startup, always begin from the first credential in the JSON pool.
        if self.sessions.get("ibm_quantum") is None:
            self._ibm_credential_next_index = 0

        threshold = float(self.ibm_min_runtime_seconds)

        # --- Sequential drain strategy: try the current credential first.
        #     If it's depleted, permanently remove it and advance to the next
        #     one in line. Repeat until we find a usable one or exhaust all. ---
        while self._ibm_credential_pool:
            idx = self._ibm_credential_next_index % len(self._ibm_credential_pool)
            cred = self._ibm_credential_pool[idx]
            label = cred.get("label", "?")
            pool_size = len(self._ibm_credential_pool)

            LOGGER.info(
                "[ibm_quantum] Trying credential %d/%d (label=%s)...",
                idx + 1, pool_size, label,
            )

            self._set_ibm_credential(
                token=cred["token"],
                instance=cred["instance"],
                persist_account=True,
                reason=f"sequential_drain_{idx + 1}_of_{pool_size}",
            )

            service = self._connect_ibm_service(
                token=cred["token"],
                instance=cred["instance"],
                allow_saved_fallback=False,
            )
            if service is None:
                LOGGER.warning(
                    "[ibm_quantum] Credential (label=%s) could not connect. Removing.",
                    label,
                )
                self._remove_depleted_credential(cred)
                continue

            remaining = _get_ibm_usage_remaining_seconds(service)
            if remaining is not None:
                LOGGER.info(
                    "[ibm_quantum] Credential (label=%s) has %.0fs remaining (threshold: %.0fs).",
                    label, remaining, threshold,
                )
                if remaining < threshold:
                    LOGGER.info(
                        "[ibm_quantum] Credential (label=%s) depleted (%.0fs < %.0fs). Removing and advancing.",
                        label, remaining, threshold,
                    )
                    self._remove_depleted_credential(cred)
                    continue
            else:
                LOGGER.info(
                    "[ibm_quantum] Credential (label=%s) budget unavailable from API, treating as usable.",
                    label,
                )

            # This credential is usable — configure the session.
            success, _ = self._configure_ibm_session(service=service, qpu=qpu)
            if success:
                return True
            else:
                LOGGER.warning(
                    "[ibm_quantum] Credential (label=%s) session configuration failed. Removing.",
                    label,
                )
                self._remove_depleted_credential(cred)
                continue

        qpu.is_available = False
        qpu.last_error = (
            f"All IBM credentials exhausted or depleted (threshold >= {threshold:.0f}s). "
            f"{len(self._ibm_depleted_credential_keys)} total depleted."
        )
        LOGGER.warning("[ibm_quantum] %s", qpu.last_error)
        return False

    def _init_ibm_device(self) -> bool:
        qpu = self.qpus["ibm_quantum"]
        if self.ibm_credentials_json is not None:
            self._load_ibm_credential_pool(force=False)
            if self._ibm_credential_pool:
                return self._init_ibm_device_from_pool()

        LOGGER.info(
            "[ibm_quantum] Connecting (token=%s, instance=%s)...",
            "yes" if self.ibm_token else "saved",
            "yes" if self.ibm_instance else "default",
        )
        try:
            service = self._connect_ibm_service(
                token=self.ibm_token,
                instance=self.ibm_instance,
                allow_saved_fallback=True,
            )
            if service is None:
                qpu.is_available = False
                qpu.last_error = "Could not connect to IBM Quantum"
                LOGGER.warning("[ibm_quantum] All connection methods failed")
                return False
            success, _ = self._configure_ibm_session(service=service, qpu=qpu)
            return success
        except Exception as exc:
            qpu.is_available = False
            qpu.last_error = str(exc)[:120]
            LOGGER.warning("[ibm_quantum] Init failed - %s", qpu.last_error)
            return False

    def _refresh_ibm_pending(self, backend_infos: list[dict[str, Any]]) -> None:
        for info in backend_infos:
            try:
                status = info["backend"].status()
                pending = getattr(status, "pending_jobs", None)
                if pending is not None:
                    info["pending_jobs"] = int(pending)
            except Exception:
                continue

    def _select_ibm_backend_info(self, num_qubits: int) -> Optional[dict[str, Any]]:
        candidates = [
            item
            for item in self.ibm_backends_preferred
            if int(item.get("num_qubits", 0)) >= num_qubits
        ]
        if not candidates:
            candidates = [
                item
                for item in self.ibm_backends
                if int(item.get("num_qubits", 0)) >= num_qubits
            ]
        if not candidates:
            LOGGER.warning("No IBM backend can satisfy qubit requirement | required_qubits=%s", num_qubits)
            return None

        self._refresh_ibm_pending(candidates)

        def _key(info: dict[str, Any]) -> tuple[int, int]:
            pending = info.get("pending_jobs")
            if pending is None:
                pending = 10**9
            return int(pending), int(info.get("num_qubits", 0))

        ordered = sorted(candidates, key=_key)
        best_pending = ordered[0].get("pending_jobs")
        tied = [info for info in ordered if info.get("pending_jobs") == best_pending]
        if len(tied) <= 1:
            LOGGER.info(
                "Selected IBM backend | selection=least_busy backend=%s qubits=%s pending_jobs=%s",
                ordered[0].get("name"),
                ordered[0].get("num_qubits"),
                ordered[0].get("pending_jobs"),
            )
            return ordered[0]

        idx = self._ibm_rr_index % len(tied)
        self._ibm_rr_index += 1
        selected = tied[idx]
        LOGGER.info(
            "Selected IBM backend | selection=least_busy_tie_round_robin backend=%s qubits=%s pending_jobs=%s",
            selected.get("name"),
            selected.get("num_qubits"),
            selected.get("pending_jobs"),
        )
        return selected

    def _refresh_ibm_usage_if_needed(self, *, force: bool = False) -> None:
        if not self._is_enabled("ibm_quantum"):
            return
        now = time.time()
        if not force and (now - self._ibm_usage_last_check < self._ibm_usage_check_interval):
            return
        self._ibm_usage_last_check = now

        qpu = self.qpus["ibm_quantum"]
        service = self.sessions.get("ibm_quantum")
        if service is None:
            return

        remaining = _get_ibm_usage_remaining_seconds(service)
        if remaining is None:
            LOGGER.debug("[ibm_quantum] Usage check: remaining runtime unavailable from API")
            return

        threshold = float(self.ibm_min_runtime_seconds)
        LOGGER.info(
            "[ibm_quantum] Usage check: %.0fs remaining (threshold: %.0fs)",
            remaining, threshold,
        )
        if remaining < threshold:
            qpu.is_available = False
            qpu.last_error = f"Runtime budget low: {remaining:.0f}s < {threshold:.0f}s"
            LOGGER.warning("[ibm_quantum] Disabled - %s", qpu.last_error)

            # Current credential is depleted — permanently remove it and
            # advance to the next one in line.
            if self.ibm_credentials_json is not None and self._ibm_credential_pool:
                # Find and remove the currently active credential.
                active_key = self._ibm_active_credential_key
                removed = False
                for cred in list(self._ibm_credential_pool):
                    key = self._credential_key(cred["token"], cred["instance"])
                    if key == active_key:
                        self._remove_depleted_credential(cred)
                        removed = True
                        break

                if not removed:
                    LOGGER.debug(
                        "[ibm_quantum] Active credential not found in pool for removal."
                    )

                # Now try the next credential in line.
                if self._ibm_credential_pool:
                    LOGGER.info(
                        "[ibm_quantum] Advancing to next credential in line (%d remaining in pool)...",
                        len(self._ibm_credential_pool),
                    )
                    self._ibm_last_reconnect_attempt = now
                    success = self._init_ibm_device_from_pool()
                    if success:
                        qpu.is_available = True
                        qpu.last_error = ""
                        LOGGER.info("[ibm_quantum] Successfully switched to next credential.")
                    else:
                        LOGGER.warning(
                            "[ibm_quantum] All remaining credentials also exhausted."
                        )
                else:
                    LOGGER.warning(
                        "[ibm_quantum] Credential pool completely exhausted. "
                        "%d total credentials depleted.",
                        len(self._ibm_depleted_credential_keys),
                    )
            else:
                # No credential pool — try single-credential re-auth
                reconnect_ready = (now - self._ibm_last_reconnect_attempt >= self._ibm_reconnect_interval)
                if force:
                    reconnect_ready = True
                if reconnect_ready:
                    self._ibm_last_reconnect_attempt = now
                    LOGGER.info("[ibm_quantum] Attempting single-credential re-auth...")
                    try:
                        success = self._init_ibm_device()
                        if success:
                            qpu.is_available = True
                            qpu.last_error = ""
                    except Exception as e:
                        LOGGER.warning(
                            "[ibm_quantum] Single-credential re-auth failed: %s", e,
                        )
        else:
            if not qpu.is_available:
                LOGGER.info("[ibm_quantum] Runtime available again (%.0fs). Re-enabling.", remaining)
            qpu.is_available = True
            qpu.last_error = ""

    def _refresh_ibm_backends_if_needed(self, force: bool = False) -> None:
        """Refresh IBM backend inventory only when runtime budget is exhausted.

        Instead of periodic rediscovery, we only refresh when the remaining
        IBM runtime budget drops below ``ibm_min_runtime_seconds``.  This
        avoids unnecessary API calls while the current backend still has
        plenty of budget.
        """
        if not self.use_ibm or not HAS_IBM_RUNTIME:
            return
        if not self._is_enabled("ibm_quantum"):
            return

        if not force:
            # Check if budget is still healthy — skip refresh if so
            service = self.sessions.get("ibm_quantum")
            if service is not None:
                remaining = _get_ibm_usage_remaining_seconds(service)
                threshold = float(self.ibm_min_runtime_seconds)
                if remaining is not None and remaining > threshold:
                    return  # budget is healthy, no need to rediscover
                if remaining is not None:
                    LOGGER.info(
                        "IBM runtime budget low (%.0fs remaining < %.0fs threshold) "
                        "— refreshing backend inventory",
                        remaining,
                        threshold,
                    )
                    # Force an immediate usage refresh so credential rotation
                    # is attempted even if the periodic usage-check interval
                    # has not elapsed yet.
                    self._refresh_ibm_usage_if_needed(force=True)
                    if self.qpus["ibm_quantum"].is_available:
                        return

            # Also respect the minimum interval to avoid spamming on budget=0
            now = time.time()
            if (now - self._ibm_last_backend_refresh) < 60.0:
                return

        LOGGER.debug(
            "Refreshing IBM backend inventory | force=%s",
            force,
        )
        self._init_ibm_device()

    def _refresh_braket_backends(
        self,
        include_unavailable: bool = True,
        in_window_only: bool = False,
        reason: str = "",
    ) -> None:
        if not HAS_BRAKET or not self.use_aws:
            return
        LOGGER.debug(
            "Refreshing AWS Braket backends | reason=%s",
            reason or "periodic",
        )
        for qpu_id in _BRAKET_QPU_IDS:
            if qpu_id not in self.qpus:
                continue
            if not self._is_enabled(qpu_id):
                continue
            qpu = self.qpus[qpu_id]
            if in_window_only and not self.is_qpu_in_time_window(qpu):
                continue
            if not include_unavailable and not qpu.is_available:
                continue
            self._init_braket_device(qpu_id)

    def refresh_credentials_if_needed(self) -> None:
        if self.ibm_credentials_json is not None:
            self._load_ibm_credential_pool(force=False)
        self._refresh_ibm_usage_if_needed()
        self._refresh_ibm_backends_if_needed(force=False)
        now = time.time()
        if now - self.last_credential_refresh > self.credential_refresh_interval:
            LOGGER.debug(
                "Refreshing cached credentials/backends | elapsed_sec=%.1f interval_sec=%.1f",
                now - self.last_credential_refresh,
                self.credential_refresh_interval,
            )
            self._refresh_braket_backends(include_unavailable=True, in_window_only=False)
            self.last_credential_refresh = now

    def refresh_qpu_windows(self, num_qubits: int = 0) -> list[str]:
        """Re-check availability windows for all QPUs.

        Marks QPUs available when a new window opens and unavailable when
        a window closes.  Returns list of QPU IDs that are currently in
        an open window and have enough qubits for *num_qubits*.
        """
        now_changed: list[str] = []
        for qpu_id, qpu in self.qpus.items():
            if self._is_simulator_provider(qpu.provider):
                continue
            in_window = self.is_qpu_in_time_window(qpu)
            has_runway = self._has_sufficient_window(qpu)
            was_available = qpu.is_available
            window_error = qpu.last_error if "window" in (qpu.last_error or "").lower() else ""

            if in_window and has_runway and not was_available and window_error:
                # QPU was marked offline due to window expiry, but a new window
                # is now open → mark it available again.
                qpu.is_available = True
                qpu.last_error = ""
                LOGGER.info(
                    "QPU '%s' window re-opened — marking AVAILABLE", qpu_id,
                )
                now_changed.append(qpu_id)
            elif (not in_window or not has_runway) and was_available:
                # Window just closed → mark offline
                remaining = self._remaining_window_minutes_sgt(qpu)
                qpu.is_available = False
                qpu.last_error = (
                    f"Availability window closed ({remaining:.0f}m remaining)"
                    if remaining is not None
                    else "Outside availability window"
                )
                LOGGER.info(
                    "QPU '%s' window closed — marking OFFLINE: %s",
                    qpu_id, qpu.last_error,
                )

        available = [
            qpu_id for qpu_id, qpu in self.qpus.items()
            if qpu.is_available and qpu.max_qubits >= num_qubits
        ]
        return available

    def get_available_qpus_for_size(
        self,
        num_qubits: int,
        include_simulators: bool,
    ) -> list[str]:
        self.refresh_credentials_if_needed()
        self._refresh_ibm_usage_if_needed()
        hardware_candidates: list[tuple[int, str]] = []
        simulator_candidates: list[tuple[int, str]] = []
        skipped: dict[str, str] = {}
        for qpu_id, qpu in self.qpus.items():
            is_sim = self._is_simulator_provider(qpu.provider)
            if not qpu.is_available:
                skipped[qpu_id] = f"unavailable - {qpu.last_error}"
                continue
            if qpu.max_qubits < num_qubits:
                skipped[qpu_id] = f"too few qubits ({qpu.max_qubits} < {num_qubits})"
                continue
            if not is_sim:
                # Hardware QPU checks
                if not self.is_qpu_in_time_window(qpu):
                    skipped[qpu_id] = "outside availability window"
                    continue
                if not self._has_sufficient_window(qpu):
                    remaining = self._remaining_window_minutes_sgt(qpu)
                    skipped[qpu_id] = (
                        f"window closing soon ({remaining:.0f}m left)"
                        if remaining is not None
                        else "window closing soon"
                    )
                    continue
                hardware_candidates.append((qpu.priority, qpu_id))
            else:
                # Simulator - track separately for fallback
                simulator_candidates.append((qpu.priority, qpu_id))

        hardware_candidates.sort(key=lambda item: (item[0], item[1]))
        selected = [item[1] for item in hardware_candidates]

        # Simulators are fallback-only: include only if no hardware QPU passed
        if not selected and include_simulators and simulator_candidates:
            simulator_candidates.sort(key=lambda item: (item[0], item[1]))
            selected = [item[1] for item in simulator_candidates]
            LOGGER.info("No hardware QPU available - falling back to simulators: %s", selected)
        elif simulator_candidates:
            sim_names = [item[1] for item in simulator_candidates]
            skipped.update({sid: "simulator reserved as fallback" for sid in sim_names})

        if skipped:
            skip_lines = [f"  {qid}: {reason}" for qid, reason in skipped.items()]
            LOGGER.info("QPUs skipped for %d-qubit problem:\n%s", num_qubits, "\n".join(skip_lines))
        LOGGER.info("QPUs selected: %s", selected if selected else "(none)")
        return selected

    def status_snapshot(self) -> dict[str, dict[str, Any]]:
        snapshot: dict[str, dict[str, Any]] = {}
        for qpu_id, qpu in self.qpus.items():
            snapshot[qpu_id] = {
                "name": qpu.name,
                "provider": qpu.provider,
                "is_available": qpu.is_available,
                "max_qubits": qpu.max_qubits,
                "backend_name": qpu.backend_name,
                "last_error": qpu.last_error,
                "total_jobs": qpu.total_jobs,
                "successful_jobs": qpu.successful_jobs,
                "failed_jobs": qpu.failed_jobs,
                "total_time_sec": qpu.total_time_sec,
                "remaining_window_minutes_sgt": self._remaining_window_minutes_sgt(qpu),
            }
        return snapshot

    def get_calibration_snapshot(self) -> dict[str, Any]:
        """Collect device calibration data for reproducibility.

        Returns a dict keyed by qpu_id with T1/T2, gate errors, readout errors,
        coupling map, and device metadata where available.
        """
        calibration: dict[str, Any] = {}

        # IBM backends
        if HAS_IBM_RUNTIME and self.ibm_backends:
            for binfo in self.ibm_backends:
                backend = binfo.get("backend")
                name = binfo.get("name", "unknown")
                entry: dict[str, Any] = {
                    "provider": "ibm",
                    "backend_name": name,
                    "num_qubits": binfo.get("num_qubits"),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    target = getattr(backend, "target", None)
                    if target is not None:
                        # Extract coupling map
                        try:
                            from qiskit.transpiler import CouplingMap
                            cm = CouplingMap(target.build_coupling_map().get_edges()) if hasattr(target, "build_coupling_map") else None
                            if cm is not None:
                                entry["coupling_map_edges"] = len(cm.get_edges())
                        except Exception:
                            pass

                        # Extract gate errors from target
                        try:
                            ops = target.operation_names
                            gate_errors: dict[str, list[float]] = {}
                            for op_name in ops:
                                if op_name in ("measure", "delay", "reset"):
                                    continue
                                errs = []
                                for qargs in target.qargs_for_operation_name(op_name):
                                    props = target[op_name].get(qargs)
                                    if props is not None and props.error is not None:
                                        errs.append(float(props.error))
                                if errs:
                                    gate_errors[op_name] = {
                                        "mean": round(sum(errs) / len(errs), 6),
                                        "max": round(max(errs), 6),
                                        "min": round(min(errs), 6),
                                        "count": len(errs),
                                    }
                            if gate_errors:
                                entry["gate_errors"] = gate_errors
                        except Exception:
                            pass

                        # Qubit properties (T1/T2, readout)
                        try:
                            t1_values, t2_values, readout_errors = [], [], []
                            for qi in range(target.num_qubits):
                                qprops = target.qubit_properties
                                if qprops and qi < len(qprops) and qprops[qi] is not None:
                                    qp = qprops[qi]
                                    if hasattr(qp, "t1") and qp.t1 is not None:
                                        t1_values.append(float(qp.t1))
                                    if hasattr(qp, "t2") and qp.t2 is not None:
                                        t2_values.append(float(qp.t2))
                                # Readout error from measure operation
                                meas_props = target["measure"].get((qi,))
                                if meas_props is not None and meas_props.error is not None:
                                    readout_errors.append(float(meas_props.error))
                            if t1_values:
                                entry["t1_us"] = {
                                    "mean": round(sum(t1_values) / len(t1_values) * 1e6, 2),
                                    "min": round(min(t1_values) * 1e6, 2),
                                }
                            if t2_values:
                                entry["t2_us"] = {
                                    "mean": round(sum(t2_values) / len(t2_values) * 1e6, 2),
                                    "min": round(min(t2_values) * 1e6, 2),
                                }
                            if readout_errors:
                                entry["readout_error"] = {
                                    "mean": round(sum(readout_errors) / len(readout_errors), 6),
                                    "max": round(max(readout_errors), 6),
                                }
                        except Exception:
                            pass
                except Exception as exc:
                    entry["calibration_error"] = str(exc)[:200]

                calibration[f"ibm_{name}"] = entry

        # AWS Braket devices
        if HAS_BRAKET:
            for qpu_id in _BRAKET_QPU_IDS:
                if qpu_id not in self.devices:
                    continue
                device = self.devices[qpu_id]
                qpu = self.qpus.get(qpu_id)
                entry = {
                    "provider": qpu.provider if qpu else "braket",
                    "device_name": qpu.name if qpu else qpu_id,
                    "num_qubits": qpu.max_qubits if qpu else None,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    props = device.properties
                    if hasattr(props, "provider") and hasattr(props.provider, "specs"):
                        specs = props.provider.specs
                        # Rigetti / IQM provide fidelity specs
                        if isinstance(specs, dict):
                            for spec_key in ("1Q", "2Q"):
                                if spec_key in specs:
                                    fidelities = []
                                    errors = []
                                    for gate_name, gate_data in specs[spec_key].items():
                                        if isinstance(gate_data, dict):
                                            for metric, value in gate_data.items():
                                                if "fidelity" in metric.lower() and isinstance(value, (int, float)):
                                                    fidelities.append(float(value))
                                                if "error" in metric.lower() and isinstance(value, (int, float)):
                                                    errors.append(float(value))
                                    if fidelities:
                                        entry[f"{spec_key}_fidelity"] = {
                                            "mean": round(sum(fidelities) / len(fidelities), 6),
                                            "min": round(min(fidelities), 6),
                                            "count": len(fidelities),
                                        }
                        # T1/T2
                        if isinstance(specs, dict) and "1Q" in specs:
                            t1s, t2s = [], []
                            for qid, qdata in specs["1Q"].items():
                                if isinstance(qdata, dict):
                                    if "T1" in qdata and isinstance(qdata["T1"], (int, float)):
                                        t1s.append(float(qdata["T1"]))
                                    if "T2" in qdata and isinstance(qdata["T2"], (int, float)):
                                        t2s.append(float(qdata["T2"]))
                            if t1s:
                                entry["t1_us"] = {
                                    "mean": round(sum(t1s) / len(t1s) * 1e6, 2),
                                    "min": round(min(t1s) * 1e6, 2),
                                }
                            if t2s:
                                entry["t2_us"] = {
                                    "mean": round(sum(t2s) / len(t2s) * 1e6, 2),
                                    "min": round(min(t2s) * 1e6, 2),
                                }
                    # Connectivity
                    if hasattr(props, "paradigm") and hasattr(props.paradigm, "connectivity"):
                        conn = props.paradigm.connectivity
                        if hasattr(conn, "connectivityGraph"):
                            entry["connectivity_type"] = getattr(conn, "fullyConnected", False)
                except Exception as exc:
                    entry["calibration_error"] = str(exc)[:200]

                calibration[qpu_id] = entry

        return calibration

    def _run_braket_counts(
        self,
        qpu_id: str,
        circuit_template: Any,
        braket_parameters: list[Any],
        theta: list[float],
        num_qubits: int,
        shots: int,
        timeout_sec: float | None,
    ) -> tuple[dict[str, int], dict[str, Any]]:
        device = self.devices[qpu_id]
        param_map = {
            braket_parameters[i].name: float(theta[i]) for i in range(len(braket_parameters))
        }
        bound_circuit = circuit_template.make_bound_circuit(param_map)
        LOGGER.info(
            "Submitting AWS Braket task | qpu_id=%s provider=%s shots=%s qubits=%s",
            qpu_id,
            self.qpus[qpu_id].provider,
            shots,
            num_qubits,
        )
        # Capture circuit metrics
        braket_circuit_depth = None
        braket_gate_count = None
        try:
            braket_circuit_depth = int(bound_circuit.depth)
            braket_gate_count = len(bound_circuit.instructions)
        except Exception:
            pass

        submit_time_utc = datetime.now(timezone.utc).isoformat()
        task = device.run(bound_circuit, shots=int(shots))
        task_id = task.id
        LOGGER.info("AWS Braket task submitted | qpu_id=%s task_id=%s", qpu_id, task_id)

        self._wait_for_terminal_status(
            runtime_handle=task,
            label=f"AWS task {task_id}",
            timeout_sec=timeout_sec,
            terminal_states={"COMPLETED", "FAILED", "CANCELLED"},
            success_states={"COMPLETED"},
        )
        result = task.result()
        complete_time_utc = datetime.now(timezone.utc).isoformat()

        counts = normalize_counts(
            raw_counts=dict(result.measurement_counts),
            num_qubits=num_qubits,
            reverse_bits=False,
        )
        LOGGER.info(
            "AWS Braket task completed | qpu_id=%s task_id=%s unique_bitstrings=%s",
            qpu_id,
            task_id,
            len(counts),
        )
        return counts, {
            "task_id": task_id,
            "provider": "braket",
            "qpu_id": qpu_id,
            "device_name": self.qpus[qpu_id].name,
            "shots": shots,
            "submit_time_utc": submit_time_utc,
            "complete_time_utc": complete_time_utc,
            "braket_circuit_depth": braket_circuit_depth,
            "braket_gate_count": braket_gate_count,
        }

    def _extract_sampler_v2_counts_from_pub_result(
        self,
        pub_result: Any,
        num_qubits: int,
    ) -> dict[str, int]:
        raw_counts: dict[Any, Any] = {}
        if hasattr(pub_result, "data"):
            data = pub_result.data
            if hasattr(data, "meas"):
                raw_counts = data.meas.get_counts()
            elif hasattr(data, "c"):
                raw_counts = data.c.get_counts()
            else:
                for attr in dir(data):
                    if attr.startswith("_"):
                        continue
                    try:
                        maybe = getattr(data, attr)
                    except Exception:
                        continue
                    if hasattr(maybe, "get_counts"):
                        raw_counts = maybe.get_counts()
                        break
        return normalize_counts(raw_counts, num_qubits=num_qubits, reverse_bits=True)

    def _extract_sampler_v2_counts(self, result: Any, num_qubits: int) -> dict[str, int]:
        try:
            pub_result = result[0]
        except Exception:
            return {}
        return self._extract_sampler_v2_counts_from_pub_result(
            pub_result=pub_result,
            num_qubits=num_qubits,
        )

    def _get_transpiled_template(
        self,
        backend: Any,
        ansatz_id: str,
        qiskit_template: Any,
    ) -> Any:
        backend_name = getattr(backend, "name", "unknown_backend")
        key = (
            str(backend_name),
            ansatz_id,
            str(self.qiskit_optimization_level),
        )
        if key in self._transpile_cache:
            LOGGER.debug(
                "Using cached transpiled circuit | backend=%s ansatz=%s optimization_level=%s",
                backend_name,
                ansatz_id,
                self.qiskit_optimization_level,
            )
            return self._transpile_cache[key]
        LOGGER.debug(
            "Transpiling circuit for backend | backend=%s ansatz=%s optimization_level=%s",
            backend_name,
            ansatz_id,
            self.qiskit_optimization_level,
        )
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=self.qiskit_optimization_level,
        )
        compiled = pm.run(qiskit_template)
        self._transpile_cache[key] = compiled
        return compiled

    def _run_ibm_counts(
        self,
        qiskit_template: Any,
        qiskit_parameters: list[Any],
        theta: list[float],
        num_qubits: int,
        shots: int,
        timeout_sec: float | None,
        ansatz_id: str,
    ) -> tuple[dict[str, int], dict[str, Any]]:
        self._refresh_ibm_usage_if_needed()
        qpu = self.qpus["ibm_quantum"]
        if not qpu.is_available:
            raise RuntimeError(f"IBM unavailable: {qpu.last_error}")

        backend_info = self._select_ibm_backend_info(num_qubits)
        if backend_info is None:
            raise RuntimeError(f"No IBM backend supports {num_qubits} qubits")
        backend = backend_info["backend"]
        backend_name = backend_info.get("name", getattr(backend, "name", "unknown"))
        service = self.sessions.get("ibm_quantum")
        runtime_remaining = _get_ibm_usage_remaining_seconds(service) if service is not None else None
        LOGGER.info(
            "Preparing IBM submission | backend=%s required_qubits=%s backend_qubits=%s pending_jobs=%s runtime_remaining_sec=%s shots=%s",
            backend_name,
            num_qubits,
            backend_info.get("num_qubits"),
            backend_info.get("pending_jobs"),
            "unknown" if runtime_remaining is None else f"{runtime_remaining:.1f}",
            shots,
        )

        compiled_template = self._get_transpiled_template(
            backend=backend,
            ansatz_id=ansatz_id,
            qiskit_template=qiskit_template,
        )

        # Capture transpiled circuit metrics for reproducibility
        transpiled_metrics: dict[str, Any] = {
            "optimization_level": self.qiskit_optimization_level,
        }
        try:
            transpiled_metrics["transpiled_depth"] = int(compiled_template.depth())
            one_q, two_q, meas = 0, 0, 0
            for inst, qargs, _ in compiled_template.data:
                if inst.name == "measure":
                    meas += 1
                elif len(qargs) == 1:
                    one_q += 1
                elif len(qargs) == 2:
                    two_q += 1
            transpiled_metrics["transpiled_1q_gates"] = one_q
            transpiled_metrics["transpiled_2q_gates"] = two_q
            transpiled_metrics["transpiled_measurements"] = meas
        except Exception:
            pass
        try:
            layout = compiled_template.layout
            if layout is not None and hasattr(layout, "final_layout"):
                final = layout.final_layout
                if final is not None:
                    qubit_map = {}
                    for vq, pq in final.get_virtual_bits().items():
                        try:
                            qubit_map[str(vq)] = int(pq)
                        except Exception:
                            pass
                    if qubit_map:
                        transpiled_metrics["qubit_mapping_sample_size"] = len(qubit_map)
        except Exception:
            pass

        # Build a name→value lookup from the *original* (pre-transpilation)
        # parameter list, then bind using the *transpiled* circuit's own
        # Parameter objects.  Transpilation (especially at opt-level 3) can
        # replace Parameter instances, so binding by original references fails.
        name_to_value = {
            qiskit_parameters[i].name: float(theta[i])
            for i in range(len(qiskit_parameters))
        }

        transpiled_params = list(compiled_template.parameters)
        num_transpiled_params = len(transpiled_params)

        # Safety check: if transpilation eliminated or changed the number of
        # parameters, re-transpile without the cache.
        if num_transpiled_params != len(qiskit_parameters):
            LOGGER.warning(
                "Parameter count mismatch after transpilation: "
                "original=%d transpiled=%d. Clearing transpile cache for this backend.",
                len(qiskit_parameters), num_transpiled_params,
            )
            cache_key = (
                str(backend_name),
                ansatz_id,
                str(self.qiskit_optimization_level),
            )
            self._transpile_cache.pop(cache_key, None)
            compiled_template = self._get_transpiled_template(
                backend=backend,
                ansatz_id=ansatz_id,
                qiskit_template=qiskit_template,
            )
            transpiled_params = list(compiled_template.parameters)
            num_transpiled_params = len(transpiled_params)

        # Try name-based binding first.
        bind_map = {}
        for p in transpiled_params:
            if p.name in name_to_value:
                bind_map[p] = name_to_value[p.name]

        submit_time_utc = datetime.now(timezone.utc).isoformat()
        sampler = SamplerV2(backend)

        if len(bind_map) == num_transpiled_params and num_transpiled_params > 0:
            # All parameters matched by name — bind and submit as before.
            bound_circuit = compiled_template.assign_parameters(bind_map)
            LOGGER.debug(
                "Parameter binding: name-based | bound=%d/%d",
                len(bind_map), num_transpiled_params,
            )
            job = sampler.run([bound_circuit], shots=int(shots))
        elif num_transpiled_params > 0:
            # Name-based binding failed or was incomplete.
            # Use the SamplerV2 PUB format: (circuit, parameter_values)
            # which passes values positionally to the transpiled circuit.
            LOGGER.warning(
                "Parameter binding: name-based matched only %d/%d params. "
                "Falling back to SamplerV2 PUB positional format.",
                len(bind_map), num_transpiled_params,
            )
            # Build positional parameter values matching the transpiled circuit's
            # parameter order.  Try to match by name; if a transpiled parameter
            # name is not found, fall back to positional order from the original.
            param_values = []
            positional_idx = 0
            for p in transpiled_params:
                if p.name in name_to_value:
                    param_values.append(name_to_value[p.name])
                elif positional_idx < len(theta):
                    param_values.append(float(theta[positional_idx]))
                    positional_idx += 1
                else:
                    param_values.append(0.0)  # should not happen
            job = sampler.run(
                [(compiled_template, param_values)],
                shots=int(shots),
            )
        else:
            # No parameters (fully concrete circuit) — submit directly.
            job = sampler.run([compiled_template], shots=int(shots))

        job_id = getattr(job, "job_id", None)
        if callable(job_id):
            job_id = job_id()
        LOGGER.info("IBM job submitted | backend=%s job_id=%s", backend_name, job_id)

        try:
            self._wait_for_terminal_status(
                runtime_handle=job,
                label=f"IBM job {job_id or 'unknown'} ({backend_name})",
                timeout_sec=timeout_sec,
                terminal_states={"DONE", "ERROR", "CANCELLED"},
                success_states={"DONE"},
                min_log_interval_sec=120.0,
            )
        except RuntimeError as err:
            # Job ended in ERROR or CANCELLED state — try to extract the actual
            # error message from the job for better diagnostics.
            error_detail = str(err)
            try:
                error_msg = getattr(job, "error_message", None)
                if callable(error_msg):
                    error_msg = error_msg()
                if error_msg:
                    error_detail = f"{error_detail} | detail: {str(error_msg)[:300]}"
            except Exception:
                pass
            try:
                # Some IBM Runtime versions expose .result() even on failure
                # and the exception message contains the backend error.
                _ = job.result()
            except Exception as result_exc:
                result_detail = str(result_exc)[:300]
                if result_detail and result_detail not in error_detail:
                    error_detail = f"{error_detail} | result_error: {result_detail}"

            # Invalidate transpile cache for this backend — the transpiled
            # circuit itself may be causing the backend-side error.
            cache_key = (
                str(backend_name),
                ansatz_id,
                str(self.qiskit_optimization_level),
            )
            if cache_key in self._transpile_cache:
                del self._transpile_cache[cache_key]
                LOGGER.info(
                    "Cleared transpile cache for backend=%s ansatz=%s after job ERROR.",
                    backend_name, ansatz_id,
                )
            raise RuntimeError(error_detail) from err

        result = job.result()
        complete_time_utc = datetime.now(timezone.utc).isoformat()
        counts = self._extract_sampler_v2_counts(result=result, num_qubits=num_qubits)
        LOGGER.info(
            "IBM job completed | backend=%s job_id=%s unique_bitstrings=%s",
            backend_name,
            job_id,
            len(counts),
        )
        return counts, {
            "provider": "ibm",
            "backend_name": backend_name,
            "job_id": job_id,
            "qpu_id": "ibm_quantum",
            "backend_qubits": backend_info.get("num_qubits"),
            "pending_jobs_at_submit": backend_info.get("pending_jobs"),
            "submit_time_utc": submit_time_utc,
            "complete_time_utc": complete_time_utc,
            "shots": shots,
            **transpiled_metrics,
        }

    def _run_ibm_counts_batch(
        self,
        qiskit_template: Any,
        qiskit_parameters: list[Any],
        thetas: list[list[float]],
        num_qubits: int,
        shots: int,
        timeout_sec: float | None,
        ansatz_id: str,
    ) -> tuple[list[dict[str, int]], list[dict[str, Any]]]:
        if not thetas:
            raise ValueError("thetas must not be empty for batched IBM execution")

        self._refresh_ibm_usage_if_needed()
        qpu = self.qpus["ibm_quantum"]
        if not qpu.is_available:
            raise RuntimeError(f"IBM unavailable: {qpu.last_error}")

        backend_info = self._select_ibm_backend_info(num_qubits)
        if backend_info is None:
            raise RuntimeError(f"No IBM backend supports {num_qubits} qubits")
        backend = backend_info["backend"]
        backend_name = backend_info.get("name", getattr(backend, "name", "unknown"))
        service = self.sessions.get("ibm_quantum")
        runtime_remaining = _get_ibm_usage_remaining_seconds(service) if service is not None else None
        LOGGER.info(
            "Preparing IBM batch submission | backend=%s required_qubits=%s backend_qubits=%s pending_jobs=%s runtime_remaining_sec=%s shots=%s batch_size=%s",
            backend_name,
            num_qubits,
            backend_info.get("num_qubits"),
            backend_info.get("pending_jobs"),
            "unknown" if runtime_remaining is None else f"{runtime_remaining:.1f}",
            shots,
            len(thetas),
        )

        compiled_template = self._get_transpiled_template(
            backend=backend,
            ansatz_id=ansatz_id,
            qiskit_template=qiskit_template,
        )
        # Build a name→value lookup from the *original* (pre-transpilation)
        # parameter list, then bind using the *transpiled* circuit's own
        # Parameter objects.  Transpilation (especially at opt-level 3) can
        # replace Parameter instances, so binding by original references fails.
        transpiled_params = list(compiled_template.parameters)
        num_transpiled_params = len(transpiled_params)

        # Safety check: if transpilation changed the parameter count, re-transpile.
        if num_transpiled_params != len(qiskit_parameters):
            LOGGER.warning(
                "Batch: parameter count mismatch after transpilation: "
                "original=%d transpiled=%d. Clearing cache.",
                len(qiskit_parameters), num_transpiled_params,
            )
            cache_key = (
                str(backend_name),
                ansatz_id,
                str(self.qiskit_optimization_level),
            )
            self._transpile_cache.pop(cache_key, None)
            compiled_template = self._get_transpiled_template(
                backend=backend, ansatz_id=ansatz_id, qiskit_template=qiskit_template,
            )
            transpiled_params = list(compiled_template.parameters)
            num_transpiled_params = len(transpiled_params)

        # Check if name-based binding will work using the first theta.
        test_name_to_value = {
            qiskit_parameters[i].name: 0.0 for i in range(len(qiskit_parameters))
        }
        test_bind_map = {p: 0.0 for p in transpiled_params if p.name in test_name_to_value}
        use_pub_format = (len(test_bind_map) != num_transpiled_params and num_transpiled_params > 0)

        if use_pub_format:
            LOGGER.warning(
                "Batch: name-based binding matched only %d/%d params. "
                "Using SamplerV2 PUB positional format.",
                len(test_bind_map), num_transpiled_params,
            )

        pubs = []
        for theta in thetas:
            name_to_value = {
                qiskit_parameters[i].name: float(theta[i])
                for i in range(len(qiskit_parameters))
            }
            if use_pub_format:
                param_values = []
                positional_idx = 0
                for p in transpiled_params:
                    if p.name in name_to_value:
                        param_values.append(name_to_value[p.name])
                    elif positional_idx < len(theta):
                        param_values.append(float(theta[positional_idx]))
                        positional_idx += 1
                    else:
                        param_values.append(0.0)
                pubs.append((compiled_template, param_values))
            else:
                bind_map = {}
                for p in transpiled_params:
                    if p.name in name_to_value:
                        bind_map[p] = name_to_value[p.name]
                pubs.append(compiled_template.assign_parameters(bind_map))

        sampler = SamplerV2(backend)
        job = sampler.run(pubs, shots=int(shots))
        job_id = getattr(job, "job_id", None)
        if callable(job_id):
            job_id = job_id()
        LOGGER.info(
            "IBM batch job submitted | backend=%s job_id=%s batch_size=%s",
            backend_name,
            job_id,
            len(bound_circuits),
        )

        self._wait_for_terminal_status(
            runtime_handle=job,
            label=f"IBM batch job {job_id or 'unknown'} ({backend_name})",
            timeout_sec=timeout_sec,
            terminal_states={"DONE", "ERROR", "CANCELLED"},
            success_states={"DONE"},
            min_log_interval_sec=120.0,
        )
        result = job.result()

        try:
            pub_results = list(result)
        except Exception:
            pub_results = []

        counts_list: list[dict[str, int]] = []
        metadata_list: list[dict[str, Any]] = []
        for idx, pub_result in enumerate(pub_results):
            counts = self._extract_sampler_v2_counts_from_pub_result(
                pub_result=pub_result,
                num_qubits=num_qubits,
            )
            counts_list.append(counts)
            metadata_list.append(
                {
                    "provider": "ibm",
                    "backend_name": backend_name,
                    "job_id": job_id,
                    "qpu_id": "ibm_quantum",
                    "batch_index": idx,
                    "batch_size": len(bound_circuits),
                }
            )
        LOGGER.info(
            "IBM batch job completed | backend=%s job_id=%s batch_size=%s",
            backend_name,
            job_id,
            len(bound_circuits),
        )
        if len(counts_list) != len(bound_circuits):
            raise RuntimeError(
                f"IBM batch result size mismatch: submitted={len(bound_circuits)} received={len(counts_list)}"
            )
        return counts_list, metadata_list

    def _run_local_qiskit_counts(
        self,
        qiskit_template: Any,
        qiskit_parameters: list[Any],
        theta: list[float],
        num_qubits: int,
        shots: int,
    ) -> tuple[dict[str, int], dict[str, Any]]:
        LOGGER.debug(
            "Running local qiskit sampler | shots=%s qubits=%s sampler_kind=%s",
            shots,
            num_qubits,
            LOCAL_SAMPLER_KIND,
        )
        # Use name-based binding to avoid stale Parameter object references.
        name_to_value = {
            qiskit_parameters[i].name: float(theta[i])
            for i in range(len(qiskit_parameters))
        }
        bind_map = {}
        for p in qiskit_template.parameters:
            if p.name in name_to_value:
                bind_map[p] = name_to_value[p.name]
        bound_circuit = qiskit_template.assign_parameters(bind_map)
        sampler = LocalSampler()
        if LOCAL_SAMPLER_KIND == "v1":
            result = sampler.run([bound_circuit], shots=int(shots)).result()
            quasi = result.quasi_dists[0]
            raw_counts = {
                int(bitstring_int): int(round(float(prob) * float(shots)))
                for bitstring_int, prob in quasi.items()
                if float(prob) > 0.0
            }
            if not raw_counts and len(quasi) > 0:
                # Guard against rare rounding-to-zero outcomes.
                argmax_key = max(quasi, key=lambda key: float(quasi[key]))
                raw_counts[int(argmax_key)] = int(shots)
            counts = normalize_counts(raw_counts, num_qubits=num_qubits, reverse_bits=True)
        else:
            result = sampler.run([(bound_circuit,)], shots=int(shots)).result()
            raw_counts = {}
            pub_result = result[0]
            if hasattr(pub_result, "data"):
                data = pub_result.data
                if hasattr(data, "meas"):
                    raw_counts = data.meas.get_counts()
                elif hasattr(data, "c"):
                    raw_counts = data.c.get_counts()
                else:
                    for attr in dir(data):
                        if attr.startswith("_"):
                            continue
                        try:
                            maybe = getattr(data, attr)
                        except Exception:
                            continue
                        if hasattr(maybe, "get_counts"):
                            raw_counts = maybe.get_counts()
                            break
            counts = normalize_counts(raw_counts, num_qubits=num_qubits, reverse_bits=True)
        if not counts:
            counts["0" * int(num_qubits)] = int(shots)
        LOGGER.debug("Local qiskit sampling completed | unique_bitstrings=%s", len(counts))
        return counts, {"provider": "local_qiskit", "qpu_id": "local_qiskit"}

    def run_counts_batch(
        self,
        qpu_id: str,
        *,
        qiskit_template: Any,
        qiskit_parameters: list[Any],
        braket_template: Any,
        braket_parameters: list[Any],
        thetas: list[list[float]],
        num_qubits: int,
        shots: int,
        timeout_sec: float | None,
        ansatz_id: str,
    ) -> tuple[list[dict[str, int]], list[dict[str, Any]]]:
        if not thetas:
            raise ValueError("thetas must not be empty")

        self.refresh_credentials_if_needed()
        if qpu_id not in self.qpus:
            raise ValueError(f"Unknown qpu_id '{qpu_id}'")
        qpu = self.qpus[qpu_id]
        if not qpu.is_available:
            raise RuntimeError(f"QPU '{qpu_id}' unavailable: {qpu.last_error}")

        if qpu.provider == "ibm" and len(thetas) > 1:
            if not HAS_IBM_RUNTIME:
                raise RuntimeError("qiskit_ibm_runtime is not installed")
            LOGGER.info(
                "Dispatching batched quantum job | qpu_id=%s provider=%s qubits=%s shots=%s batch_size=%s",
                qpu_id,
                qpu.provider,
                num_qubits,
                shots,
                len(thetas),
            )
            start = time.time()
            try:
                with self._qpu_locks[qpu_id]:
                    counts_list, metadata_list = self._run_ibm_counts_batch(
                        qiskit_template=qiskit_template,
                        qiskit_parameters=qiskit_parameters,
                        thetas=thetas,
                        num_qubits=num_qubits,
                        shots=shots,
                        timeout_sec=timeout_sec,
                        ansatz_id=ansatz_id,
                    )
            except Exception as exc:
                elapsed = time.time() - start
                with self._lock:
                    qpu.total_jobs += 1
                    qpu.failed_jobs += 1
                    qpu.total_time_sec += elapsed
                LOGGER.exception(
                    "Batched quantum job failed | qpu_id=%s provider=%s elapsed_sec=%.3f batch_size=%s error=%s",
                    qpu_id,
                    qpu.provider,
                    elapsed,
                    len(thetas),
                    exc,
                )
                raise

            elapsed = time.time() - start
            with self._lock:
                qpu.total_jobs += 1
                qpu.successful_jobs += 1
                qpu.total_time_sec += elapsed
            per_eval_elapsed = elapsed / float(max(1, len(metadata_list)))
            for metadata in metadata_list:
                metadata["elapsed_sec"] = elapsed
                metadata["per_evaluation_elapsed_sec"] = per_eval_elapsed
            LOGGER.info(
                "Batched quantum job completed | qpu_id=%s provider=%s elapsed_sec=%.3f batch_size=%s total_jobs=%s",
                qpu_id,
                qpu.provider,
                elapsed,
                len(thetas),
                qpu.total_jobs,
            )
            return counts_list, metadata_list

        counts_list: list[dict[str, int]] = []
        metadata_list: list[dict[str, Any]] = []
        for theta in thetas:
            counts, metadata = self.run_counts(
                qpu_id=qpu_id,
                qiskit_template=qiskit_template,
                qiskit_parameters=qiskit_parameters,
                braket_template=braket_template,
                braket_parameters=braket_parameters,
                theta=theta,
                num_qubits=num_qubits,
                shots=shots,
                timeout_sec=timeout_sec,
                ansatz_id=ansatz_id,
            )
            counts_list.append(counts)
            metadata_list.append(metadata)
        return counts_list, metadata_list

    def run_counts(
        self,
        qpu_id: str,
        *,
        qiskit_template: Any,
        qiskit_parameters: list[Any],
        braket_template: Any,
        braket_parameters: list[Any],
        theta: list[float],
        num_qubits: int,
        shots: int,
        timeout_sec: float | None,
        ansatz_id: str,
    ) -> tuple[dict[str, int], dict[str, Any]]:
        self.refresh_credentials_if_needed()
        if qpu_id not in self.qpus:
            raise ValueError(f"Unknown qpu_id '{qpu_id}'")
        qpu = self.qpus[qpu_id]
        if not qpu.is_available:
            raise RuntimeError(f"QPU '{qpu_id}' unavailable: {qpu.last_error}")

        remaining_window = self._remaining_window_minutes_sgt(qpu)
        runtime_remaining_sec: float | None = None
        if qpu.provider == "ibm":
            service = self.sessions.get("ibm_quantum")
            runtime_remaining_sec = (
                _get_ibm_usage_remaining_seconds(service) if service is not None else None
            )

        # Guard: reject dispatch if the availability window has closed
        if remaining_window is not None and remaining_window <= 0 and not self._is_simulator_provider(qpu.provider):
            qpu.is_available = False
            qpu.last_error = "Availability window has closed"
            raise QPUWindowExpiredError(
                qpu_id,
                f"QPU '{qpu_id}' availability window has closed (0m remaining). "
                "Job not dispatched.",
            )
        # Also check sufficient runway
        if not self._is_simulator_provider(qpu.provider) and not self._has_sufficient_window(qpu):
            qpu.is_available = False
            remaining_str = f"{remaining_window:.0f}m" if remaining_window is not None else "unknown"
            qpu.last_error = f"Availability window closing soon ({remaining_str} remaining)"
            raise QPUWindowExpiredError(
                qpu_id,
                f"QPU '{qpu_id}' window closing soon ({remaining_str} remaining). "
                "Job not dispatched.",
            )

        window_info = ""
        if remaining_window is not None:
            window_info = f" | window: {remaining_window:.0f}m left"
        if runtime_remaining_sec is not None:
            window_info += f" | IBM budget: {runtime_remaining_sec:.0f}s left"
        LOGGER.info(
            "Dispatching job -> %s | shots=%s qubits=%s%s",
            qpu_id, shots, num_qubits, window_info,
        )
        start = time.time()
        try:
            with self._qpu_locks[qpu_id]:
                if qpu.provider.startswith("braket_"):
                    if braket_template is None or braket_parameters is None:
                        raise RuntimeError("Braket execution requested but Braket ansatz is not available")
                    counts, metadata = self._run_braket_counts(
                        qpu_id=qpu_id,
                        circuit_template=braket_template,
                        braket_parameters=braket_parameters,
                        theta=theta,
                        num_qubits=num_qubits,
                        shots=shots,
                        timeout_sec=timeout_sec,
                    )
                elif qpu.provider == "ibm":
                    if not HAS_IBM_RUNTIME:
                        raise RuntimeError("qiskit_ibm_runtime is not installed")
                    counts, metadata = self._run_ibm_counts(
                        qiskit_template=qiskit_template,
                        qiskit_parameters=qiskit_parameters,
                        theta=theta,
                        num_qubits=num_qubits,
                        shots=shots,
                        timeout_sec=timeout_sec,
                        ansatz_id=ansatz_id,
                    )
                elif qpu.provider == "local_qiskit":
                    if not HAS_QISKIT:
                        raise RuntimeError("qiskit is not installed for local simulation")
                    counts, metadata = self._run_local_qiskit_counts(
                        qiskit_template=qiskit_template,
                        qiskit_parameters=qiskit_parameters,
                        theta=theta,
                        num_qubits=num_qubits,
                        shots=shots,
                    )
                else:
                    raise RuntimeError(f"Unsupported provider '{qpu.provider}'")
        except Exception as exc:
            elapsed = time.time() - start
            with self._lock:
                qpu.total_jobs += 1
                qpu.failed_jobs += 1
                qpu.total_time_sec += elapsed
            LOGGER.exception(
                "Quantum job failed | qpu_id=%s provider=%s elapsed_sec=%.3f error=%s",
                qpu_id,
                qpu.provider,
                elapsed,
                exc,
            )
            raise

        elapsed = time.time() - start
        with self._lock:
            qpu.total_jobs += 1
            qpu.successful_jobs += 1
            qpu.total_time_sec += elapsed
        metadata["elapsed_sec"] = elapsed
        LOGGER.info(
            "Job completed on %s in %.1fs (total: %d ok, %d failed)",
            qpu_id, elapsed, qpu.successful_jobs, qpu.failed_jobs,
        )
        return counts, metadata
