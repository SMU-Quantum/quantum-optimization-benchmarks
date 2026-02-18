from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
IBM_MIN_RUNTIME_SECONDS = 50.0

LOGGER = logging.getLogger("qobench.hardware_manager")


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
        "amazon_tn1": QPUConfig(
            name="Amazon TN1",
            provider="braket_sim_tn1",
            max_qubits=50,
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
            name="Local Qiskit Sampler",
            provider="local_qiskit",
            max_qubits=32,
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
        use_aws: bool = True,
        use_ibm: bool = True,
        allow_simulators: bool = True,
        enabled_qpu_ids: Optional[set[str]] = None,
        qiskit_optimization_level: int = 3,
    ) -> None:
        self.aws_profile = aws_profile
        self.ibm_token = ibm_token
        self.ibm_instance = ibm_instance
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
        self._ibm_last_backend_refresh = 0.0
        self.ibm_backend_refresh_interval = 300.0
        self.min_window_remaining_minutes = MIN_WINDOW_REMAINING_MINUTES
        self.min_aws_window_remaining_minutes = MIN_AWS_WINDOW_REMAINING_MINUTES
        self.ibm_min_runtime_seconds = IBM_MIN_RUNTIME_SECONDS
        self.job_status_log_interval = 30.0
        LOGGER.debug(
            "QuantumHardwareManager initialized | use_aws=%s use_ibm=%s allow_simulators=%s aws_profile=%s enabled_qpus=%s qiskit_optimization_level=%s",
            self.use_aws,
            self.use_ibm,
            self.allow_simulators,
            self.aws_profile,
            sorted(self.enabled_qpu_ids) if self.enabled_qpu_ids is not None else "all",
            self.qiskit_optimization_level,
        )

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
        results: dict[str, bool] = {}
        LOGGER.debug(
            "Initializing backends | use_aws=%s use_ibm=%s allow_simulators=%s",
            self.use_aws,
            self.use_ibm,
            self.allow_simulators,
        )

        if self.use_aws and HAS_BRAKET:
            for qpu_id in (
                "rigetti_ankaa3",
                "iqm_emerald",
                "iqm_garnet",
                "amazon_sv1",
                "amazon_tn1",
            ):
                if not self._is_enabled(qpu_id):
                    self._mark_disabled(qpu_id)
                    results[qpu_id] = False
                    continue
                if not self.allow_simulators and self._is_simulator_provider(
                    self.qpus[qpu_id].provider
                ):
                    self.qpus[qpu_id].is_available = False
                    self.qpus[qpu_id].last_error = "Simulators disabled by configuration"
                    LOGGER.debug("Skipping simulator backend by configuration | qpu_id=%s", qpu_id)
                    results[qpu_id] = False
                    continue
                success = self._init_braket_device(qpu_id)
                results[qpu_id] = success
        else:
            for qpu_id in ("rigetti_ankaa3", "iqm_emerald", "iqm_garnet", "amazon_sv1", "amazon_tn1"):
                if not self._is_enabled(qpu_id):
                    self._mark_disabled(qpu_id)
                else:
                    self.qpus[qpu_id].is_available = False
                    self.qpus[qpu_id].last_error = "AWS Braket unavailable or disabled"
                    LOGGER.debug("AWS backend unavailable/disabled | qpu_id=%s", qpu_id)
                results[qpu_id] = False

        if not self._is_enabled("ibm_quantum"):
            self._mark_disabled("ibm_quantum")
            results["ibm_quantum"] = False
        elif self.use_ibm and HAS_IBM_RUNTIME:
            success = self._init_ibm_device()
            results["ibm_quantum"] = success
        else:
            self.qpus["ibm_quantum"].is_available = False
            self.qpus["ibm_quantum"].last_error = "IBM runtime unavailable or disabled"
            LOGGER.debug("IBM backend unavailable/disabled")
            results["ibm_quantum"] = False

        if not self._is_enabled("local_qiskit"):
            self._mark_disabled("local_qiskit")
            results["local_qiskit"] = False
        elif self.allow_simulators and HAS_QISKIT:
            self.qpus["local_qiskit"].is_available = True
            self.qpus["local_qiskit"].last_error = ""
            LOGGER.debug("Local qiskit backend enabled")
            results["local_qiskit"] = True
        else:
            self.qpus["local_qiskit"].is_available = False
            self.qpus["local_qiskit"].last_error = "Local qiskit simulator unavailable/disabled"
            LOGGER.debug("Local qiskit backend unavailable/disabled")
            results["local_qiskit"] = False

        LOGGER.debug("Backend initialization complete | results=%s", results)
        return results

    def _init_braket_device(self, qpu_id: str) -> bool:
        qpu = self.qpus[qpu_id]
        LOGGER.debug(
            "Initializing AWS Braket backend | qpu_id=%s provider=%s region=%s profile=%s",
            qpu_id,
            qpu.provider,
            qpu.region,
            self.aws_profile,
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
            elif qpu.provider == "braket_sim_tn1":
                device = AwsDevice(
                    "arn:aws:braket:::device/quantum-simulator/amazon/tn1",
                    aws_session=aws_session,
                )
            else:
                raise ValueError(f"Unknown Braket provider: {qpu.provider}")

            if str(device.status).upper() != "ONLINE":
                qpu.is_available = False
                qpu.last_error = f"Device status is {device.status}"
                LOGGER.warning(
                    "AWS Braket backend offline | qpu_id=%s status=%s",
                    qpu_id,
                    device.status,
                )
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
            LOGGER.debug(
                "AWS Braket backend ready | qpu_id=%s status=ONLINE max_qubits=%s remaining_window_min_sgt=%s",
                qpu_id,
                qpu.max_qubits,
                remaining_window,
            )
            return True
        except Exception as exc:
            qpu.is_available = False
            qpu.last_error = str(exc)
            LOGGER.exception("AWS Braket backend init failed | qpu_id=%s error=%s", qpu_id, exc)
            return False

    def _init_ibm_device(self) -> bool:
        qpu = self.qpus["ibm_quantum"]
        LOGGER.debug(
            "Initializing IBM runtime service | token_provided=%s instance_provided=%s",
            bool(self.ibm_token),
            bool(self.ibm_instance),
        )
        try:
            service = None

            if self.ibm_token and self.ibm_instance:
                LOGGER.debug("Trying IBM connection with explicit token + instance")
                for channel in ("ibm_cloud", "ibm_quantum"):
                    try:
                        service = QiskitRuntimeService(
                            channel=channel,
                            token=self.ibm_token,
                            instance=self.ibm_instance,
                        )
                        if service.active_account() is not None:
                            LOGGER.debug("IBM connection established | channel=%s", channel)
                            break
                    except Exception:
                        service = None

            if service is None:
                LOGGER.debug("Trying IBM connection with saved system credentials")
                try:
                    service = QiskitRuntimeService()
                    if service.active_account() is None:
                        service = None
                    else:
                        LOGGER.debug("IBM connection established from saved credentials")
                except Exception:
                    service = None

            if service is None and self.ibm_token:
                LOGGER.debug("Trying IBM connection with token only")
                try:
                    service = QiskitRuntimeService(
                        channel="ibm_quantum",
                        token=self.ibm_token,
                    )
                    if service.active_account() is None:
                        service = None
                    else:
                        LOGGER.debug("IBM connection established with token only")
                except Exception:
                    service = None

            if service is None:
                qpu.is_available = False
                qpu.last_error = "Could not initialize IBM runtime service"
                LOGGER.warning("IBM initialization failed: could not initialize runtime service")
                return False

            remaining = _get_ibm_usage_remaining_seconds(service)
            LOGGER.debug(
                "IBM runtime budget check | remaining_seconds=%s threshold_seconds=%.0f",
                "unknown" if remaining is None else f"{remaining:.1f}",
                float(self.ibm_min_runtime_seconds),
            )
            if (
                remaining is not None
                and remaining < float(self.ibm_min_runtime_seconds)
            ):
                qpu.is_available = False
                qpu.last_error = (
                    f"IBM runtime remaining {remaining:.0f}s < {self.ibm_min_runtime_seconds:.0f}s"
                )
                LOGGER.warning("IBM disabled due to low runtime budget | error=%s", qpu.last_error)
                return False

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
                    LOGGER.debug(
                        "IBM backend discovered | backend=%s qubits=%s pending_jobs=%s",
                        backend.name,
                        int(backend.num_qubits),
                        "n/a" if pending_jobs is None else int(pending_jobs),
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
                qpu.max_qubits = int(self.ibm_backends_preferred[0]["num_qubits"])
                preferred_names = [
                    f"{item['name']}(qubits={item.get('num_qubits')},pending={item.get('pending_jobs')})"
                    for item in self.ibm_backends_preferred
                ]
                LOGGER.debug(
                    "IBM preferred backend candidates (least busy first) | backends=%s",
                    preferred_names,
                )

            self.sessions["ibm_quantum"] = service
            qpu.is_available = len(self.ibm_backends) > 0
            qpu.last_error = "" if qpu.is_available else "No operational IBM hardware backend"
            self._ibm_last_backend_refresh = time.time()
            LOGGER.debug(
                "IBM initialization complete | available=%s discovered_backends=%s selected_backend=%s selected_qubits=%s",
                qpu.is_available,
                len(self.ibm_backends),
                qpu.backend_name,
                qpu.max_qubits,
            )
            return qpu.is_available
        except Exception as exc:
            qpu.is_available = False
            qpu.last_error = str(exc)
            LOGGER.exception("IBM initialization failed | error=%s", exc)
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

    def _refresh_ibm_usage_if_needed(self) -> None:
        if not self._is_enabled("ibm_quantum"):
            return
        now = time.time()
        if now - self._ibm_usage_last_check < self._ibm_usage_check_interval:
            return
        self._ibm_usage_last_check = now

        qpu = self.qpus["ibm_quantum"]
        service = self.sessions.get("ibm_quantum")
        if service is None:
            return

        remaining = _get_ibm_usage_remaining_seconds(service)
        if remaining is None:
            LOGGER.debug("IBM usage refresh | remaining runtime unavailable from API")
            return

        threshold = float(self.ibm_min_runtime_seconds)
        LOGGER.debug(
            "IBM usage refresh | remaining_seconds=%.1f threshold_seconds=%.0f",
            remaining,
            threshold,
        )
        if remaining < threshold:
            qpu.is_available = False
            qpu.last_error = (
                f"IBM runtime remaining {remaining:.0f}s < {threshold:.0f}s"
            )
            LOGGER.warning("IBM disabled by runtime threshold | error=%s", qpu.last_error)
            if now - self._ibm_last_reconnect_attempt >= self._ibm_reconnect_interval:
                self._ibm_last_reconnect_attempt = now
                LOGGER.debug("Attempting IBM service reconnect after low runtime signal")
                self._init_ibm_device()
        else:
            qpu.is_available = True
            qpu.last_error = ""
            LOGGER.debug("IBM remains schedulable after usage refresh")

    def _refresh_ibm_backends_if_needed(self, force: bool = False) -> None:
        if not self.use_ibm or not HAS_IBM_RUNTIME:
            return
        if not self._is_enabled("ibm_quantum"):
            return
        now = time.time()
        if not force and (now - self._ibm_last_backend_refresh) < float(self.ibm_backend_refresh_interval):
            return
        LOGGER.debug(
            "Refreshing IBM backend inventory | force=%s refresh_interval_sec=%.1f",
            force,
            float(self.ibm_backend_refresh_interval),
        )
        self._init_ibm_device()

    def _refresh_braket_backends(
        self,
        include_unavailable: bool = True,
        in_window_only: bool = False,
    ) -> None:
        if not HAS_BRAKET or not self.use_aws:
            return
        LOGGER.debug(
            "Refreshing AWS Braket backends | include_unavailable=%s in_window_only=%s",
            include_unavailable,
            in_window_only,
        )
        for qpu_id in ("rigetti_ankaa3", "iqm_emerald", "iqm_garnet", "amazon_sv1", "amazon_tn1"):
            if not self._is_enabled(qpu_id):
                continue
            qpu = self.qpus[qpu_id]
            if in_window_only and not self.is_qpu_in_time_window(qpu):
                continue
            if not include_unavailable and not qpu.is_available:
                continue
            self._init_braket_device(qpu_id)

    def refresh_credentials_if_needed(self) -> None:
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

    def get_available_qpus_for_size(
        self,
        num_qubits: int,
        include_simulators: bool,
    ) -> list[str]:
        self.refresh_credentials_if_needed()
        candidates: list[tuple[int, str]] = []
        skipped: dict[str, str] = {}
        for qpu_id, qpu in self.qpus.items():
            if not qpu.is_available:
                skipped[qpu_id] = f"unavailable:{qpu.last_error}"
                continue
            if qpu.max_qubits < num_qubits:
                skipped[qpu_id] = f"insufficient_qubits:{qpu.max_qubits}<{num_qubits}"
                continue
            if not include_simulators and self._is_simulator_provider(qpu.provider):
                skipped[qpu_id] = "simulator_excluded"
                continue
            if not self.is_qpu_in_time_window(qpu):
                skipped[qpu_id] = "outside_time_window"
                continue
            if not self._has_sufficient_window(qpu):
                remaining = self._remaining_window_minutes_sgt(qpu)
                skipped[qpu_id] = (
                    f"window_too_short:{remaining:.1f}m"
                    if remaining is not None
                    else "window_too_short"
                )
                continue
            candidates.append((qpu.priority, qpu_id))

        candidates.sort(key=lambda item: (item[0], item[1]))
        selected = [item[1] for item in candidates]
        LOGGER.debug(
            "QPU availability query | required_qubits=%s include_simulators=%s selected=%s",
            num_qubits,
            include_simulators,
            selected,
        )
        if skipped:
            LOGGER.debug("QPU availability skipped reasons | %s", skipped)
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
        return counts, {"task_id": task_id, "provider": "braket", "qpu_id": qpu_id}

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
        bind_map = {qiskit_parameters[i]: float(theta[i]) for i in range(len(qiskit_parameters))}
        bound_circuit = compiled_template.assign_parameters(bind_map)

        sampler = SamplerV2(backend)
        job = sampler.run([bound_circuit], shots=int(shots))
        job_id = getattr(job, "job_id", None)
        if callable(job_id):
            job_id = job_id()
        LOGGER.info("IBM job submitted | backend=%s job_id=%s", backend_name, job_id)

        self._wait_for_terminal_status(
            runtime_handle=job,
            label=f"IBM job {job_id or 'unknown'} ({backend_name})",
            timeout_sec=timeout_sec,
            terminal_states={"DONE", "ERROR", "CANCELLED"},
            success_states={"DONE"},
            min_log_interval_sec=120.0,
        )
        result = job.result()
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
        bound_circuits = []
        for theta in thetas:
            bind_map = {
                qiskit_parameters[i]: float(theta[i]) for i in range(len(qiskit_parameters))
            }
            bound_circuits.append(compiled_template.assign_parameters(bind_map))

        sampler = SamplerV2(backend)
        job = sampler.run(bound_circuits, shots=int(shots))
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
        bind_map = {qiskit_parameters[i]: float(theta[i]) for i in range(len(qiskit_parameters))}
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
        LOGGER.debug(
            "Dispatching quantum job | qpu_id=%s provider=%s shots=%s qubits=%s timeout_sec=%s remaining_window_min_sgt=%s runtime_remaining_sec=%s",
            qpu_id,
            qpu.provider,
            shots,
            num_qubits,
            timeout_sec,
            remaining_window,
            runtime_remaining_sec,
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
        LOGGER.debug(
            "Quantum job completed | qpu_id=%s provider=%s elapsed_sec=%.3f total_jobs=%s success=%s failed=%s",
            qpu_id,
            qpu.provider,
            elapsed,
            qpu.total_jobs,
            qpu.successful_jobs,
            qpu.failed_jobs,
        )
        return counts, metadata
