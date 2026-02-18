#!/usr/bin/env python3
from __future__ import annotations
"""
Multi-QPU Quantum Evaluation Suite for CVRP Agent

This script orchestrates quantum execution across multiple quantum processors:
- IBM Quantum (least busy selection, variable qubits)
- Rigetti Ankaa-3 (AWS Braket, 82 qubits)
- IQM Garnet (AWS Braket, 20 qubits)
- IQM Emerald (AWS Braket, 54 qubits)

Features:
1. Intelligent QPU routing based on problem size (qubit requirements)
2. Automatic fallback between QPUs when one is unavailable
3. Parallel knapsack solving on different QPUs
4. Probability-based solution selection (heatmap approach)
5. Detailed logging with timestamps and QPU tracking
6. Checkpoint and resume capability
7. Local noise model simulation as final fallback

Author: Multi-QPU Orchestration System
Date: 2026-02-04
"""

import os
import sys
import glob
import csv
import json
import re
import math
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
import threading
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import Batch

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
LOGGER = logging.getLogger("cvrp.multi_qpu")
LOGGER.setLevel(logging.INFO)

# Add paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_cvrp_root = os.path.join(_project_root, "cvrp_rl_project")
if os.path.isdir(_cvrp_root) and _cvrp_root not in sys.path:
    sys.path.insert(0, _cvrp_root)

# ============================================================
# OPTIONAL IMPORTS - Handle missing dependencies gracefully
# ============================================================

# AWS Braket
HAS_BRAKET = False
try:
    from braket.aws import AwsDevice, AwsQuantumTask
    from braket.circuits import Circuit, Gate, FreeParameter
    from braket.devices import LocalSimulator, Devices
    from braket.tracking import Tracker
    from braket.aws import AwsSession
    import boto3
    from botocore.exceptions import ClientError
    HAS_BRAKET = True
except ImportError as e:
    LOGGER.warning(f"AWS Braket SDK not available: {e}")

# IBM Quantum
HAS_IBM = False
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    HAS_IBM = True
except ImportError as e:
    LOGGER.warning(f"IBM Quantum Runtime not available: {e}")

# Qiskit for local simulation
HAS_QISKIT = False
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    HAS_QISKIT = True
except ImportError as e:
    LOGGER.warning(f"Qiskit Aer not available: {e}")

# ============================================================
# NOISE-AWARE MODULES
# ============================================================
try:
    from depth_bandit import ConfigurationBandit, GlobalState
    from circuit_scaling import gate_budget_safe
    from smart_placement import SmartPlacement
    from entanglement_map import get_entanglement_map
    from hardware_descriptor import HardwareDescriptor, IBMAdapter, JSONAdapter

    def _get_topology_metrics(hw: HardwareDescriptor) -> Tuple[float, float]:
        g = hw.to_nx_graph()
        active_nodes = [n for n in g.nodes if g.nodes[n].get('available', False)]
        if not active_nodes:
            return 0.0, 0.0

        subgraph = g.subgraph(active_nodes)
        avg_degree = sum(dict(subgraph.degree()).values()) / len(active_nodes)

        if len(active_nodes) > 1:
            largest_cc = max(nx.connected_components(subgraph), key=len)
            diameter = nx.diameter(subgraph.subgraph(largest_cc))
        else:
            diameter = 0.0

        return float(avg_degree), float(diameter)

except ImportError as e:
    LOGGER.error(f"CRITICAL: Could not import Noise-Aware modules: {e}")
    sys.exit(1)

# Project imports
from src.imitation.preconditioned_policy import PreconditionedGATPolicy, DiagonalPrecondPolicy
from src.imitation.curriculum_env import CurriculumCVRPEnv
from src.utils.vrp_parser import CVRPParser


# ============================================================
# QPU CONFIGURATION WITH AVAILABILITY WINDOWS
# ============================================================
from datetime import datetime, timezone, timedelta

# Singapore timezone (UTC+8)
SGT = timezone(timedelta(hours=8))
MIN_WINDOW_REMAINING_MINUTES = 10  # Generic minimum window (minutes)
MIN_AWS_WINDOW_REMAINING_MINUTES = 30  # Strict minimum for AWS Braket hardware QPUs
IBM_MIN_RUNTIME_SECONDS = 50.0  # Minimum IBM runtime budget required to schedule new jobs

@dataclass
class QPUConfig:
    """Configuration for a quantum processing unit."""
    name: str
    provider: str  # 'ibm', 'braket_rigetti', 'braket_iqm_*', 'braket_sim_*', 'local'
    max_qubits: int
    region: str = 'us-west-1'
    priority: int = 1  # Lower = higher priority for similar qubit counts
    is_available: bool = True
    last_error: str = ""
    last_check: float = 0.0
    total_jobs: int = 0
    total_time: float = 0.0
    successful_jobs: int = 0
    failed_jobs: int = 0
    backend_name: str = ""  # Actual backend name (e.g., "ibm_brisbane")
    # Availability windows in SGT (list of (start_hour, end_hour) tuples)
    availability_windows_sgt: List[Tuple[int, int]] = None
    weekdays_only: bool = False  # If True, only available Mon-Fri
    allowed_weekdays: Optional[List[int]] = None  # If set, restricts availability to these weekday indices
    # Optional per-weekday availability windows in SGT: {weekday: [(sh, sm, eh, em), ...]}
    availability_windows_sgt_by_weekday: Optional[Dict[int, List[Tuple[int, int, int, int]]]] = None
    descriptor: Optional[HardwareDescriptor] = None
    json_path: str = ""


def _time_in_windows(now_dt: datetime, windows: List[Tuple]) -> bool:
    now_min = now_dt.hour * 60 + now_dt.minute
    for window in windows:
        if len(window) == 2:
            start_h, end_h = window
            start_min = start_h * 60
            end_min = end_h * 60
        elif len(window) == 4:
            start_h, start_m, end_h, end_m = window
            start_min = start_h * 60 + start_m
            end_min = end_h * 60 + end_m
        else:
            continue

        if start_min <= end_min:
            if start_min <= now_min < end_min:
                return True
        else:
            # Overnight window (e.g., 18:00-02:00)
            if now_min >= start_min or now_min < end_min:
                return True
    return False


def _remaining_minutes_in_windows(now_dt: datetime, windows: List[Tuple]) -> float:
    now_min = now_dt.hour * 60 + now_dt.minute
    for window in windows:
        if len(window) == 2:
            start_h, end_h = window
            start_min = start_h * 60
            end_min = end_h * 60
        elif len(window) == 4:
            start_h, start_m, end_h, end_m = window
            start_min = start_h * 60 + start_m
            end_min = end_h * 60 + end_m
        else:
            continue

        if start_min <= end_min:
            if start_min <= now_min < end_min:
                return float(end_min - now_min)
        else:
            # Overnight window (e.g., 18:00-02:00)
            if now_min >= start_min:
                return float((24 * 60 - now_min) + end_min)
            if now_min < end_min:
                return float(end_min - now_min)
    return 0.0


def _remaining_window_minutes_sgt(qpu: "QPUConfig") -> Optional[float]:
    now_sgt = datetime.now(SGT)
    if qpu.availability_windows_sgt_by_weekday is not None:
        windows = qpu.availability_windows_sgt_by_weekday.get(now_sgt.weekday())
        if not windows:
            return 0.0
        return _remaining_minutes_in_windows(now_sgt, windows)
    if qpu.availability_windows_sgt is None:
        return None
    return _remaining_minutes_in_windows(now_sgt, qpu.availability_windows_sgt)


def _get_ibm_usage_remaining_seconds(service: Any) -> Optional[float]:
    """Best-effort remaining runtime seconds from IBM service. Returns None if unavailable."""
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
    except (TypeError, ValueError):
        return None


def is_qpu_in_time_window(qpu: QPUConfig) -> bool:
    """Check if QPU is currently within its availability window (SGT time)."""
    now_sgt = datetime.now(SGT)
    current_weekday = now_sgt.weekday()  # 0=Monday, 6=Sunday

    if qpu.allowed_weekdays is not None and current_weekday not in qpu.allowed_weekdays:
        return False

    if qpu.availability_windows_sgt_by_weekday is not None:
        sgt_windows = qpu.availability_windows_sgt_by_weekday.get(current_weekday)
        if not sgt_windows:
            return False
        return _time_in_windows(now_sgt, sgt_windows)

    if qpu.availability_windows_sgt is None:
        return True  # No windows defined = always available
    
    # Check weekday restriction
    if qpu.weekdays_only and current_weekday >= 5:  # Saturday=5, Sunday=6
        return False

    # Handles both hour-only windows [(sh, eh)] and minute-precision windows [(sh, sm, eh, em)].
    return _time_in_windows(now_sgt, qpu.availability_windows_sgt)


# Default QPU configurations with availability windows (SGT)
AVAILABLE_QPUS = {
    'rigetti_ankaa3': QPUConfig(
        name="Rigetti Ankaa-3",
        provider="braket_rigetti",
        max_qubits=82,
        region='us-west-1',
        priority=1,
        # Available: 08:00-15:00, 17:00-03:00, 05:00-08:00 SGT (everyday)
        availability_windows_sgt=[(8, 15), (17, 24), (0, 3), (5, 8)],
        weekdays_only=False,
        json_path="rigetti_ankaa3.json"
    ),
    'iqm_emerald': QPUConfig(
        name="IQM Emerald",
        provider="braket_iqm_emerald",
        max_qubits=54,
        region='eu-north-1',
        priority=2,
        # Available: 08:00-11:30, 15:00-00:30, 04:00-08:00 SGT (weekdays)
        # Use minute-precision windows to avoid late submissions near close.
        availability_windows_sgt=[(8, 0, 11, 30), (15, 0, 0, 30), (4, 0, 8, 0)],
        weekdays_only=True,
        json_path="iqm_emerald.json"
    ),
    'iqm_garnet': QPUConfig(
        name="IQM Garnet",
        provider="braket_iqm_garnet",
        max_qubits=20,
        region='eu-north-1',
        priority=3,
        # Available: 08:00-09:30, 11:15-23:30, 01:15-08:00 SGT (weekdays)
        # Use minute-precision windows to avoid late submissions near close.
        availability_windows_sgt=[(8, 0, 9, 30), (11, 15, 23, 30), (1, 15, 8, 0)],
        weekdays_only=True,
        json_path="iqm_garnet.json"
    ),
    'amazon_sv1': QPUConfig(
        name="Amazon SV1 (Simulator)",
        provider="braket_sim_sv1",
        max_qubits=34,
        region='us-west-1',
        priority=6,
        availability_windows_sgt=None,
        weekdays_only=False
    ),
    'amazon_tn1': QPUConfig(
        name="Amazon TN1 (Simulator)",
        provider="braket_sim_tn1",
        max_qubits=50,
        region='us-west-1',
        priority=6,
        availability_windows_sgt=None,
        weekdays_only=False
    ),
    'ibm_quantum': QPUConfig(
        name="IBM Quantum",
        provider="ibm",
        max_qubits=156,  # User has backends with 156, 156, and 133 qubits
        priority=4,
        availability_windows_sgt=None,  # IBM is always available (queue-based)
        weekdays_only=False
    )
}


# ============================================================
# MULTI-QPU MANAGER
# ============================================================
class MultiQPUManager:
    """
    Manages multiple quantum processing units with intelligent routing,
    automatic failover, and credential refresh.
    """
    
    def __init__(self, aws_profile: str = None, ibm_token: str = None, ibm_instance: str = None, use_ibm: bool = True):
        self.aws_profile = aws_profile
        self.ibm_token = ibm_token
        self.ibm_instance = ibm_instance  # CRN for IBM Quantum instance
        self.use_ibm = use_ibm
        self.qpus: Dict[str, QPUConfig] = dict(AVAILABLE_QPUS)
        self.devices: Dict[str, Any] = {}
        self.sessions: Dict[str, Any] = {}
        self.lock = threading.Lock()
        self.last_credential_refresh = time.time()
        self.credential_refresh_interval = 2700  # 45 minutes
        self._ibm_rr_index = 0
        
        # Job tracking
        self.job_log: List[Dict] = []
        self.min_window_remaining_minutes = MIN_WINDOW_REMAINING_MINUTES
        self._ibm_usage_last_check = 0.0
        self._ibm_usage_check_interval = 30.0  # seconds
        self._ibm_last_reconnect_attempt = 0.0
        self._ibm_reconnect_interval = 120.0  # seconds
        self._last_schedule_reason_log = 0.0
        self._schedule_reason_log_interval = 45.0  # seconds
        self._last_opportunistic_refresh = 0.0
        self._opportunistic_refresh_interval = 180.0  # seconds
        self.min_aws_window_remaining_minutes = MIN_AWS_WINDOW_REMAINING_MINUTES
        self.ibm_min_runtime_seconds = IBM_MIN_RUNTIME_SECONDS

    @staticmethod
    def _is_simulator_provider(provider: str) -> bool:
        if provider is None:
            return False
        return provider == 'local' or provider.startswith('braket_sim_')

    @staticmethod
    def _braket_qpu_ids() -> List[str]:
        return [
            'rigetti_ankaa3',
            'iqm_emerald',
            'iqm_garnet',
            'amazon_sv1',
            'amazon_tn1',
        ]

    def _refresh_braket_backends(
        self,
        include_unavailable: bool = True,
        in_window_only: bool = False,
        reason: str = "periodic",
    ) -> None:
        if not HAS_BRAKET:
            return
        LOGGER.info(f"[MultiQPU] Refreshing Braket backends ({reason})...")
        for qpu_id in self._braket_qpu_ids():
            qpu = self.qpus.get(qpu_id)
            if qpu is None:
                continue
            if in_window_only and not is_qpu_in_time_window(qpu):
                continue
            if not include_unavailable and not qpu.is_available:
                continue
            try:
                success = self._init_braket_device(qpu_id)
                qpu.is_available = success
                if success:
                    qpu.last_error = ""
            except Exception as e:
                qpu.is_available = False
                qpu.last_error = str(e)
                LOGGER.warning(f"Failed to refresh {qpu_id}: {e}")

    def _qpu_runtime_score(self, qpu_id: str, qpu: QPUConfig, num_qubits: int) -> Tuple[float, Dict[str, float]]:
        attempts = qpu.successful_jobs + qpu.failed_jobs
        reliability = (qpu.successful_jobs + 1.0) / (attempts + 2.0)

        if qpu.successful_jobs > 0:
            avg_time = qpu.total_time / max(1, qpu.successful_jobs)
        else:
            avg_time = 30.0
        speed_score = 1.0 / (1.0 + (avg_time / 30.0))

        priority_score = 1.0 / max(1.0, float(qpu.priority))
        headroom = max(0.0, float(qpu.max_qubits - num_qubits))
        size_fit_score = 1.0 / (1.0 + (headroom / 40.0))

        remaining_window = _remaining_window_minutes_sgt(qpu)
        if remaining_window is None:
            window_signal = 0.6
            window_urgency = 0.0
        else:
            window_signal = 1.0
            window_urgency = max(0.0, 1.0 - min(remaining_window / 180.0, 1.0))

        failure_penalty = min(1.0, qpu.failed_jobs / 10.0)

        score = (
            2.2 * reliability
            + 1.2 * speed_score
            + 0.6 * priority_score
            + 0.5 * size_fit_score
            + 0.4 * window_signal
            + 0.8 * window_urgency
            - 0.6 * failure_penalty
        )

        parts = {
            "score": score,
            "reliability": reliability,
            "speed": speed_score,
            "priority": priority_score,
            "size_fit": size_fit_score,
            "window_signal": window_signal,
            "window_urgency": window_urgency,
            "failure_penalty": failure_penalty,
            "remaining_window": -1.0 if remaining_window is None else float(remaining_window),
        }
        return score, parts

    def _log_scheduler_snapshot(
        self,
        num_qubits: int,
        candidates: List[Tuple[float, int, str, Dict[str, float]]],
        skipped: Dict[str, str],
    ) -> None:
        now = time.time()
        should_log = len(candidates) <= 1 or (now - self._last_schedule_reason_log) >= self._schedule_reason_log_interval
        if not should_log:
            return
        self._last_schedule_reason_log = now

        if candidates:
            ranked = []
            for score, _, qpu_id, parts in candidates:
                remaining = parts.get("remaining_window", -1.0)
                if remaining < 0:
                    rem_txt = "always-on"
                else:
                    rem_txt = f"{remaining:.1f}m left"
                ranked.append(
                    f"{qpu_id}(score={score:.2f}, rel={parts['reliability']:.2f}, speed={parts['speed']:.2f}, {rem_txt})"
                )
            LOGGER.info(f"[Scheduler] Candidates for {num_qubits}q: {', '.join(ranked)}")
        else:
            LOGGER.warning(f"[Scheduler] No candidates for {num_qubits}q")

        if skipped:
            reasons = ", ".join(f"{qid}={reason}" for qid, reason in skipped.items())
            LOGGER.info(f"[Scheduler] Skipped: {reasons}")

    def _has_sufficient_window(self, qpu: QPUConfig) -> bool:
        remaining = _remaining_window_minutes_sgt(qpu)
        if remaining is None:
            return True
        required = self.min_window_remaining_minutes
        # AWS hardware QPUs require stricter buffer to avoid submitting near close.
        if qpu.provider.startswith('braket') and not self._is_simulator_provider(qpu.provider):
            required = max(required, self.min_aws_window_remaining_minutes)
            # Adaptive guard: if this backend is slow in practice, require extra runway.
            if qpu.successful_jobs > 0:
                avg_job_minutes = (qpu.total_time / max(1, qpu.successful_jobs)) / 60.0
                adaptive_required = min(180.0, (2.0 * avg_job_minutes) + 5.0)
                required = max(required, adaptive_required)
        return remaining >= required

    def _refresh_ibm_usage_if_needed(self) -> None:
        now = time.time()
        if now - self._ibm_usage_last_check < self._ibm_usage_check_interval:
            return
        self._ibm_usage_last_check = now

        qpu = self.qpus.get('ibm_quantum')
        if qpu is None:
            return
        if qpu.last_error and qpu.last_error.startswith("Disabled by --qpu"):
            return

        service = self.sessions.get('ibm_quantum')
        if service is None:
            return
        remaining = _get_ibm_usage_remaining_seconds(service)
        if remaining is None:
            LOGGER.info("[ibm_quantum] Usage check: remaining runtime unavailable from API")
            return
        threshold = float(self.ibm_min_runtime_seconds)
        LOGGER.info(f"[ibm_quantum] Usage check: remaining_runtime={remaining:.1f}s threshold={threshold:.0f}s")
        if remaining < threshold:
            qpu.is_available = False
            qpu.last_error = f"IBM runtime remaining {remaining:.0f}s < {threshold:.0f}s"
            LOGGER.warning(f"[ibm_quantum] Disabled for scheduling (remaining={remaining:.1f}s < {threshold:.0f}s)")
            # Mid-run re-auth/reload hook: detects refreshed account/token without restarting process.
            if now - self._ibm_last_reconnect_attempt >= self._ibm_reconnect_interval:
                self._ibm_last_reconnect_attempt = now
                LOGGER.info("[ibm_quantum] Attempting mid-run IBM re-auth/reload...")
                try:
                    success = self._init_ibm_device()
                    refreshed_service = self.sessions.get('ibm_quantum')
                    refreshed_remaining = _get_ibm_usage_remaining_seconds(refreshed_service) if refreshed_service else None
                    if refreshed_remaining is None:
                        qpu.is_available = False
                        qpu.last_error = "IBM runtime remaining unavailable after reload"
                        LOGGER.info("[ibm_quantum] Re-auth/reload completed; runtime after reload unavailable.")
                    elif success and refreshed_remaining >= threshold:
                        qpu.is_available = True
                        qpu.last_error = ""
                        LOGGER.info(
                            f"[ibm_quantum] Re-auth/reload succeeded; post-reload remaining_runtime="
                            f"{refreshed_remaining:.1f}s (>= {threshold:.0f}s). IBM re-enabled for scheduling."
                        )
                    else:
                        qpu.is_available = False
                        qpu.last_error = (
                            f"IBM runtime remaining {refreshed_remaining:.0f}s < {threshold:.0f}s after reload"
                            if refreshed_remaining is not None
                            else f"IBM runtime remaining < {threshold:.0f}s after reload"
                        )
                        LOGGER.info(
                            f"[ibm_quantum] Re-auth/reload completed; IBM still unavailable "
                            f"(post-reload remaining_runtime={refreshed_remaining:.1f}s, threshold={threshold:.0f}s)."
                        )
                except Exception as e:
                    LOGGER.warning(f"[ibm_quantum] Mid-run re-auth/reload failed: {e}")
        else:
            if not qpu.is_available:
                LOGGER.info(f"[ibm_quantum] Runtime available again ({remaining:.0f}s). Re-enabling IBM QPU.")
            qpu.is_available = True
            qpu.last_error = ""
        
    def initialize(self) -> Dict[str, bool]:
        """Initialize all available QPUs. Returns dict of QPU name -> success."""
        results = {}
        
        # Initialize AWS Braket QPUs
        if HAS_BRAKET:
            for qpu_id in self._braket_qpu_ids():
                try:
                    success = self._init_braket_device(qpu_id)
                    results[qpu_id] = success
                    self.qpus[qpu_id].is_available = success
                except Exception as e:
                    LOGGER.error(f"Failed to initialize {qpu_id}: {e}")
                    self.qpus[qpu_id].is_available = False
                    self.qpus[qpu_id].last_error = str(e)
                    results[qpu_id] = False
        else:
            for qpu_id in self._braket_qpu_ids():
                self.qpus[qpu_id].is_available = False
                self.qpus[qpu_id].last_error = "Braket SDK not installed"
                results[qpu_id] = False
        
        # Initialize IBM Quantum
        if HAS_IBM and self.use_ibm:
            try:
                success = self._init_ibm_device()
                results['ibm_quantum'] = success
                self.qpus['ibm_quantum'].is_available = success
            except Exception as e:
                LOGGER.error(f"Failed to initialize IBM Quantum: {e}")
                self.qpus['ibm_quantum'].is_available = False
                self.qpus['ibm_quantum'].last_error = str(e)
                results['ibm_quantum'] = False
        else:
            self.qpus['ibm_quantum'].is_available = False
            if not self.use_ibm:
                self.qpus['ibm_quantum'].last_error = "Disabled by user"
            else:
                self.qpus['ibm_quantum'].last_error = "IBM Runtime not installed"
            results['ibm_quantum'] = False
        
        # Initialize local MPS simulator as fallback (not shown in status, only if needed)
        if HAS_QISKIT:
            self._init_local_simulator()
            self.local_sim_available = True
        else:
            self.local_sim_available = False
        
        return results
    
    def _init_braket_device(self, qpu_id: str) -> bool:
        """Initialize a Braket device."""
        qpu = self.qpus[qpu_id]

        # Load Hardware Descriptor for Bandit
        if qpu.json_path:
            descriptor_path = qpu.json_path
            if not os.path.isabs(descriptor_path):
                descriptor_path = os.path.join(_script_dir, descriptor_path)
            if os.path.exists(descriptor_path):
                try:
                    qpu.descriptor = JSONAdapter.from_json(descriptor_path)
                    qpu.max_qubits = qpu.descriptor.num_qubits
                    LOGGER.info(f"[{qpu_id}] Loaded hardware descriptor from {descriptor_path}")
                except Exception as e:
                    LOGGER.warning(f"[{qpu_id}] Failed to load JSON descriptor: {e}")
            else:
                LOGGER.warning(f"[{qpu_id}] JSON descriptor not found at {descriptor_path}")

        # Create boto3 session with profile
        if self.aws_profile:
            LOGGER.info(f"[{qpu_id}] Using AWS profile: {self.aws_profile}")
            boto_session = boto3.Session(profile_name=self.aws_profile, region_name=qpu.region)
        else:
            boto_session = boto3.Session(region_name=qpu.region)
        
        # Create Braket session
        aws_session = AwsSession(boto_session=boto_session)
        
        # Get device based on provider
        if qpu.provider == 'braket_rigetti':
            device = AwsDevice(Devices.Rigetti.Ankaa3, aws_session=aws_session)
        elif qpu.provider == 'braket_iqm_emerald':
            device = AwsDevice(Devices.IQM.Emerald, aws_session=aws_session)
        elif qpu.provider == 'braket_iqm_garnet':
            device = AwsDevice(Devices.IQM.Garnet, aws_session=aws_session)
        elif qpu.provider == 'braket_sim_sv1':
            device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1", aws_session=aws_session)
        elif qpu.provider == 'braket_sim_tn1':
            device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/tn1", aws_session=aws_session)
        else:
            raise ValueError(f"Unknown Braket provider: {qpu.provider}")
        
        # Verify device is online
        status = device.status
        if status != 'ONLINE':
            LOGGER.warning(f"[{qpu_id}] Device status: {status}")
            qpu.is_available = False
            qpu.last_error = f"Device status: {status}"
            return False
        
        self.devices[qpu_id] = device
        self.sessions[qpu_id] = {'boto': boto_session, 'aws': aws_session}
        
        # Get actual qubit count
        try:
            props = device.properties
            if hasattr(props, 'paradigm') and hasattr(props.paradigm, 'qubitCount'):
                qpu.max_qubits = props.paradigm.qubitCount
        except:
            pass

        qpu.last_error = ""
        LOGGER.info(f"[{qpu_id}] Initialized: {qpu.name} ({qpu.max_qubits} qubits)")
        return True
    
    def _init_ibm_device(self) -> bool:
        """
        Initialize IBM Quantum service and discover available backends by qubit count.
        
        Connection priority:
        1. Saved credentials with instance override (if ibm_instance provided)
        2. Token + instance CRN (for new instances like open-instance)
        3. Saved credentials only
        4. Token only (with ibm_quantum channel)
        """
        try:
            service = None
            
            # If we have both token and instance, this is likely a new/different IBM account
            if self.ibm_token and self.ibm_instance:
                LOGGER.info(f"[ibm_quantum] Connecting with token + instance CRN...")
                LOGGER.info(f"[ibm_quantum] Instance: {self.ibm_instance[:80]}...")
                
                # Try ibm_cloud channel first (for newer IBM Cloud quantum instances)
                for channel in ['ibm_cloud', 'ibm_quantum']:
                    try:
                        service = QiskitRuntimeService(
                            channel=channel,
                            token=self.ibm_token,
                            instance=self.ibm_instance
                        )
                        account = service.active_account()
                        if account is not None:
                            LOGGER.info(f"[ibm_quantum] Connected via {channel} with instance CRN")
                            break
                    except Exception as e:
                        LOGGER.debug(f"Channel {channel} with instance failed: {e}")
                        service = None
                        continue
            
            # If no instance-based connection, try saved credentials
            if service is None:
                try:
                    LOGGER.info("[ibm_quantum] Trying saved credentials...")
                    service = QiskitRuntimeService()
                    account = service.active_account()
                    if account is None:
                        raise Exception("No active account in saved credentials")
                    LOGGER.info(f"[ibm_quantum] Connected via saved credentials")
                except Exception as saved_err:
                    LOGGER.debug(f"Saved credentials failed: {saved_err}")
                    service = None
            
            # If still no connection, try token with ibm_quantum channel (legacy)
            if service is None and self.ibm_token:
                LOGGER.info("[ibm_quantum] Trying token with ibm_quantum channel...")
                try:
                    service = QiskitRuntimeService(channel="ibm_quantum", token=self.ibm_token)
                    account = service.active_account()
                    if account is not None:
                        LOGGER.info(f"[ibm_quantum] Connected via ibm_quantum channel")
                except Exception as token_err:
                    LOGGER.debug(f"Token auth failed: {token_err}")
                    service = None
            
            if service is None:
                raise Exception("Could not connect to IBM Quantum with any method")
            
            threshold = float(self.ibm_min_runtime_seconds)
            remaining = _get_ibm_usage_remaining_seconds(service)
            if remaining is not None and remaining < threshold:
                self.qpus['ibm_quantum'].is_available = False
                self.qpus['ibm_quantum'].last_error = f"IBM runtime remaining {remaining:.0f}s < {threshold:.0f}s"
                LOGGER.warning(
                    f"[ibm_quantum] Skipping IBM backend: remaining runtime {remaining:.1f}s < {threshold:.0f}s"
                )
                return False
            
            # Discover all available backends and store them sorted by qubit count
            try:
                all_backends = service.backends(operational=True, simulator=False)
                self.ibm_backends = []
                
                for backend in all_backends:
                    try:
                        num_qubits = backend.num_qubits
                        backend_name = backend.name
                        pending_jobs = None
                        try:
                            status = backend.status()
                            pending_jobs = getattr(status, "pending_jobs", None)
                        except Exception:
                            pending_jobs = None

                        self.ibm_backends.append({
                            'backend': backend,
                            'name': backend_name,
                            'num_qubits': num_qubits,
                            'pending_jobs': pending_jobs
                        })
                        pj_str = "n/a" if pending_jobs is None else pending_jobs
                        LOGGER.info(f"[ibm_quantum] Found backend: {backend_name} ({num_qubits} qubits, pending={pj_str})")
                    except Exception as be:
                        LOGGER.debug(f"Error checking backend: {be}")
                        continue
                
                # Sort backends by qubit count (largest first) for optimal selection
                self.ibm_backends.sort(key=lambda x: x['num_qubits'], reverse=True)
                
                if self.ibm_backends:
                    # Pick two least busy backends as preferred
                    candidates = [b for b in self.ibm_backends if b.get('pending_jobs') is not None]
                    if candidates:
                        candidates.sort(key=lambda x: (x.get('pending_jobs', 10**9), -x.get('num_qubits', 0)))
                        self.ibm_backends_preferred = candidates[:2]
                    else:
                        # Fallback if pending_jobs isn't available
                        self.ibm_backends_preferred = self.ibm_backends[:2]

                    pref_str = ", ".join(
                        f"{b['name']}(pending={b.get('pending_jobs','n/a')})" for b in self.ibm_backends_preferred
                    )
                    if pref_str:
                        LOGGER.info(f"[ibm_quantum] Preferred backends (least busy): {pref_str}")

                    # Use the best preferred backend as default for reporting
                    default_backend = self.ibm_backends_preferred[0] if self.ibm_backends_preferred else self.ibm_backends[0]
                    self.qpus['ibm_quantum'].backend_name = default_backend['name']
                    self.qpus['ibm_quantum'].max_qubits = default_backend['num_qubits']
                    LOGGER.info(f"[ibm_quantum] {len(self.ibm_backends)} backends available, max qubits: {default_backend['num_qubits']}")

                    # Load Hardware Descriptor for Bandit (use default backend)
                    try:
                        self.qpus['ibm_quantum'].descriptor = IBMAdapter.from_backend(default_backend['backend'])
                        self.devices['ibm_backend'] = default_backend['backend']
                        LOGGER.info(f"[ibm_quantum] Loaded hardware descriptor for {default_backend['name']}")
                    except Exception as e:
                        LOGGER.warning(f"[ibm_quantum] Failed to create descriptor: {e}")
                else:
                    LOGGER.warning("No operational IBM backends found")
                    self.qpus['ibm_quantum'].backend_name = "none"
                    self.ibm_backends = []
                    
            except Exception as e:
                LOGGER.warning(f"Could not enumerate backends: {e}")
                self.qpus['ibm_quantum'].backend_name = "unknown"
                self.ibm_backends = []
            
            self.sessions['ibm_quantum'] = service
            self.qpus['ibm_quantum'].is_available = True
            self.qpus['ibm_quantum'].last_error = ""
            return True
            
        except Exception as e:
            LOGGER.error(f"IBM Quantum initialization failed: {e}")
            return False

    def _refresh_ibm_pending(self, backend_infos: List[Dict[str, Any]]) -> None:
        for info in backend_infos:
            try:
                status = info['backend'].status()
                pending = getattr(status, "pending_jobs", None)
                if pending is not None:
                    info['pending_jobs'] = pending
            except Exception:
                pass

    def _select_ibm_backend_info(self, num_qubits: int) -> Optional[Dict[str, Any]]:
        candidates = [
            b for b in getattr(self, 'ibm_backends_preferred', [])
            if b.get('num_qubits', 0) >= num_qubits
        ]
        if not candidates:
            candidates = [
                b for b in getattr(self, 'ibm_backends', [])
                if b.get('num_qubits', 0) >= num_qubits
            ]
        if not candidates:
            return None

        self._refresh_ibm_pending(candidates)

        def _key(b):
            pending = b.get('pending_jobs')
            if pending is None:
                pending = 10**9
            return (pending, b.get('num_qubits', 0))

        ordered = sorted(candidates, key=_key)
        if not ordered:
            return None

        best_pending = ordered[0].get('pending_jobs')
        tied = []
        for b in ordered:
            if b.get('pending_jobs') == best_pending:
                tied.append(b)
            else:
                break

        if len(tied) == 1:
            return tied[0]

        idx = self._ibm_rr_index % len(tied)
        self._ibm_rr_index += 1
        return tied[idx]
    
    def _init_local_simulator(self):
        """Initialize local MPS simulator (supports 100+ qubits)."""
        # Use matrix_product_state for large qubit counts
        simulator = AerSimulator(method='matrix_product_state')
        self.devices['local_mps_simulator'] = simulator
        
        # Register in QPU list so run_job can find it
        self.qpus['local_simulator'] = QPUConfig(
            name="Local MPS Simulator",
            provider="local",
            max_qubits=100,
            priority=99,
            is_available=True
        )
        LOGGER.info("[Fallback] MPS simulator ready (100+ qubits supported)")
    
    def get_available_qpus_for_size(self, num_qubits: int) -> List[str]:
        """
        Get list of all available QPUs that can handle the given problem size.
        Checks both availability status AND time windows.
        Returns QPU IDs sorted by runtime score (reliability, speed, window, fit).
        """
        self._refresh_ibm_usage_if_needed()
        attempted_opportunistic_refresh = False

        while True:
            hardware_available = any(
                q.is_available and not self._is_simulator_provider(q.provider)
                for q in self.qpus.values()
            )

            candidates: List[Tuple[float, int, str, Dict[str, float]]] = []
            skipped: Dict[str, str] = {}

            for qpu_id, qpu in self.qpus.items():
                if qpu.max_qubits < num_qubits:
                    skipped[qpu_id] = f"insufficient_qubits({qpu.max_qubits}<{num_qubits})"
                    continue
                if not qpu.is_available:
                    detail = qpu.last_error if qpu.last_error else "flagged unavailable"
                    skipped[qpu_id] = f"unavailable({detail[:60]})"
                    continue
                if self._is_simulator_provider(qpu.provider) and hardware_available:
                    skipped[qpu_id] = "simulator_reserved_for_fallback"
                    continue
                if not is_qpu_in_time_window(qpu):
                    skipped[qpu_id] = "outside_time_window"
                    continue
                if not self._has_sufficient_window(qpu):
                    remaining = _remaining_window_minutes_sgt(qpu)
                    required = self.min_window_remaining_minutes
                    if qpu.provider.startswith('braket') and not self._is_simulator_provider(qpu.provider):
                        required = max(required, self.min_aws_window_remaining_minutes)
                        if qpu.successful_jobs > 0:
                            avg_job_minutes = (qpu.total_time / max(1, qpu.successful_jobs)) / 60.0
                            adaptive_required = min(180.0, (2.0 * avg_job_minutes) + 5.0)
                            required = max(required, adaptive_required)
                    if remaining is None:
                        skipped[qpu_id] = f"window_ending_soon(required>={required:.1f}m)"
                    else:
                        skipped[qpu_id] = f"window_ending_soon({remaining:.1f}m<{required:.1f}m)"
                    continue
                if not self._check_qpu_available(qpu_id):
                    skipped[qpu_id] = "provider_status_offline"
                    continue

                score, parts = self._qpu_runtime_score(qpu_id, qpu, num_qubits)
                candidates.append((score, qpu.total_jobs, qpu_id, parts))

            # Sort by score (high to low), then load (low to high)
            candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
            available_ids = [qpu_id for _, _, qpu_id, _ in candidates]

            hardware_ids = [
                qpu_id for _, _, qpu_id, _ in candidates
                if not self._is_simulator_provider(self.qpus[qpu_id].provider)
            ]

            should_try_opportunistic = (
                not attempted_opportunistic_refresh
                and HAS_BRAKET
                and len(hardware_ids) <= 1
                and (time.time() - self._last_opportunistic_refresh) >= self._opportunistic_refresh_interval
            )
            if should_try_opportunistic:
                attempted_opportunistic_refresh = True
                self._last_opportunistic_refresh = time.time()
                self._refresh_braket_backends(
                    include_unavailable=True,
                    in_window_only=True,
                    reason=f"low-hardware-pool({len(hardware_ids)} ready for {num_qubits}q)",
                )
                continue

            self._log_scheduler_snapshot(num_qubits, candidates, skipped)
            return available_ids
    
    def assign_problems_to_qpus(self, problem_sizes: List[int]) -> List[str]:
        """
        Assign multiple problems to different QPUs for MAXIMUM PARALLELISM.
        Uses strict round-robin to ensure problems go to DIFFERENT QPUs.
        
        Args:
            problem_sizes: List of qubit counts for each problem
            
        Returns:
            List of QPU IDs (one per problem), or 'local_fallback' if no hardware
        """
        assignments = []
        
        # Get ALL capable QPUs for the problem size (all same size)
        if problem_sizes:
            capable_qpus = self.get_available_qpus_for_size(problem_sizes[0])
        else:
            capable_qpus = []
        
        if not capable_qpus:
            # No hardware available - all go to local fallback
            return ['local_fallback'] * len(problem_sizes)
        
        # Log available QPUs
        LOGGER.info(f"  [QPU Pool] {len(capable_qpus)} QPUs available: {capable_qpus}")

        # Weighted fair assignment by runtime score: use high-quality QPUs more,
        # while still keeping all available QPUs active.
        score_map: Dict[str, float] = {}
        for qpu_id in capable_qpus:
            qpu = self.qpus[qpu_id]
            score, _ = self._qpu_runtime_score(qpu_id, qpu, problem_sizes[0])
            score_map[qpu_id] = max(0.05, score)
        LOGGER.info(
            "  [QPU Weights] "
            + ", ".join(f"{qpu_id}:{score_map[qpu_id]:.2f}" for qpu_id in capable_qpus)
        )

        assigned_count = {qpu_id: 0 for qpu_id in capable_qpus}
        for _ in problem_sizes:
            qpu_id = min(
                capable_qpus,
                key=lambda q: (
                    assigned_count[q] / score_map[q],
                    assigned_count[q],
                    self.qpus[q].total_jobs,
                ),
            )
            assignments.append(qpu_id)
            assigned_count[qpu_id] += 1
        
        return assignments
    
    def select_qpu_for_problem(self, num_qubits: int, exclude: List[str] = None) -> Optional[str]:
        """
        Select an available QPU for a problem of given size.
        Uses load balancing (least jobs first) instead of fixed priority.
        
        Args:
            num_qubits: Required qubit count
            exclude: List of QPU IDs to exclude (e.g., ones that just failed)
            
        Returns:
            QPU ID or None if no hardware available
        """
        exclude = exclude or []
        available = self.get_available_qpus_for_size(num_qubits)
        
        # Filter out excluded QPUs
        available = [q for q in available if q not in exclude]
        
        if available:
            return available[0]  # Already sorted by load
        
        # No hardware QPU available
        return None
    
    def _check_qpu_available(self, qpu_id: str) -> bool:
        """Check if a QPU is currently available (quick check)."""
        qpu = self.qpus[qpu_id]
        
        # Don't re-check too frequently
        if time.time() - qpu.last_check < 60:
            return qpu.is_available
        
        qpu.last_check = time.time()
        
        if qpu.provider.startswith('braket'):
            try:
                device = self.devices.get(qpu_id)
                if device and device.status == 'ONLINE':
                    qpu.is_available = True
                    return True
            except:
                pass
            qpu.is_available = False
            return False
            
        elif qpu.provider == 'ibm':
            self._refresh_ibm_usage_if_needed()
            # For IBM, we'll trust it's available if initialized
            return qpu.is_available
        
        elif qpu.provider == 'local':
            return True
        
        return qpu.is_available

    def _ensure_ibm_schedulable(self, context: str = "") -> None:
        """
        Enforce current IBM scheduling availability before any submission.

        This prevents in-flight optimization loops from continuing to submit
        after usage checks mark IBM unavailable.
        """
        self._refresh_ibm_usage_if_needed()
        qpu = self.qpus.get('ibm_quantum')
        if qpu is None:
            raise RuntimeError("IBM QPU entry missing from scheduler state")
        if not qpu.is_available:
            detail = qpu.last_error or "IBM unavailable"
            if context:
                raise RuntimeError(f"IBM unavailable for scheduling ({context}): {detail}")
            raise RuntimeError(f"IBM unavailable for scheduling: {detail}")
    
    def refresh_credentials_if_needed(self):
        """Refresh AWS credentials if they're nearing expiry."""
        self._refresh_ibm_usage_if_needed()
        if time.time() - self.last_credential_refresh > self.credential_refresh_interval:
            self._refresh_braket_backends(
                include_unavailable=True,
                in_window_only=False,
                reason="credential-refresh",
            )
            self.last_credential_refresh = time.time()
    
    def run_job(self, qpu_id: str, circuit: Any, shots: int,
                timeout: Optional[float] = None, max_retries: int = 3) -> Tuple[Dict, float, str]:
        """
        Run a quantum job on specified QPU with retry logic.
        
        If timeout is provided, failover to local simulator on timeout.
        
        Returns:
            (result_counts, job_time, actual_qpu_used)
        """
        self.refresh_credentials_if_needed()
        self._refresh_ibm_usage_if_needed()
        
        qpu = self.qpus[qpu_id]
        start_time = time.time()
        last_error = None
        timed_out = False
        
        for attempt in range(max_retries):
            try:
                if qpu.provider.startswith('braket'):
                    if not self._has_sufficient_window(qpu):
                        raise RuntimeError("Availability window ending soon")
                    result = self._run_braket_job(qpu_id, circuit, shots, timeout)
                elif qpu.provider == 'ibm':
                    result = self._run_ibm_job(circuit, shots, timeout)
                elif qpu.provider == 'local':
                    result = self._run_local_job(circuit, shots)
                else:
                    raise ValueError(f"Unknown provider: {qpu.provider}")
                
                job_time = time.time() - start_time
                
                # Update stats
                with self.lock:
                    qpu.total_jobs += 1
                    qpu.successful_jobs += 1
                    qpu.total_time += job_time
                    self.job_log.append({
                        'timestamp': datetime.now().isoformat(),
                        'qpu': qpu_id,
                        'shots': shots,
                        'time': job_time,
                        'success': True
                    })
                
                return result, job_time, qpu_id
                
            except Exception as e:
                last_error = str(e)
                err_lower = last_error.lower()
                LOGGER.warning(f"[{qpu_id}] Attempt {attempt+1}/{max_retries} failed: {e}")
                
                # Mark QPU as unavailable if persistent error or TIMEOUT (QPU likely in queue)
                if any(x in err_lower for x in ['expiredtoken', 'offline', 'timeout', 'timed out']):
                    qpu.is_available = False
                    qpu.last_error = str(e)[:100]
                    LOGGER.warning(f"[{qpu_id}] Marked UNAVAILABLE due to: {str(e)[:60]}")
                    if 'timeout' in err_lower or 'timed out' in err_lower:
                        timed_out = True
                    break  # Don't retry, failover immediately
                
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        # All retries failed - update stats and try fallback
        with self.lock:
            qpu.failed_jobs += 1
            self.job_log.append({
                'timestamp': datetime.now().isoformat(),
                'qpu': qpu_id,
                'shots': shots,
                'time': time.time() - start_time,
                'success': False,
                'error': last_error
            })
        
        if timed_out:
            raise TimeoutError(last_error)

        # Try fallback QPU
        fallback_qpu = self._get_fallback_qpu(qpu_id, circuit)
        if fallback_qpu:
            LOGGER.info(f"[{qpu_id}] Falling back to {fallback_qpu}")
            return self.run_job(fallback_qpu, circuit, shots, timeout, max_retries=1)
        
        raise RuntimeError(f"All QPUs failed for job. Last error: {last_error}")
    
    def _run_braket_job(self, qpu_id: str, circuit: Any, shots: int, timeout: Optional[float]) -> Dict:
        """Run a job on AWS Braket device."""
        device = self.devices[qpu_id]
        
        # Submit task
        task = device.run(circuit, shots=int(shots))
        task_id = task.id
        
        LOGGER.info(f"    [{qpu_id}] Task submitted: {task_id}")
        
        # Wait for completion
        terminal_states = ["COMPLETED", "FAILED", "CANCELLED"]
        start_time = time.time()
        
        while True:
            self._refresh_ibm_usage_if_needed()
            status = task.state()
            elapsed = time.time() - start_time
            
            if status in terminal_states:
                break
            
            if timeout is not None and elapsed > timeout:
                raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
            
            time.sleep(15)
        
        if status == "COMPLETED":
            result = task.result()
            LOGGER.info(f"    [{qpu_id}] Task completed in {elapsed:.1f}s")
            return result.measurement_counts
        else:
            raise RuntimeError(f"Task {task_id} ended with status: {status}")
    
    def _run_ibm_job(self, circuit: Any, shots: int, timeout: Optional[float]) -> Dict:
        """Run a job on IBM Quantum using least-busy backend selection."""
        self._ensure_ibm_schedulable(context="run_job pre-submit")
        service = self.sessions['ibm_quantum']
        num_qubits = circuit.num_qubits

        # Select backend based on least pending jobs (among preferred backends)
        backend = None
        backend_name = "unknown"
        backend_info = self._select_ibm_backend_info(num_qubits)
        if backend_info:
            backend = backend_info['backend']
            backend_name = backend_info.get('name', getattr(backend, 'name', 'unknown'))
            pending = backend_info.get('pending_jobs', 'n/a')
            LOGGER.info(
                f"    [ibm_quantum] Selected backend (least busy): {backend_info['name']} "
                f"({backend_info['num_qubits']} qubits, pending={pending})"
            )

        # Fallback: if no suitable backend found in cache, query service
        if backend is None:
            backends = service.backends(operational=True, simulator=False, min_num_qubits=num_qubits)
            if backends:
                scored = []
                for b in backends:
                    try:
                        pending = b.status().pending_jobs
                    except Exception:
                        pending = None
                    scored.append((pending if pending is not None else 10**9, b.num_qubits, b))
                scored.sort(key=lambda x: (x[0], x[1]))
                backend = scored[0][2]
                backend_name = getattr(backend, "name", "unknown")
                LOGGER.info(
                    f"    [ibm_quantum] Using backend (fallback): {backend.name} "
                    f"(pending={scored[0][0]})"
                )
            else:
                raise Exception(f"No IBM backend found with >= {num_qubits} qubits")

        # Submit job (SamplerV2 used directly, no context manager)
        from qiskit_ibm_runtime import SamplerV2
        sampler = SamplerV2(backend)

        self._ensure_ibm_schedulable(context=f"run_job submit on {backend_name}")
        job = sampler.run([circuit], shots=shots)
        job_id = getattr(job, "job_id", None)
        if callable(job_id):
            job_id = job_id()
        if job_id:
            LOGGER.info(f"    [IBM] Job submitted ({backend_name}): {job_id}")
        job_start = time.time()
        try:
            if timeout is None:
                result = job.result()
            else:
                result = job.result(timeout=timeout)
        except FutureTimeoutError as e:
            raise TimeoutError(f"IBM job timed out after {timeout}s") from e
        elapsed = time.time() - job_start
        if job_id:
            LOGGER.info(f"    [IBM] Job completed on {backend_name} in {elapsed:.1f}s: {job_id}")
        
        # Extract counts - handle different SamplerV2 result formats
        pub_result = result[0]
        if hasattr(pub_result.data, 'meas'):
            counts = pub_result.data.meas.get_counts()
        elif hasattr(pub_result.data, 'c'):
            counts = pub_result.data.c.get_counts()
        else:
            # Fallback: iterate through data attributes
            counts = {}
            for attr in dir(pub_result.data):
                if not attr.startswith('_'):
                    try:
                        data_obj = getattr(pub_result.data, attr)
                        if hasattr(data_obj, 'get_counts'):
                            counts = data_obj.get_counts()
                            break
                    except:
                        pass
        
        return counts
    
    def _run_local_job(self, circuit: Any, shots: int) -> Dict:
        """Run a job on local simulator with noise model."""
        simulator = self.devices['local_mps_simulator']
        
        # Add measurements if not present
        if circuit.num_clbits == 0:
            circuit.measure_all()
        
        # Run simulation
        job = simulator.run(circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        LOGGER.info(f"    [local_simulator] Completed {shots} shots")
        return counts
    
    def _get_fallback_qpu(self, failed_qpu: str, circuit: Any) -> Optional[str]:
        """Get a fallback QPU for a failed job."""
        # Determine qubit count
        if hasattr(circuit, 'qubit_count'):
            num_qubits = circuit.qubit_count  # Braket
        elif hasattr(circuit, 'num_qubits'):
            num_qubits = circuit.num_qubits  # Qiskit
        else:
            return None

        hardware_available = any(
            q.is_available and not self._is_simulator_provider(q.provider)
            for q in self.qpus.values()
        )

        # Find alternative QPU and choose best by runtime score
        fallback_candidates: List[Tuple[float, int, str]] = []
        for qpu_id, qpu in self.qpus.items():
            if qpu_id != failed_qpu and qpu.is_available and qpu.max_qubits >= num_qubits:
                if self._is_simulator_provider(qpu.provider) and hardware_available:
                    continue
                if not is_qpu_in_time_window(qpu):
                    continue
                if not self._has_sufficient_window(qpu):
                    continue
                score, _ = self._qpu_runtime_score(qpu_id, qpu, num_qubits)
                fallback_candidates.append((score, qpu.total_jobs, qpu_id))

        if fallback_candidates:
            fallback_candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
            return fallback_candidates[0][2]

        return None
    
    def get_status_report(self) -> str:
        """Get a formatted status report of all hardware QPUs."""
        now_sgt = datetime.now(SGT)
        weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        lines = [
            "\n" + "="*80, 
            f"MULTI-QPU STATUS REPORT @ {now_sgt.strftime('%Y-%m-%d %H:%M SGT')}",
            "="*80
        ]
        
        for qpu_id, qpu in self.qpus.items():
            # Check both availability flag AND time window
            in_window = is_qpu_in_time_window(qpu)
            
            if qpu.is_available and in_window:
                status = "[OK] ONLINE "
            elif qpu.is_available and not in_window:
                status = "[~] OFFLINE (outside time window)"
            else:
                status = "[X] OFFLINE "
            
            # For IBM, show actual backend name
            if qpu.provider == 'ibm' and qpu.backend_name:
                name_str = f"{qpu.name} ({qpu.backend_name})"
            else:
                name_str = qpu.name
            
            lines.append(f"  {name_str:<35} | {status:<30} | {qpu.max_qubits:>3}q | Jobs: {qpu.total_jobs}")
            if not qpu.is_available and qpu.last_error:
                lines.append(f"    +-- Error: {qpu.last_error[:60]}")
            if qpu.availability_windows_sgt_by_weekday:
                lines.append("    +-- Availability (SGT):")
                for day_idx in range(7):
                    windows = qpu.availability_windows_sgt_by_weekday.get(day_idx)
                    if not windows:
                        continue
                    win_str = ", ".join(
                        f"{w[0]:02d}:{w[1]:02d}-{w[2]:02d}:{w[3]:02d}" for w in windows
                    )
                    lines.append(f"       {weekday_names[day_idx]}: {win_str}")
            elif qpu.availability_windows_sgt is not None:
                parts = []
                for w in qpu.availability_windows_sgt:
                    if len(w) == 2:
                        parts.append(f"{w[0]:02d}:00-{w[1]:02d}:00")
                    elif len(w) == 4:
                        parts.append(f"{w[0]:02d}:{w[1]:02d}-{w[2]:02d}:{w[3]:02d}")
                win_str = ", ".join(parts)
                lines.append(f"    +-- Availability (SGT): {win_str}")
        
        # Add note about fallback
        if self.local_sim_available:
            lines.append("-"*80)
            lines.append("  [Fallback] Local MPS Simulator available (100+ qubits) - used if all QPUs fail")
        
        lines.append("="*80)
        return "\n".join(lines)


# ============================================================
# KNAPSACK SOLVER WITH MULTI-QPU SUPPORT
# ============================================================
class MultiQPUKnapsackSolver:
    """
    Solves knapsack subproblems using the best available QPU.
    Implements probability-based solution selection with smart fallback.
    """
    
    def __init__(self, qpu_manager: MultiQPUManager, shots: int = 100, depth: int = 1,
                 bandit_model: str = "configuration_bandit.pkl"):
        self.qpu_manager = qpu_manager
        self.shots = shots
        self.depth = depth
        self.gate_limit = 20000

        # Load trained ConfigurationBandit
        try:
            self.bandit = ConfigurationBandit.load(bandit_model)
            LOGGER.info(f"[Bandit] Loaded ConfigurationBandit from {bandit_model}")
        except Exception as e:
            LOGGER.warning(f"[Bandit] Could not load bandit ({e}). Using default config.")
            self.bandit = None

    def _gate_budget_for_config(
        self,
        qpu: QPUConfig,
        problem_size: int,
        config: Dict,
        descriptor: Optional[HardwareDescriptor] = None,
    ) -> Tuple[bool, int, float, float]:
        target_desc = descriptor if descriptor is not None else qpu.descriptor
        if target_desc is not None:
            conn, diam = _get_topology_metrics(target_desc)
        else:
            # Conservative fallback when descriptor is unavailable.
            conn = max(1.5, min(float(problem_size) / 8.0, 3.0))
            diam = max(4.0, min(float(problem_size) / 2.0, 20.0))

        safe, est_gates, _ = gate_budget_safe(
            problem_size=int(problem_size),
            strategy=config["entanglement"],
            depth=int(config["depth"]),
            connectivity=float(conn),
            diameter=float(diam),
            placement=config["placement"],
            gate_limit=self.gate_limit,
        )
        return safe, int(est_gates), float(conn), float(diam)

    def _consult_bandit(self, qpu: QPUConfig, problem_size: float) -> Dict:
        """Ask the bandit for the best configuration for this hardware/problem size."""
        safe_default = {"placement": "dense", "entanglement": "linear", "depth": 1 if problem_size >= 32 else max(1, min(self.depth, 2))}

        if qpu.descriptor is not None:
            avg_err = float(np.mean(qpu.descriptor.node_features[:, 2])) if qpu.descriptor.node_features.size else 0.0
            conn, diam = _get_topology_metrics(qpu.descriptor)
        else:
            avg_err = 0.01
            conn, diam = 2.0, 8.0

        if not self.bandit:
            return safe_default

        try:
            state = GlobalState(
                hamiltonian_complexity=float(problem_size),
                hardware_avg_error=avg_err,
                hardware_connectivity=float(conn),
                hardware_diameter=float(diam),
                problem_size=float(problem_size),
            )

            ranked_actions = self.bandit.rank_actions(state) if hasattr(self.bandit, "rank_actions") else [self.bandit.select_action(state)]
            if not ranked_actions:
                return safe_default

            top_idx = ranked_actions[0]
            for rank_pos, idx in enumerate(ranked_actions):
                candidate = self.bandit.actions[idx]
                is_safe, est_gates, _, _ = self._gate_budget_for_config(qpu, int(problem_size), candidate)
                if is_safe:
                    if rank_pos > 0:
                        top = self.bandit.actions[top_idx]
                        LOGGER.info(
                            "  [Bandit] Gate-budget override: "
                            f"{top['placement']}-{top['entanglement']}-D{top['depth']} -> "
                            f"{candidate['placement']}-{candidate['entanglement']}-D{candidate['depth']} "
                            f"(est_gates={est_gates}, limit={self.gate_limit})"
                        )
                    return candidate

            LOGGER.warning(
                f"  [Bandit] All 27 actions exceed gate budget for N={int(problem_size)} "
                f"on {qpu.name}; using safe default."
            )
            return safe_default
        except Exception as e:
            LOGGER.warning(f"[Bandit] Query failed: {e}")
            return safe_default

    def _try_local_fallback(self, profits: List[float], weights: List[float], capacity: int,
                             filtered_indices: List[int], customer_indices: List[int],
                             reduced_costs: List[float], qubit_stats: Dict,
                             num_items: int, reason: str) -> Optional[Tuple[List[int], float, Dict, Dict, str]]:
        if not getattr(self.qpu_manager, "local_sim_available", False):
            return None
        try:
            LOGGER.info(f"  [FALLBACK] {reason} Using local noise simulator for {num_items} qubits")
            result_counts = self._run_local_vqe(profits, weights, capacity)

            qubit_stats['vqe_executed'] = True
            qubit_stats['qpu_used'] = 'local_mps_simulator'

            chosen_indices, zk, item_probs = self._process_results_with_probability(
                result_counts, profits, weights, capacity,
                filtered_indices, customer_indices, reduced_costs
            )
            return chosen_indices, zk, qubit_stats, item_probs, 'local_mps_simulator'
        except Exception as e:
            LOGGER.error(f"  Local simulator also failed: {e}")
            return None
    
    def solve(self, capacity: int, demands: List[float], reduced_costs: List[float],
              customer_indices: List[int], force_all_items: bool = True,
              preferred_qpu: str = None) -> Tuple[List[int], float, Dict, Dict, str]:
        """
        Solve knapsack problem on best available QPU.
        
        Args:
            preferred_qpu: Pre-assigned QPU to try first (for load distribution)
            
        Returns:
            chosen_indices: Selected customer indices
            zk: Total reduced cost
            qubit_stats: Statistics about qubit usage
            item_probs: Probability of each item being selected (for heatmap)
            qpu_used: Which QPU was used
        """
        # Prepare problem
        profits = []
        weights = []
        filtered_indices = []
        
        if force_all_items:
            max_cost = max(abs(c) for c in reduced_costs) if reduced_costs else 1.0
            for idx, cost in enumerate(reduced_costs):
                profit = max_cost - cost + 1.0
                profits.append(profit)
                weights.append(demands[idx])
                filtered_indices.append(customer_indices[idx])
        else:
            for idx, cost in enumerate(reduced_costs):
                profit = -cost
                if profit > 0:
                    profits.append(profit)
                    weights.append(demands[idx])
                    filtered_indices.append(customer_indices[idx])
        
        num_items = len(profits)
        item_probs = {idx: 0.0 for idx in customer_indices}
        chosen_indices = []
        zk = 0.0
        
        # Qubit stats
        num_slack_qubits = math.floor(math.log2(capacity)) + 1 if capacity > 0 else 0
        qubit_stats = {
            'num_items': num_items,
            'unbalanced_qubits': num_items,
            'slack_qubits': num_items + num_slack_qubits,
            'vqe_executed': False,
            'qpu_used': None
        }
        
        if num_items < 2:
            # Trivial case
            if num_items == 1 and profits[0] > 0:
                chosen_indices = [filtered_indices[0]]
                zk = reduced_costs[customer_indices.index(filtered_indices[0])]
                item_probs[filtered_indices[0]] = 0.9
            return chosen_indices, zk, qubit_stats, item_probs, 'classical'
        
        # Build list of QPUs to try (preferred first, then others, then local fallback)
        qpus_to_try = []
        failed_qpus = []
        
        # Add preferred QPU first if it's a hardware QPU
        if preferred_qpu and preferred_qpu != 'local_fallback':
            qpus_to_try.append(preferred_qpu)
        
        # Add all other available hardware QPUs
        for qpu_id in self.qpu_manager.get_available_qpus_for_size(num_items):
            if qpu_id not in qpus_to_try:
                qpus_to_try.append(qpu_id)
        
        # Try each QPU in order until one succeeds
        actual_qpu_used = None
        result_counts = None

        for qpu_id in qpus_to_try:
            try:
                qpu = self.qpu_manager.qpus[qpu_id]
                config = self._consult_bandit(qpu, num_items)
                is_safe_cfg, est_cfg_gates, _, _ = self._gate_budget_for_config(qpu, num_items, config)
                LOGGER.info(
                    f"  [VQE] {num_items} qubits -> {qpu.name} | "
                    f"{config['placement'].upper()}-{config['entanglement'].upper()}-D{config['depth']} "
                    f"| est_gates={est_cfg_gates}/{self.gate_limit}"
                )
                if not is_safe_cfg:
                    raise RuntimeError(
                        f"Estimated gate count {est_cfg_gates} exceeds limit {self.gate_limit} "
                        f"for config {config} on {qpu.name}."
                    )

                # Build circuit based on QPU type
                if qpu.provider.startswith('braket'):
                    result_counts = self._run_braket_vqe(qpu_id, profits, weights, capacity, config)
                elif qpu.provider == 'ibm':
                    result_counts = self._run_ibm_vqe(profits, weights, capacity, config)
                else:
                    result_counts = self._run_local_vqe(profits, weights, capacity)
                
                actual_qpu_used = qpu_id
                qubit_stats['vqe_executed'] = True
                qubit_stats['qpu_used'] = actual_qpu_used
                qubit_stats['config'] = config
                
                # Process results with probability-based selection
                chosen_indices, zk, item_probs = self._process_results_with_probability(
                    result_counts, profits, weights, capacity,
                    filtered_indices, customer_indices, reduced_costs
                )
                
                # Success - return immediately
                return chosen_indices, zk, qubit_stats, item_probs, actual_qpu_used
                
            except TimeoutError as e:
                LOGGER.warning(f"  [TIMEOUT] {qpu.name} timed out: {e}")
                local_result = self._try_local_fallback(
                    profits, weights, capacity,
                    filtered_indices, customer_indices, reduced_costs,
                    qubit_stats, num_items,
                    reason="QPU timed out."
                )
                if local_result is not None:
                    return local_result
                failed_qpus.append(qpu_id)
                self.qpu_manager.qpus[qpu_id].is_available = False
                continue
            except Exception as e:
                LOGGER.warning(f"  [FAILOVER] {qpu.name} failed: {e}, trying next QPU...")
                failed_qpus.append(qpu_id)
                # Mark as unavailable temporarily
                self.qpu_manager.qpus[qpu_id].is_available = False
                continue  # Try next QPU

        # All hardware QPUs failed - use local MPS simulator as fallback
        hardware_available = any(
            q.is_available and q.provider != 'local'
            for q in self.qpu_manager.qpus.values()
        )
        if not hardware_available:
            local_result = self._try_local_fallback(
                profits, weights, capacity,
                filtered_indices, customer_indices, reduced_costs,
                qubit_stats, num_items,
                reason="No hardware QPU available."
            )
            if local_result is not None:
                return local_result
        
        # Complete failure - return empty
        LOGGER.error(f"All QPUs failed for {num_items} qubits - returning empty solution")
        return chosen_indices, zk, qubit_stats, item_probs, 'all_failed'
    
    def _run_braket_vqe(self, qpu_id: str, profits: List[float], weights: List[float],
                        capacity: int, config: Dict) -> Dict:
        """Run VQE on AWS Braket with bandit-selected placement/entanglement/depth."""
        num_qubits = len(profits)
        qpu = self.qpu_manager.qpus[qpu_id]
        is_safe_cfg, est_cfg_gates, _, _ = self._gate_budget_for_config(qpu, num_qubits, config)
        if not is_safe_cfg:
            raise RuntimeError(
                f"Gate budget exceeded before execution: est_gates={est_cfg_gates}, limit={self.gate_limit}"
            )

        # Placement (physical qubits)
        if qpu.descriptor is None:
            LOGGER.warning(f"[{qpu_id}] No HardwareDescriptor found; using sequential layout.")
            physical_qubits = list(range(num_qubits))
        else:
            placer = SmartPlacement(qpu.descriptor)
            physical_qubits = placer.get_best_layout(num_qubits, strategy=config['placement'])

        # Entanglement map (logical -> physical)
        ent_pairs_logical = get_entanglement_map(list(range(num_qubits)), strategy=config['entanglement'])
        l2p = {i: physical_qubits[i] for i in range(num_qubits)}
        ent_pairs = [(l2p[u], l2p[v]) for u, v in ent_pairs_logical]

        # Build hardware-efficient ansatz
        circuit, params = self._build_braket_ansatz(physical_qubits, ent_pairs, config['depth'])

        self._log_circuit_details(
            label=f"{qpu.name}",
            framework="braket",
            num_qubits=num_qubits,
            config=config,
            layout=physical_qubits,
            ent_pairs=ent_pairs,
            params=params,
            circuit=circuit,
            estimated_gate_count=est_cfg_gates,
        )
        
        # Simple COBYLA optimization
        from scipy.optimize import minimize
        
        best_counts = {}
        best_cost = float('inf')
        
        def objective(param_values):
            nonlocal best_counts, best_cost
            
            # Bind parameters
            param_dict = {params[i].name: float(param_values[i]) for i in range(len(params))}
            bound_circuit = circuit.make_bound_circuit(param_dict)
            
            # Run on QPU
            counts, _, _ = self.qpu_manager.run_job(qpu_id, bound_circuit, self.shots)
            
            # Compute cost
            avg_cost = self._compute_avg_cost(counts, profits, weights, capacity)
            
            if avg_cost < best_cost:
                best_cost = avg_cost
                best_counts = counts
            
            return avg_cost
        
        # Optimize
        x0 = np.random.uniform(0, 2*np.pi, len(params))
        minimize(objective, x0, method='COBYLA', options={'maxiter': 10, 'rhobeg': 0.5})
        
        return best_counts
    
    def _build_braket_ansatz(self, layout: List[int], ent_pairs: List[Tuple[int, int]],
                             depth: int) -> Tuple[Circuit, List]:
        """Build hardware-efficient ansatz for Braket on physical qubits."""
        circuit = Circuit()
        params = []

        # Initial layer
        for i, q in enumerate(layout):
            p = FreeParameter(f"theta_{i}_0")
            params.append(p)
            circuit.ry(q, p)

        # Entangling layers
        for d in range(depth):
            # Entanglement
            for u, v in ent_pairs:
                circuit.cz(u, v)

            # Rotation layer
            for i, q in enumerate(layout):
                p = FreeParameter(f"theta_{i}_{d+1}")
                params.append(p)
                circuit.ry(q, p)

        return circuit, params
    
    def _run_ibm_vqe(self, profits: List[float], weights: List[float],
                     capacity: int, config: Dict) -> Dict:
        """Run VQE on IBM Quantum using bandit-selected configuration."""
        num_qubits = len(profits)
        
        try:
            self.qpu_manager._ensure_ibm_schedulable(context="vqe start")
            service = self.qpu_manager.sessions.get('ibm_quantum')
            if service is None:
                raise Exception("IBM Quantum service not initialized")
            
            # Select backend based on least pending jobs (among preferred backends)
            backend = None
            backend_info = self.qpu_manager._select_ibm_backend_info(num_qubits)
            if backend_info:
                backend = backend_info['backend']
                pending = backend_info.get('pending_jobs', 'n/a')
                LOGGER.info(
                    f"    [IBM] Selected backend (least busy): {backend_info['name']} "
                    f"({backend_info['num_qubits']} qubits, pending={pending})"
                )

            # Fallback: if no suitable backend found in cache, query service
            if backend is None:
                backends = service.backends(operational=True, simulator=False, min_num_qubits=num_qubits)
                if backends:
                    scored = []
                    for b in backends:
                        try:
                            pending = b.status().pending_jobs
                        except Exception:
                            pending = None
                        scored.append((pending if pending is not None else 10**9, b.num_qubits, b))
                    scored.sort(key=lambda x: (x[0], x[1]))
                    backend = scored[0][2]
                    LOGGER.info(
                        f"    [IBM] Using backend (fallback query): {backend.name} "
                        f"({backend.num_qubits} qubits, pending={scored[0][0]})"
                    )
                else:
                    raise Exception(f"No IBM backend found with >= {num_qubits} qubits")
            
            backend_name = backend.name
            
            # Build bandit-aware variational circuit
            from qiskit.circuit import Parameter
            from qiskit_ibm_runtime import SamplerV2
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

            # Refresh descriptor for selected backend (ensures matching topology)
            qpu = self.qpu_manager.qpus['ibm_quantum']
            descriptor = None
            try:
                descriptor = IBMAdapter.from_backend(backend)
                qpu.descriptor = descriptor
            except Exception as e:
                LOGGER.warning(f"    [IBM] Failed to refresh descriptor for {backend_name}: {e}")
                descriptor = qpu.descriptor

            if descriptor is None:
                raise RuntimeError("IBM HardwareDescriptor unavailable for placement")
            is_safe_cfg, est_cfg_gates, _, _ = self._gate_budget_for_config(
                qpu, num_qubits, config, descriptor=descriptor
            )
            if not is_safe_cfg:
                raise RuntimeError(
                    f"Gate budget exceeded before IBM compilation: est_gates={est_cfg_gates}, limit={self.gate_limit}"
                )

            placer = SmartPlacement(descriptor)
            physical_qubits = placer.get_best_layout(num_qubits, strategy=config['placement'])

            ent_pairs_logical = get_entanglement_map(list(range(num_qubits)), strategy=config['entanglement'])

            qc = QuantumCircuit(num_qubits, num_qubits)
            params = []

            # Initial layer
            for i in range(num_qubits):
                p = Parameter(f"theta_{i}_0")
                params.append(p)
                qc.ry(p, i)

            # Entangling layers
            for d in range(config['depth']):
                for u, v in ent_pairs_logical:
                    qc.cx(u, v)
                for i in range(num_qubits):
                    p = Parameter(f"theta_{i}_{d+1}")
                    params.append(p)
                    qc.ry(p, i)

            qc.measure_all()

            self._log_circuit_details(
                label=f"IBM {backend_name}",
                framework="qiskit",
                num_qubits=num_qubits,
                config=config,
                layout=physical_qubits,
                ent_pairs=ent_pairs_logical,
                params=params,
                circuit=qc,
                estimated_gate_count=est_cfg_gates,
            )

            # Transpile for target backend with initial layout
            pm = generate_preset_pass_manager(
                backend=backend,
                optimization_level=1,
                initial_layout=physical_qubits
            )
            isa_circuit = pm.run(qc)
            
            # Create SamplerV2 instance (NOT as context manager - SamplerV2 doesn't support it)
            sampler = SamplerV2(backend)
            
            # Simple VQE optimization with COBYLA
            best_counts = {}
            best_cost = float('inf')
            
            def objective(param_values):
                nonlocal best_counts, best_cost
                # Hard stop if usage checks disable IBM mid-optimization.
                self.qpu_manager._ensure_ibm_schedulable(
                    context=f"vqe iteration on {backend_name}"
                )
                
                # Bind parameters
                bound_circuit = isa_circuit.assign_parameters(
                    {params[i]: float(param_values[i]) for i in range(len(params))}
                )
                
                # Run on IBM hardware (SamplerV2 used directly, no context manager)
                job = sampler.run([bound_circuit], shots=self.shots)
                job_id = getattr(job, "job_id", None)
                if callable(job_id):
                    job_id = job_id()
                if job_id:
                    LOGGER.info(f"    [IBM] Job submitted ({backend_name}): {job_id}")

                job_start = time.time()
                try:
                    result = job.result()
                except FutureTimeoutError as e:
                    raise TimeoutError("IBM job timed out") from e
                if job_id:
                    LOGGER.info(
                        f"    [IBM] Job completed on {backend_name} in {time.time() - job_start:.1f}s: {job_id}"
                    )
                
                # Extract counts from SamplerV2 result
                pub_result = result[0]
                # SamplerV2 returns data in different format - try both 'meas' and 'c'
                if hasattr(pub_result.data, 'meas'):
                    counts = pub_result.data.meas.get_counts()
                elif hasattr(pub_result.data, 'c'):
                    counts = pub_result.data.c.get_counts()
                else:
                    # Fallback: iterate through data attributes
                    counts = {}
                    for attr in dir(pub_result.data):
                        if not attr.startswith('_'):
                            try:
                                data_obj = getattr(pub_result.data, attr)
                                if hasattr(data_obj, 'get_counts'):
                                    counts = data_obj.get_counts()
                                    break
                            except:
                                pass
                
                # Compute cost
                avg_cost = self._compute_avg_cost(counts, profits, weights, capacity)
                
                if avg_cost < best_cost:
                    best_cost = avg_cost
                    best_counts = counts
                
                return avg_cost
            
            # Optimize (fewer iterations for hardware)
            from scipy.optimize import minimize
            x0 = np.random.uniform(0, 2*np.pi, len(params))
            minimize(objective, x0, method='COBYLA', options={'maxiter': 10, 'rhobeg': 0.5})
            
            return best_counts
            
        except Exception as e:
            LOGGER.error(f"IBM VQE execution failed: {e}")
            raise
    
    def _run_local_vqe(self, profits: List[float], weights: List[float],
                       capacity: int) -> Dict:
        """Run VQE on local simulator with noise."""
        num_qubits = len(profits)
        
        # Build simple variational circuit
        from qiskit.circuit import Parameter
        
        qc = QuantumCircuit(num_qubits, num_qubits)
        params = []
        
        for q in range(num_qubits):
            p = Parameter(f"theta_{q}")
            params.append(p)
            qc.ry(p, q)
        
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
        
        qc.measure_all()
        
        # Run with random parameters
        param_values = np.random.uniform(0, 2*np.pi, len(params))
        bound_qc = qc.assign_parameters({params[i]: param_values[i] for i in range(len(params))})
        
        counts, _, _ = self.qpu_manager.run_job('local_simulator', bound_qc, self.shots)
        return counts

    def _log_circuit_details(self, label: str, framework: str, num_qubits: int, config: Dict,
                             layout: List[int], ent_pairs: List[Tuple[int, int]],
                             params: List[Any], circuit: Any,
                             estimated_gate_count: Optional[int] = None) -> None:
        layout_preview = layout if len(layout) <= 20 else layout[:20] + ["..."]
        ent_preview = ent_pairs[:10]

        depth_info = None
        gate_count = None
        try:
            if hasattr(circuit, "depth"):
                depth_attr = circuit.depth
                depth_info = depth_attr() if callable(depth_attr) else depth_attr
        except Exception:
            depth_info = None

        try:
            if hasattr(circuit, "size"):
                gate_count = circuit.size()
            elif hasattr(circuit, "instructions"):
                gate_count = len(circuit.instructions)
        except Exception:
            gate_count = None

        LOGGER.info(
            f"    [Circuit] {label} | qubits={num_qubits} "
            f"placement={config['placement']} ent={config['entanglement']} depth={config['depth']}"
        )
        LOGGER.info(f"    [Circuit] layout={layout_preview}")
        LOGGER.info(f"    [Circuit] entanglement_pairs={len(ent_pairs)} sample={ent_preview}")
        if estimated_gate_count is None:
            LOGGER.info(f"    [Circuit] params={len(params)} gates={gate_count} depth={depth_info} ({framework})")
        else:
            LOGGER.info(
                f"    [Circuit] params={len(params)} gates={gate_count} depth={depth_info} "
                f"est_gates={estimated_gate_count}/{self.gate_limit} ({framework})"
            )

    def _compute_avg_cost(self, counts: Dict, profits: List[float],
                          weights: List[float], capacity: int) -> float:
        """Compute average cost from measurement counts."""
        total_cost = 0.0
        total_shots = sum(counts.values())
        penalty = 100.0
        
        for bitstring, count in counts.items():
            bits = [int(b) for b in str(bitstring)]
            
            # Pad or truncate to match number of items
            while len(bits) < len(profits):
                bits.insert(0, 0)
            bits = bits[-len(profits):]
            
            total_profit = sum(p * b for p, b in zip(profits, bits))
            total_weight = sum(w * b for w, b in zip(weights, bits))
            
            cost = -total_profit
            if total_weight > capacity:
                cost += penalty * (total_weight - capacity)
            
            total_cost += cost * count
        
        return total_cost / max(total_shots, 1)
    
    def _process_results_with_probability(self, counts: Dict, profits: List[float],
                                           weights: List[float], capacity: int,
                                           filtered_indices: List[int],
                                           customer_indices: List[int],
                                           reduced_costs: List[float]) -> Tuple[List[int], float, Dict]:
        """
        Process VQE results using probability-based selection.
        Creates a heatmap of item selection probabilities.
        """
        item_probs = {idx: 0.0 for idx in customer_indices}
        total_shots = sum(counts.values())
        
        # Calculate per-item selection probability
        item_counts = {i: 0 for i in range(len(profits))}
        
        for bitstring, count in counts.items():
            # Handle bitstrings with spaces (e.g., "00 11" from multiple registers)
            sanitized_bitstring = str(bitstring).replace(" ", "")
            bits = [int(b) for b in sanitized_bitstring]
            while len(bits) < len(profits):
                bits.insert(0, 0)
            bits = bits[-len(profits):]
            
            for i, b in enumerate(bits):
                if b == 1:
                    item_counts[i] += count
        
        # Convert to probabilities
        for i, global_idx in enumerate(filtered_indices):
            prob = item_counts[i] / max(total_shots, 1)
            item_probs[global_idx] = prob
        
        # Find best feasible solution
        best_profit = -float('inf')
        best_bitstring = None
        
        for bitstring, count in counts.items():
            sanitized_bitstring = str(bitstring).replace(" ", "")
            bits = [int(b) for b in sanitized_bitstring]
            while len(bits) < len(profits):
                bits.insert(0, 0)
            bits = bits[-len(profits):]
            
            total_weight = sum(w * b for w, b in zip(weights, bits))
            total_profit = sum(p * b for p, b in zip(profits, bits))
            
            if total_weight <= capacity and total_profit > best_profit:
                best_profit = total_profit
                best_bitstring = bits
        
        # Extract selected items
        chosen_indices = []
        zk = 0.0
        
        if best_bitstring:
            for i, b in enumerate(best_bitstring):
                if b == 1:
                    global_idx = filtered_indices[i]
                    chosen_indices.append(global_idx)
                    orig_idx = customer_indices.index(global_idx)
                    zk += reduced_costs[orig_idx]
        
        return chosen_indices, zk, item_probs


# ============================================================
# CHECKPOINT / RESUME
# ============================================================
def save_checkpoint(path: str, data: Dict):
    """Save checkpoint to file."""
    data['timestamp'] = datetime.now().isoformat()
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    LOGGER.info(f"[Checkpoint] Saved: {len(data.get('completed_instances', []))} instances")


def load_checkpoint(path: str) -> Optional[Dict]:
    """Load checkpoint if exists."""
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            LOGGER.info(f"[Checkpoint] Loaded: {len(data.get('completed_instances', []))} instances")
            return data
        except Exception as e:
            LOGGER.warning(f"Failed to load checkpoint: {e}")
    return None


# ============================================================


def parse_bks_from_file(filepath):
    """Extract BKS from COMMENT line in .vrp file."""
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if 'COMMENT' in line.upper() and 'OPTIMAL' in line.upper():
                    match = re.search(r'(\d+)', line.split(':')[-1])
                    if match:
                        return int(match.group(1))
    except:
        pass
    return None


def check_feasibility(assignments, num_customers, demands, capacity):
    """Check solution feasibility."""
    visited = set()
    for route in assignments:
        route_demand = 0
        for cust in route:
            if cust == 0:
                continue
            if cust in visited:
                return False, f"Customer {cust} visited twice"
            visited.add(cust)
            route_demand += demands.get(cust, 0)
        if route_demand > capacity:
            return False, f"Capacity exceeded: {route_demand} > {capacity}"
    
    expected = set(range(1, num_customers + 1))
    missed = expected - visited
    if missed:
        return False, f"Missed {len(missed)} customers"
    
    return True, "OK"


# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================
def evaluate_instance_multi_qpu(env, policy, knapsack_solver: MultiQPUKnapsackSolver,
                                 torch_device, max_steps=100, vqe_interval=10):
    """
    Evaluate a single instance using multi-QPU quantum solving.
    """
    start_time = time.time()
    vqe_total_time = 0.0
    
    obs, info = env.reset()
    
    best_cost = float('inf')
    best_assignments = None
    best_step = 0
    total_steps = 0
    final_lb = 0.0
    
    # Quantum stats
    quantum_stats = {
        'total_circuits': 0,
        'qubit_sizes': [],
        'qpu_usage': {},
        'total_qpu_time': 0.0,
        'item_probs_history': []
    }
    
    # Heatmap for probability-based solution
    parser = env.parser
    num_vehicles = env.solver.num_vehicles
    customer_indices = list(range(1, parser.dimension))
    heatmap = {c: {k: 0.0 for k in range(num_vehicles)} for c in customer_indices}
    
    solver = env.solver
    
    for step in range(max_steps):
        total_steps = step + 1
        
        # Get policy action using obs from last step or reset
        batch = Batch.from_data_list([obs]).to(torch_device)
        
        with torch.no_grad():
            output = policy(batch)
            # Policy output varies - first element is always action_mean
            if isinstance(output, tuple):
                action_mean = output[0]
            else:
                action_mean = output
        
        action = action_mean.squeeze().cpu().numpy()
        obs, _, term, trunc, info = env.step(action)
        
        if 'lower_bound' in info:
            final_lb = info['lower_bound']
        
        # Run VQE at intervals
        if step % vqe_interval == 0:
            vqe_start = time.time()
            
            lambdas = env.lambdas.numpy()
            y = {i: [0] * num_vehicles for i in customer_indices}
            
            # Prepare all vehicle problems
            vehicle_problems = []
            for k in range(num_vehicles):
                reduced_costs = []
                demands_list = []
                for i, cust in enumerate(customer_indices):
                    d_ik = solver.dik.get((cust, k), 1000.0)
                    reduced_cost = d_ik - lambdas[i]
                    reduced_costs.append(reduced_cost)
                    demands_list.append(parser.demands.get(cust, 0))
                vehicle_problems.append({
                    'k': k,
                    'capacity': parser.capacity,
                    'demands': demands_list,
                    'reduced_costs': reduced_costs,
                    'customer_indices': customer_indices
                })
            
            # Calculate qubit size for each problem (all same for a given instance)
            # Qubit count = num_customers (15 for P-n16)
            problem_sizes = [len(customer_indices)] * num_vehicles
            
            # Pre-assign QPUs to problems for MAXIMUM DISTRIBUTION
            qpu_assignments = knapsack_solver.qpu_manager.assign_problems_to_qpus(problem_sizes)
            
            # Log the distribution
            qpu_dist = {}
            for q in qpu_assignments:
                qpu_dist[q] = qpu_dist.get(q, 0) + 1
            LOGGER.info(f"  [QPU Distribution] {dict(qpu_dist)}")
            
            # Solve ALL vehicles in PARALLEL across DIFFERENT QPUs
            def solve_vehicle_knapsack(problem, assigned_qpu):
                k = problem['k']
                chosen, zk, qstats, probs_k, qpu_used = knapsack_solver.solve(
                    problem['capacity'], problem['demands'], problem['reduced_costs'],
                    problem['customer_indices'], force_all_items=True,
                    preferred_qpu=assigned_qpu  # Pass the pre-assigned QPU
                )
                return {'k': k, 'chosen': chosen, 'probs': probs_k, 'qstats': qstats, 'qpu_used': qpu_used}
            
            # Use ThreadPoolExecutor for parallel QPU execution
            results_list = []
            with ThreadPoolExecutor(max_workers=min(num_vehicles, len(knapsack_solver.qpu_manager.qpus))) as executor:
                futures = [
                    executor.submit(solve_vehicle_knapsack, prob, qpu_assignments[i]) 
                    for i, prob in enumerate(vehicle_problems)
                ]
                for future in as_completed(futures):
                    try:
                        results_list.append(future.result())
                    except Exception as e:
                        LOGGER.warning(f"Vehicle VQE failed: {e}")
            
            # Process parallel results
            for result in results_list:
                k = result['k']
                for cust in result['chosen']:
                    y[cust][k] = 1
                
                # Update heatmap
                for cust_idx, prob in result['probs'].items():
                    heatmap[cust_idx][k] = max(heatmap[cust_idx][k], prob)
                
                # Track stats
                qstats = result['qstats']
                if qstats.get('vqe_executed', False):
                    quantum_stats['total_circuits'] += 1
                    quantum_stats['qubit_sizes'].append(qstats['unbalanced_qubits'])
                    qpu_name = result['qpu_used'] or 'unknown'
                    quantum_stats['qpu_usage'][qpu_name] = quantum_stats['qpu_usage'].get(qpu_name, 0) + 1
            
            vqe_batch_time = time.time() - vqe_start
            vqe_total_time += vqe_batch_time
            quantum_stats['total_qpu_time'] += vqe_batch_time
            
            LOGGER.info(f"  Step {step}: {len(results_list)} vehicles solved in PARALLEL in {vqe_batch_time:.1f}s")
            
            # Try quantum repair
            if hasattr(solver, 'build_feasible_solution_quantum'):
                try:
                    q_assign, q_cost_raw, stats = solver.build_feasible_solution_quantum(heatmap, alpha=0.6)
                    q_route_cost = env.router.route(q_assign, use_ortools=False)
                    
                    if q_route_cost < best_cost:
                        best_cost = q_route_cost
                        best_assignments = [list(r) for r in q_assign]
                        best_step = step + 1
                        LOGGER.info(f"  [+] Quantum repair improved: {q_route_cost:.1f}")
                except:
                    pass
        
        # Track best from environment
        try:
            current_cost = info.get('primal_cost', float('inf'))
            if current_cost < best_cost:
                best_cost = current_cost
                best_assignments = solver.last_primal_assignments
                best_step = step + 1
        except:
            pass
        
        if term or trunc:
            break
    
    # Final metrics
    solve_time = time.time() - start_time
    is_feasible = True
    feasibility_reason = "OK"
    
    if best_assignments:
        is_feasible, feasibility_reason = check_feasibility(
            best_assignments, parser.dimension - 1, parser.demands, parser.capacity
        )
    
    return {
        'cost': best_cost,
        'solve_time': solve_time,
        'vqe_time': vqe_total_time,
        'num_steps': total_steps,
        'best_step': best_step,
        'assignments': best_assignments,
        'is_feasible': is_feasible,
        'feasibility_reason': feasibility_reason,
        'num_customers': parser.dimension - 1,
        'num_vehicles': num_vehicles,
        'lower_bound': final_lb,
        'quantum_stats': quantum_stats
    }


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-QPU Quantum Evaluation for CVRP")
    
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--bandit_model', default='configuration_bandit.pkl', help='Path to ConfigurationBandit model')
    parser.add_argument('--val_dir', default='data/validation', help='Validation directory')
    parser.add_argument('--output_dir', default='results/multi_qpu_eval_new', help='Output directory')
    parser.add_argument('--device', default='auto', help='Torch device')
    
    # AWS/IBM credentials
    parser.add_argument('--profile', default=None, help='AWS CLI profile')
    parser.add_argument('--ibm_token', default=None, help='IBM Quantum token or IBM Cloud API key')
    parser.add_argument('--ibm_instance', default=None, help='IBM Quantum instance CRN (required for ibm_cloud channel, e.g., crn:v1:bluemix:public:quantum-computing:...)')
    parser.add_argument('--no_ibm', action='store_true', help='Disable IBM Quantum usage')
    
    # Instance selection
    parser.add_argument('--instances', nargs='+', default=None, help='Specific instances')
    parser.add_argument('--limit', type=int, default=-1, help='Limit instances')
    
    # Execution settings
    parser.add_argument('--max_steps', type=int, default=20, help='Max Lagrangian steps')
    parser.add_argument('--vqe_interval', type=int, default=1, help='VQE every N steps')
    parser.add_argument('--shots', type=int, default=100, help='Shots per circuit')
    parser.add_argument('--depth', type=int, default=1, help='Ansatz depth')
    parser.add_argument('--local-only', action='store_true', help='Allow running with only local simulator')
    
    # Resume
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--quiet', action='store_true', help='Reduce logging')
    
    args = parser.parse_args()
    
    if args.quiet:
        LOGGER.setLevel(logging.WARNING)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    torch_device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu'
    
    # Print header
    print(f"\n{'='*80}")
    print("MULTI-QPU QUANTUM EVALUATION")
    print("="*80)
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Bandit Model:   {args.bandit_model}")
    print(f"AWS Profile:    {args.profile}")
    print(f"IBM Instance:   {args.ibm_instance or 'default'}")
    print(f"Use IBM Q:      {'NO' if args.no_ibm else 'YES'}")
    print(f"VQE Interval:   Every {args.vqe_interval} steps")
    print(f"Shots:          {args.shots}")
    print(f"Local Only:     {args.local_only}")
    print("="*80)
    
    # Initialize Multi-QPU Manager
    print("\n[+] Initializing Multi-QPU System...")
    qpu_manager = MultiQPUManager(aws_profile=args.profile, ibm_token=args.ibm_token, ibm_instance=args.ibm_instance, use_ibm=not args.no_ibm)
    init_results = qpu_manager.initialize()
    
    print(qpu_manager.get_status_report())
    
    # Check at least one HARDWARE QPU is available (not just local simulator)
    hardware_available = any(qpu.is_available for qpu in qpu_manager.qpus.values())
    if not hardware_available and not args.local_only:
        print("ERROR: No quantum hardware backends available!")
        print("       Please check your AWS/IBM credentials and QPU availability.")
        print("       Use --local-only to run with local MPS simulator instead.")
        return
    
    if not hardware_available and args.local_only:
        print("\n[!] Running in LOCAL-ONLY mode with MPS simulator")
    
    # Initialize knapsack solver
    knapsack_solver = MultiQPUKnapsackSolver(
        qpu_manager,
        shots=args.shots,
        depth=args.depth,
        bandit_model=args.bandit_model
    )
    
    # Load model
    print("\n[*] Loading model checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=torch_device)
    
    sd = checkpoint['policy_state_dict']
    
    # Infer architecture from state dict if not explicitly stored
    policy_class = checkpoint.get('policy_class', 'diagonal')
    
    # Infer hidden_dim from encoder.weight shape
    if 'encoder.weight' in sd:
        hidden_dim = sd['encoder.weight'].shape[0]
    else:
        hidden_dim = checkpoint.get('hidden_dim', 256)
    
    # Infer heads from layers.0.att shape
    if 'layers.0.att' in sd:
        heads = sd['layers.0.att'].shape[1]
    else:
        heads = checkpoint.get('heads', 8)
    
    # Infer num_layers by counting layer keys
    num_layers = 0
    for key in sd.keys():
        if key.startswith('layers.') and '.att' in key:
            layer_idx = int(key.split('.')[1])
            num_layers = max(num_layers, layer_idx + 1)
    if num_layers == 0:
        num_layers = checkpoint.get('num_layers', 4)
    
    print(f"   Inferred: hidden_dim={hidden_dim}, heads={heads}, num_layers={num_layers}")
    
    policy_cfg = {'hidden_dim': hidden_dim, 'num_layers': num_layers, 'heads': heads}
    
    if policy_class == 'diagonal':
        policy = DiagonalPrecondPolicy(policy_cfg).to(torch_device)
    else:
        policy = PreconditionedGATPolicy(policy_cfg).to(torch_device)
    
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    # Get validation files
    all_files = sorted(glob.glob(os.path.join(args.val_dir, "*.vrp")))
    
    if args.instances:
        val_files = []
        for inst in args.instances:
            inst_name = inst if inst.endswith('.vrp') else inst + '.vrp'
            matching = [f for f in all_files if os.path.basename(f) == inst_name]
            if matching:
                val_files.extend(matching)
    else:
        val_files = all_files
    
    if args.limit > 0:
        val_files = val_files[:args.limit]
    
    if not val_files:
        print("No validation files found!")
        return
    
    # Checkpoint path
    checkpoint_path = os.path.join(args.output_dir, 'multi_qpu_checkpoint.json')
    
    # Initialize tracking
    results = []
    completed_instances = set()
    total_eval_time = 0.0
    total_vqe_time = 0.0
    gaps_to_bks = []
    
    # Resume if requested
    if args.resume:
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt:
            results = ckpt.get('results', [])
            completed_instances = set(ckpt.get('completed_instances', []))
            total_eval_time = ckpt.get('total_eval_time', 0.0)
            total_vqe_time = ckpt.get('total_vqe_time', 0.0)
            gaps_to_bks = ckpt.get('gaps_to_bks', [])
            print(f"   [RESUME] {len(completed_instances)} instances already completed:")
            for i, inst in enumerate(sorted(list(completed_instances))):
                print(f"     {i+1:2d}. {inst}")
    
    # Evaluation header
    print(f"\n{'='*100}")
    print(f"  EVALUATING {len(val_files)} INSTANCE(S) ON MULTI-QPU SYSTEM")
    print("="*100)
    
    header = f"{'Instance':<18} | {'N':>4} | {'K':>3} | {'Cost':>8} | {'BKS':>8} | {'Gap%':>7} | {'Feas':>4} | {'Time':>7} | QPUs Used"
    print(header)
    print("-" * len(header))
    
    for vf in tqdm(val_files, desc="Evaluating"):
        try:
            instance_name = os.path.basename(vf)
            
            if instance_name in completed_instances:
                tqdm.write(f"[SKIP] {instance_name} already completed")
                continue
            
            bks = parse_bks_from_file(vf)
            tqdm.write(f"\n[>] Starting: {instance_name}")
            
            env = CurriculumCVRPEnv(vf, {'max_steps': args.max_steps, 'k_neighbors': 5})
            
            metrics = evaluate_instance_multi_qpu(
                env, policy, knapsack_solver, torch_device,
                max_steps=args.max_steps,
                vqe_interval=args.vqe_interval
            )
            
            # Calculate gap
            gap_to_bks = None
            if bks and metrics['cost'] < float('inf'):
                gap_to_bks = 100 * (metrics['cost'] - bks) / bks
                gaps_to_bks.append(gap_to_bks)
            
            total_eval_time += metrics['solve_time']
            total_vqe_time += metrics['vqe_time']
            
            # Format QPU usage
            qpu_usage = metrics['quantum_stats'].get('qpu_usage', {})
            qpu_str = ", ".join([f"{k}:{v}" for k, v in qpu_usage.items()]) or "none"
            
            result = {
                'Instance': instance_name,
                'NumCustomers': metrics['num_customers'],
                'NumVehicles': metrics['num_vehicles'],
                'BKS': bks if bks else 'N/A',
                'OurCost': round(metrics['cost'], 2) if metrics['cost'] < float('inf') else 'N/A',
                'GapToBKS': round(gap_to_bks, 2) if gap_to_bks else 'N/A',
                'Feasible': 'Yes' if metrics['is_feasible'] else 'No',
                'TotalTime_sec': round(metrics['solve_time'], 2),
                'VQE_Time_sec': round(metrics['vqe_time'], 2),
                'QPU_Usage': qpu_usage,
                'TotalCircuits': metrics['quantum_stats']['total_circuits']
            }
            results.append(result)
            completed_instances.add(instance_name)
            
            # Print row
            bks_str = str(bks) if bks else 'N/A'
            cost_str = f"{metrics['cost']:.1f}" if metrics['cost'] < float('inf') else 'N/A'
            gap_str = f"{gap_to_bks:.1f}%" if gap_to_bks else 'N/A'
            feas_str = 'Yes' if metrics['is_feasible'] else 'NO'
            
            row = f"{instance_name[:18]:<18} | {metrics['num_customers']:>4} | {metrics['num_vehicles']:>3} | {cost_str:>8} | {bks_str:>8} | {gap_str:>7} | {feas_str:>4} | {metrics['solve_time']:>6.1f}s | {qpu_str}"
            tqdm.write(row)
            
            # Save checkpoint
            save_checkpoint(checkpoint_path, {
                'results': results,
                'completed_instances': list(completed_instances),
                'total_eval_time': total_eval_time,
                'total_vqe_time': total_vqe_time,
                'gaps_to_bks': gaps_to_bks,
                'qpu_job_log': qpu_manager.job_log
            })
            
            env.close()
            
        except Exception as e:
            import traceback
            tqdm.write(f"{os.path.basename(vf):<18} | ERROR: {e}")
            traceback.print_exc()
            
            # Save checkpoint on error
            save_checkpoint(checkpoint_path, {
                'results': results,
                'completed_instances': list(completed_instances),
                'total_eval_time': total_eval_time,
                'total_vqe_time': total_vqe_time,
                'gaps_to_bks': gaps_to_bks,
                'qpu_job_log': qpu_manager.job_log
            })
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"  Total Instances: {len(results)}")
    print(f"  Total Time: {total_eval_time:.1f}s")
    print(f"  VQE Time: {total_vqe_time:.1f}s ({100*total_vqe_time/max(total_eval_time,1):.1f}%)")
    
    if gaps_to_bks:
        print(f"  Mean Gap: {np.mean(gaps_to_bks):.2f}%")
    
    print("\nQPU Usage Summary:")
    for qpu_id, qpu in qpu_manager.qpus.items():
        if qpu.total_jobs > 0:
            print(f"  {qpu.name}: {qpu.total_jobs} jobs, {qpu.successful_jobs} success, {qpu.total_time:.1f}s")
    
    # Save final results
    csv_path = os.path.join(args.output_dir, 'multi_qpu_results.csv')
    if results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Instance', 'NumCustomers', 'NumVehicles', 
                                                    'BKS', 'OurCost', 'GapToBKS', 'Feasible',
                                                    'TotalTime_sec', 'VQE_Time_sec', 'TotalCircuits'],
                                    extrasaction='ignore')
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nResults saved to: {csv_path}")
    
    print("="*80)


if __name__ == "__main__":
    main()


## python -u .\evaluate_multi_qpu.py --checkpoint .\results\two_phase\finetuned_best.pt --val_dir .\data\small\ --max_steps 20 --vqe_interval 1 --profile quantum --no_ibm --resume > parallel_qpu.log 2>&1 
