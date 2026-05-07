"""
analog_circuit_generator.py

Physics-aware analog circuit graph generator.

Features implemented:
1. Device-level heterogeneous parameterization with 64-dim feature vectors
   - Discrete: device type (NMOS/PMOS/Inductor/Capacitor/Resistor/Diode/BJT),
                voltage domain (HVD/CVD/LVD)
   - Continuous: W, L, Multiplier, Number-of-Fingers (normalized)
   - Technology-node process constraints (110nm / 180nm min L & W)

2. Structural sizing rules
   - Current-mirror proportionality: shared L, integer-ratio W or M
   - Inverter sizing: Wpmos/Wnmos ratio in [2.0, 3.0]
   - Differential-pair matching: W1==W2, L1==L2

3. Physics-aware 4-terminal topology
   - Mandatory Body port (no simplified 3-terminal model)
   - Port relations: {Gate, Drain, Source, Body}
   - Body-tied-to-source support

4. Three-phase synthesis pipeline
   - Phase 1 – Seed injection: randomly select an expert template
                (OTA / VoltageReference / Cascode / BasicInverter / CommonSource)
   - Phase 2 – Node expansion: sample devices from industrial distribution
                (MOS ~83.6%), expand to 50-600 total nodes
   - Phase 3 – Topology completion: connect remaining ports following empirical
                pattern distribution (inverter-pair ~32.52%), respecting fan-in/fan-out

5. Physical validity checks
   - Rule 1 – VDD/GND isolation: no conductive path that bypasses all devices
   - Rule 2 – No floating gate: MOS Gate net must have degree ≥ 2
   - Bipartite consistency: every device port connected, no dangling branch
"""

from __future__ import annotations

import os
import json
import random
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import igraph as ig
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1. Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class DeviceType(IntEnum):
    NMOS      = 0
    PMOS      = 1
    INDUCTOR  = 2
    CAPACITOR = 3
    RESISTOR  = 4
    DIODE     = 5
    BJT       = 6


class VoltageDomain(IntEnum):
    HVD = 0   # High-Voltage Domain
    CVD = 1   # Common-Voltage Domain
    LVD = 2   # Low-Voltage Domain


class PortType(IntEnum):
    GATE   = 0
    DRAIN  = 1
    SOURCE = 2
    BODY   = 3

PORT_NAMES = {PortType.GATE: "Gate", PortType.DRAIN: "Drain",
              PortType.SOURCE: "Source", PortType.BODY: "Body"}

# MOS transistors (need 4-terminal model)
MOS_TYPES   = {DeviceType.NMOS, DeviceType.PMOS}
# All MOS including BJT for floating-gate rule
ACTIVE_TYPES = {DeviceType.NMOS, DeviceType.PMOS, DeviceType.BJT}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Technology-node process constraints
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TechNode:
    name:           str
    min_L:          float   # metres
    min_W:          float   # metres
    max_L:          float   # metres
    max_W:          float   # metres
    max_multiplier: int = 32
    max_fingers:    int = 32


TECH_NODES: Dict[str, TechNode] = {
    "110nm": TechNode("110nm",
                      min_L=110e-9, min_W=220e-9,
                      max_L=2e-6,   max_W=50e-6),
    "180nm": TechNode("180nm",
                      min_L=180e-9, min_W=360e-9,
                      max_L=5e-6,   max_W=100e-6),
}

# ──────────────────────────────────────────────────────────────────────────────
# 3. Device sizing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rand_W(tech: TechNode) -> float:
    """Sample a random width respecting process minimum."""
    return random.uniform(tech.min_W, tech.max_W)


def _rand_L(tech: TechNode) -> float:
    """Sample a random length respecting process minimum."""
    return random.uniform(tech.min_L, tech.max_L)


def _rand_multiplier(tech: TechNode) -> int:
    return random.randint(1, tech.max_multiplier)


def _rand_fingers(tech: TechNode) -> int:
    return random.randint(1, tech.max_fingers)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 64-dim feature vector
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_DIM       = 64
NUM_DEVICE_TYPES  = len(DeviceType)   # 7
NUM_VOLT_DOMAINS  = len(VoltageDomain)  # 3
# Layout: [0:7] type one-hot | [7:10] domain one-hot |
#         [10] W_norm | [11] L_norm | [12] M_norm | [13] F_norm |
#         [14] body_tied_flag | [15:64] zero-padded


def compute_feature_vector(
    device_type:        int,
    voltage_domain:     int,
    W:                  float,
    L:                  float,
    multiplier:         int,
    num_fingers:        int,
    body_tied_to_source: bool,
    tech:               TechNode,
) -> List[float]:
    """Generate the 64-dim initial hidden state h_vD(0) for one device node."""
    feat = [0.0] * FEATURE_DIM

    # Device-type one-hot  (dims 0-6)
    feat[device_type] = 1.0

    # Voltage-domain one-hot  (dims 7-9)
    feat[NUM_DEVICE_TYPES + voltage_domain] = 1.0

    base = NUM_DEVICE_TYPES + NUM_VOLT_DOMAINS  # 10

    # Normalised continuous size parameters  (dims 10-13)
    W_range = max(tech.max_W - tech.min_W, 1e-15)
    L_range = max(tech.max_L - tech.min_L, 1e-15)
    M_range = max(tech.max_multiplier - 1, 1)
    F_range = max(tech.max_fingers - 1, 1)

    feat[base + 0] = float(np.clip((W - tech.min_W) / W_range, 0.0, 1.0))
    feat[base + 1] = float(np.clip((L - tech.min_L) / L_range, 0.0, 1.0))
    feat[base + 2] = float(np.clip((multiplier - 1) / M_range, 0.0, 1.0))
    feat[base + 3] = float(np.clip((num_fingers - 1) / F_range, 0.0, 1.0))

    # Body-tied-to-source flag  (dim 14)
    feat[base + 4] = 1.0 if body_tied_to_source else 0.0

    # dims 15-63 are zero-padded for downstream learnable expansion
    return feat


# ──────────────────────────────────────────────────────────────────────────────
# 5. Expert template library (Phase 1 – Seed Injection)
# ──────────────────────────────────────────────────────────────────────────────
#
# Each template is a dict:
#   "devices"     : list of dicts {id, type, domain}
#   "nets"        : list of net names (strings)
#   "connections" : list of (device_id, PortType, net_name)
#   "hints"       : structural pattern hints for sizing rules
#       "diff_pairs"      : list of (dev_id, dev_id)
#       "current_mirrors" : list of [dev_id, ...]
#       "inverter_pairs"  : list of (pmos_id, nmos_id)

TEMPLATES: List[Dict] = [
    # ── OTA (5-transistor) ────────────────────────────────────────────────
    {
        "name": "OTA",
        "devices": [
            {"id": "M1", "type": DeviceType.NMOS, "domain": VoltageDomain.CVD},
            {"id": "M2", "type": DeviceType.NMOS, "domain": VoltageDomain.CVD},
            {"id": "M3", "type": DeviceType.PMOS, "domain": VoltageDomain.CVD},
            {"id": "M4", "type": DeviceType.PMOS, "domain": VoltageDomain.CVD},
            {"id": "M5", "type": DeviceType.NMOS, "domain": VoltageDomain.CVD},
        ],
        "nets": ["VDD", "GND", "INP", "INN", "OUT", "TAIL", "DIODE"],
        "connections": [
            ("M1", PortType.GATE,   "INP"),
            ("M1", PortType.DRAIN,  "OUT"),
            ("M1", PortType.SOURCE, "TAIL"),
            ("M1", PortType.BODY,   "GND"),
            ("M2", PortType.GATE,   "INN"),
            ("M2", PortType.DRAIN,  "DIODE"),
            ("M2", PortType.SOURCE, "TAIL"),
            ("M2", PortType.BODY,   "GND"),
            ("M3", PortType.GATE,   "DIODE"),
            ("M3", PortType.DRAIN,  "DIODE"),
            ("M3", PortType.SOURCE, "VDD"),
            ("M3", PortType.BODY,   "VDD"),
            ("M4", PortType.GATE,   "DIODE"),
            ("M4", PortType.DRAIN,  "OUT"),
            ("M4", PortType.SOURCE, "VDD"),
            ("M4", PortType.BODY,   "VDD"),
            ("M5", PortType.GATE,   "INN"),
            ("M5", PortType.DRAIN,  "TAIL"),
            ("M5", PortType.SOURCE, "GND"),
            ("M5", PortType.BODY,   "GND"),
        ],
        "hints": {
            "diff_pairs":      [("M1", "M2")],
            "current_mirrors": [["M3", "M4"]],
            "inverter_pairs":  [],
        },
    },
    # ── Voltage Reference (bandgap-like, 4 transistors) ───────────────────
    {
        "name": "VoltageReference",
        "devices": [
            {"id": "M1", "type": DeviceType.PMOS, "domain": VoltageDomain.HVD},
            {"id": "M2", "type": DeviceType.PMOS, "domain": VoltageDomain.HVD},
            {"id": "M3", "type": DeviceType.NMOS, "domain": VoltageDomain.CVD},
            {"id": "M4", "type": DeviceType.NMOS, "domain": VoltageDomain.CVD},
        ],
        "nets": ["VDD", "GND", "VREF", "BIAS", "DIODE_N"],
        "connections": [
            ("M1", PortType.GATE,   "BIAS"),
            ("M1", PortType.DRAIN,  "VREF"),
            ("M1", PortType.SOURCE, "VDD"),
            ("M1", PortType.BODY,   "VDD"),
            ("M2", PortType.GATE,   "BIAS"),
            ("M2", PortType.DRAIN,  "BIAS"),
            ("M2", PortType.SOURCE, "VDD"),
            ("M2", PortType.BODY,   "VDD"),
            ("M3", PortType.GATE,   "VREF"),
            ("M3", PortType.DRAIN,  "VREF"),
            ("M3", PortType.SOURCE, "GND"),
            ("M3", PortType.BODY,   "GND"),
            ("M4", PortType.GATE,   "DIODE_N"),
            ("M4", PortType.DRAIN,  "BIAS"),
            ("M4", PortType.SOURCE, "GND"),
            ("M4", PortType.BODY,   "GND"),
        ],
        "hints": {
            "diff_pairs":      [],
            "current_mirrors": [["M1", "M2"]],
            "inverter_pairs":  [],
        },
    },
    # ── Cascode amplifier (4 transistors) ────────────────────────────────
    {
        "name": "Cascode",
        "devices": [
            {"id": "M1", "type": DeviceType.NMOS, "domain": VoltageDomain.CVD},
            {"id": "M2", "type": DeviceType.NMOS, "domain": VoltageDomain.CVD},
            {"id": "M3", "type": DeviceType.PMOS, "domain": VoltageDomain.CVD},
            {"id": "M4", "type": DeviceType.PMOS, "domain": VoltageDomain.CVD},
        ],
        "nets": ["VDD", "GND", "IN", "OUT", "BIAS_N", "BIAS_P", "MID"],
        "connections": [
            ("M1", PortType.GATE,   "IN"),
            ("M1", PortType.DRAIN,  "MID"),
            ("M1", PortType.SOURCE, "GND"),
            ("M1", PortType.BODY,   "GND"),
            ("M2", PortType.GATE,   "BIAS_N"),
            ("M2", PortType.DRAIN,  "OUT"),
            ("M2", PortType.SOURCE, "MID"),
            ("M2", PortType.BODY,   "GND"),
            ("M3", PortType.GATE,   "BIAS_P"),
            ("M3", PortType.DRAIN,  "OUT"),
            ("M3", PortType.SOURCE, "VDD"),
            ("M3", PortType.BODY,   "VDD"),
            ("M4", PortType.GATE,   "BIAS_P"),
            ("M4", PortType.DRAIN,  "BIAS_P"),
            ("M4", PortType.SOURCE, "VDD"),
            ("M4", PortType.BODY,   "VDD"),
        ],
        "hints": {
            "diff_pairs":      [],
            "current_mirrors": [["M3", "M4"]],
            "inverter_pairs":  [],
        },
    },
    # ── Basic Inverter ────────────────────────────────────────────────────
    {
        "name": "BasicInverter",
        "devices": [
            {"id": "MP", "type": DeviceType.PMOS, "domain": VoltageDomain.CVD},
            {"id": "MN", "type": DeviceType.NMOS, "domain": VoltageDomain.CVD},
        ],
        "nets": ["VDD", "GND", "IN", "OUT"],
        "connections": [
            ("MP", PortType.GATE,   "IN"),
            ("MP", PortType.DRAIN,  "OUT"),
            ("MP", PortType.SOURCE, "VDD"),
            ("MP", PortType.BODY,   "VDD"),
            ("MN", PortType.GATE,   "IN"),
            ("MN", PortType.DRAIN,  "OUT"),
            ("MN", PortType.SOURCE, "GND"),
            ("MN", PortType.BODY,   "GND"),
        ],
        "hints": {
            "diff_pairs":      [],
            "current_mirrors": [],
            "inverter_pairs":  [("MP", "MN")],
        },
    },
    # ── Common-Source Amplifier ───────────────────────────────────────────
    {
        "name": "CommonSource",
        "devices": [
            {"id": "M1", "type": DeviceType.NMOS, "domain": VoltageDomain.CVD},
            {"id": "M2", "type": DeviceType.PMOS, "domain": VoltageDomain.CVD},
        ],
        "nets": ["VDD", "GND", "IN", "OUT", "BIAS"],
        "connections": [
            ("M1", PortType.GATE,   "IN"),
            ("M1", PortType.DRAIN,  "OUT"),
            ("M1", PortType.SOURCE, "GND"),
            ("M1", PortType.BODY,   "GND"),
            ("M2", PortType.GATE,   "BIAS"),
            ("M2", PortType.DRAIN,  "OUT"),
            ("M2", PortType.SOURCE, "VDD"),
            ("M2", PortType.BODY,   "VDD"),
        ],
        "hints": {
            "diff_pairs":      [],
            "current_mirrors": [],
            "inverter_pairs":  [],
        },
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# 6. Device-type sampling distribution (industrial statistics)
# ──────────────────────────────────────────────────────────────────────────────
#   NMOS ~42 %, PMOS ~41.6 %, Capacitor ~6 %, Resistor ~5 %,
#   Inductor ~2.4 %, Diode ~2 %, BJT ~1 %

_DEVICE_TYPES    = [DeviceType.NMOS, DeviceType.PMOS, DeviceType.CAPACITOR,
                    DeviceType.RESISTOR, DeviceType.INDUCTOR,
                    DeviceType.DIODE,    DeviceType.BJT]
_DEVICE_WEIGHTS  = [0.420, 0.416, 0.060, 0.050, 0.024, 0.020, 0.010]


def _sample_device_type() -> DeviceType:
    return random.choices(_DEVICE_TYPES, weights=_DEVICE_WEIGHTS, k=1)[0]


def _sample_voltage_domain() -> VoltageDomain:
    return random.choice(list(VoltageDomain))


# ──────────────────────────────────────────────────────────────────────────────
# 7. Core bipartite-graph builder helpers
# ──────────────────────────────────────────────────────────────────────────────

class CircuitGraph:
    """
    Thin wrapper around an igraph Graph that maintains a device-net bipartite
    representation.

    Vertex attributes:
        node_type       : "device" | "net"
        -- device nodes only --
        device_type     : int (DeviceType)
        voltage_domain  : int (VoltageDomain)
        W               : float  (metres)
        L               : float  (metres)
        multiplier      : int
        num_fingers     : int
        body_tied       : bool   (Body-tied-to-Source)
        feature_vector  : list[float]  (64-dim)
        -- net nodes only --
        net_name        : str

    Edge attributes:
        port_type       : int (PortType)
        port_name       : str  ("Gate"/"Drain"/"Source"/"Body")
    """

    def __init__(self, tech: TechNode):
        self.tech  = tech
        self.g     = ig.Graph(directed=False)
        self._net_index:    Dict[str, int] = {}   # net_name  -> vertex id
        self._dev_index:    Dict[str, int] = {}   # device_id -> vertex id
        self._net_counter   = 0

    # ── low-level vertex adders ───────────────────────────────────────────

    def add_net(self, net_name: str) -> int:
        if net_name in self._net_index:
            return self._net_index[net_name]
        vid = self.g.vcount()
        self.g.add_vertex(
            node_type="net",
            net_name=net_name,
            # Dummy values for device attributes (not used for net nodes)
            device_type=-1, voltage_domain=-1,
            W=0.0, L=0.0, multiplier=0, num_fingers=0,
            body_tied=False, feature_vector=[0.0] * FEATURE_DIM,
        )
        self._net_index[net_name] = vid
        return vid

    def add_device(
        self,
        dev_id:      str,
        dtype:       DeviceType,
        domain:      VoltageDomain,
        W:           Optional[float] = None,
        L:           Optional[float] = None,
        multiplier:  Optional[int]   = None,
        num_fingers: Optional[int]   = None,
        body_tied:   bool            = False,
    ) -> int:
        tech = self.tech
        W          = W          if W          is not None else _rand_W(tech)
        L          = L          if L          is not None else _rand_L(tech)
        multiplier = multiplier if multiplier is not None else _rand_multiplier(tech)
        num_fingers = num_fingers if num_fingers is not None else _rand_fingers(tech)

        fv = compute_feature_vector(int(dtype), int(domain), W, L,
                                    multiplier, num_fingers, body_tied, tech)
        vid = self.g.vcount()
        self.g.add_vertex(
            node_type="device",
            net_name="",
            device_type=int(dtype),
            voltage_domain=int(domain),
            W=W, L=L,
            multiplier=multiplier,
            num_fingers=num_fingers,
            body_tied=body_tied,
            feature_vector=fv,
        )
        self._dev_index[dev_id] = vid
        return vid

    def connect(self, dev_id: str, port: PortType, net_name: str):
        """Add an edge between a device port and a net."""
        dev_vid = self._dev_index[dev_id]
        net_vid = self._net_index[net_name]
        self.g.add_edge(dev_vid, net_vid,
                        port_type=int(port),
                        port_name=PORT_NAMES[port])

    # ── convenience ──────────────────────────────────────────────────────

    def new_signal_net(self) -> str:
        """Allocate a fresh anonymous signal net and return its name."""
        name = "net_%d" % self._net_counter
        self._net_counter += 1
        self.add_net(name)
        return name

    def device_ids(self) -> List[str]:
        return list(self._dev_index.keys())

    def net_names(self) -> List[str]:
        return list(self._net_index.keys())

    def num_devices(self) -> int:
        return len(self._dev_index)

    def num_nets(self) -> int:
        return len(self._net_index)

    def device_vertex(self, dev_id: str):
        return self.g.vs[self._dev_index[dev_id]]

    def net_vertex(self, net_name: str):
        return self.g.vs[self._net_index[net_name]]

    def ports_of_device(self, dev_id: str) -> List[Tuple[PortType, str]]:
        """Return [(port_type, net_name), ...] for a device."""
        vid = self._dev_index[dev_id]
        result = []
        for e in self.g.incident(vid):
            edge = self.g.es[e]
            neighbor_id = edge.source if edge.target == vid else edge.target
            net_name = self.g.vs[neighbor_id]["net_name"]
            result.append((PortType(edge["port_type"]), net_name))
        return result

    def nets_of_device(self, dev_id: str) -> Dict[PortType, str]:
        return {pt: net for pt, net in self.ports_of_device(dev_id)}

    def devices_on_net(self, net_name: str) -> List[Tuple[str, PortType]]:
        """Return [(dev_id, port_type), ...] for every device connected to a net."""
        net_vid = self._net_index[net_name]
        result  = []
        id_map  = {v: k for k, v in self._dev_index.items()}
        for e in self.g.incident(net_vid):
            edge = self.g.es[e]
            dev_vid = edge.source if edge.target == net_vid else edge.target
            if dev_vid in id_map:
                result.append((id_map[dev_vid], PortType(edge["port_type"])))
        return result


# ──────────────────────────────────────────────────────────────────────────────
# 8. Structural pattern detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_current_mirrors(cg: CircuitGraph) -> List[List[str]]:
    """
    Identify current-mirror groups: sets of MOS transistors that share the
    same Gate net AND are of the same DeviceType.
    Returns a list of groups; each group is a list of device_ids.
    """
    gate_net_to_devs: Dict[str, List[str]] = defaultdict(list)
    for dev_id in cg.device_ids():
        dtype = DeviceType(cg.device_vertex(dev_id)["device_type"])
        if dtype not in MOS_TYPES:
            continue
        nets = cg.nets_of_device(dev_id)
        gate_net = nets.get(PortType.GATE)
        if gate_net:
            gate_net_to_devs[gate_net].append(dev_id)

    mirrors = []
    for gate_net, devs in gate_net_to_devs.items():
        if len(devs) < 2:
            continue
        # Group by device type
        by_type: Dict[int, List[str]] = defaultdict(list)
        for d in devs:
            dt = cg.device_vertex(d)["device_type"]
            by_type[dt].append(d)
        for dt, group in by_type.items():
            if len(group) >= 2:
                mirrors.append(group)
    return mirrors


def detect_inverter_pairs(cg: CircuitGraph) -> List[Tuple[str, str]]:
    """
    Identify inverter pairs: one PMOS + one NMOS sharing both Gate and Drain
    nets.  Returns list of (pmos_id, nmos_id).
    """
    # Build mapping: (gate_net, drain_net) -> {PMOS: dev_id, NMOS: dev_id}
    key_map: Dict[Tuple[str, str], Dict[DeviceType, str]] = defaultdict(dict)
    for dev_id in cg.device_ids():
        dtype = DeviceType(cg.device_vertex(dev_id)["device_type"])
        if dtype not in MOS_TYPES:
            continue
        nets = cg.nets_of_device(dev_id)
        g_net = nets.get(PortType.GATE)
        d_net = nets.get(PortType.DRAIN)
        if g_net and d_net:
            key_map[(g_net, d_net)][dtype] = dev_id

    pairs = []
    for key, dtype_map in key_map.items():
        if DeviceType.PMOS in dtype_map and DeviceType.NMOS in dtype_map:
            pairs.append((dtype_map[DeviceType.PMOS], dtype_map[DeviceType.NMOS]))
    return pairs


def detect_differential_pairs(cg: CircuitGraph) -> List[Tuple[str, str]]:
    """
    Identify differential pairs: two MOS of the same type sharing the same
    Source net (which must be a signal net, not VDD/GND) but with different
    Gate nets.  Returns list of (dev1_id, dev2_id).
    """
    source_net_to_devs: Dict[str, List[str]] = defaultdict(list)
    for dev_id in cg.device_ids():
        dtype = DeviceType(cg.device_vertex(dev_id)["device_type"])
        if dtype not in MOS_TYPES:
            continue
        nets = cg.nets_of_device(dev_id)
        s_net = nets.get(PortType.SOURCE)
        # Exclude power rails – real diff-pair tail is a signal net
        if s_net and s_net not in ("VDD", "GND"):
            source_net_to_devs[s_net].append(dev_id)

    diff_pairs = []
    for s_net, devs in source_net_to_devs.items():
        if len(devs) < 2:
            continue
        # Check every pair
        for i in range(len(devs)):
            for j in range(i + 1, len(devs)):
                d1, d2 = devs[i], devs[j]
                if (cg.device_vertex(d1)["device_type"]
                        != cg.device_vertex(d2)["device_type"]):
                    continue
                n1 = cg.nets_of_device(d1).get(PortType.GATE)
                n2 = cg.nets_of_device(d2).get(PortType.GATE)
                if n1 and n2 and n1 != n2:
                    diff_pairs.append((d1, d2))
    return diff_pairs


# ──────────────────────────────────────────────────────────────────────────────
# 9. Structural sizing rules
# ──────────────────────────────────────────────────────────────────────────────

def _update_device_size(cg: CircuitGraph, dev_id: str,
                         W=None, L=None, multiplier=None, num_fingers=None,
                         body_tied=None):
    """Patch sizing attributes and recompute feature vector."""
    v = cg.device_vertex(dev_id)
    if W          is not None: v["W"]           = W
    if L          is not None: v["L"]           = L
    if multiplier is not None: v["multiplier"]  = multiplier
    if num_fingers is not None: v["num_fingers"] = num_fingers
    if body_tied  is not None: v["body_tied"]   = body_tied

    v["feature_vector"] = compute_feature_vector(
        v["device_type"], v["voltage_domain"],
        v["W"], v["L"], v["multiplier"], v["num_fingers"],
        v["body_tied"], cg.tech)


def apply_current_mirror_sizing(cg: CircuitGraph):
    """Force all branches of a current mirror to share the same L and use
    integer-ratio W/M (e.g. 1:1, 1:2, 1:N)."""
    mirrors = detect_current_mirrors(cg)
    for group in mirrors:
        # Choose a common L
        ref_L = _rand_L(cg.tech)
        # Choose a base W
        base_W = _rand_W(cg.tech)
        for dev_id in group:
            ratio = random.randint(1, 8)
            _update_device_size(cg, dev_id,
                                W=base_W * ratio,
                                L=ref_L,
                                multiplier=ratio)


def apply_inverter_sizing(cg: CircuitGraph):
    """Set Wpmos/Wnmos ratio in [2.0, 3.0] for detected inverter pairs."""
    pairs = detect_inverter_pairs(cg)
    for pmos_id, nmos_id in pairs:
        Wn   = _rand_W(cg.tech)
        ratio = random.uniform(2.0, 3.0)
        Wp   = Wn * ratio
        L    = _rand_L(cg.tech)
        _update_device_size(cg, nmos_id, W=Wn, L=L)
        _update_device_size(cg, pmos_id, W=Wp, L=L)


def apply_differential_pair_sizing(cg: CircuitGraph):
    """Enforce W1==W2, L1==L2 for differential pairs."""
    diff_pairs = detect_differential_pairs(cg)
    for d1, d2 in diff_pairs:
        W = _rand_W(cg.tech)
        L = _rand_L(cg.tech)
        _update_device_size(cg, d1, W=W, L=L)
        _update_device_size(cg, d2, W=W, L=L)


def apply_all_sizing_constraints(cg: CircuitGraph):
    """Apply all three structural sizing rules in sequence."""
    apply_current_mirror_sizing(cg)
    apply_inverter_sizing(cg)
    apply_differential_pair_sizing(cg)


# ──────────────────────────────────────────────────────────────────────────────
# 10. Phase 1 – Seed injection
# ──────────────────────────────────────────────────────────────────────────────

def inject_seed(cg: CircuitGraph, template: Optional[Dict] = None) -> Dict:
    """
    Instantiate one expert template into *cg* and return the template hints
    (used later for sizing).  If *template* is None a random one is chosen.
    """
    if template is None:
        template = random.choice(TEMPLATES)

    # Add nets
    for net_name in template["nets"]:
        cg.add_net(net_name)

    # Add devices
    for dev in template["devices"]:
        cg.add_device(dev["id"], dev["type"], dev["domain"])

    # Connect ports
    for dev_id, port, net_name in template["connections"]:
        cg.connect(dev_id, port, net_name)

    return template["hints"]


# ──────────────────────────────────────────────────────────────────────────────
# 11. Phase 2 – Node expansion
# ──────────────────────────────────────────────────────────────────────────────

# Topology pattern probabilities (empirical, from problem statement)
_TOPO_PATTERNS   = ["inverter_pair", "current_mirror_branch",
                    "bypass_cap",    "series_resistor",
                    "cascode_stage", "simple_branch"]
_TOPO_WEIGHTS    = [0.3252, 0.20, 0.15, 0.10, 0.12, 0.1048]

# Maximum fan-out per signal net (typical analog constraint)
_MAX_FANOUT = 8


def _existing_signal_nets(cg: CircuitGraph) -> List[str]:
    """Return all non-power nets."""
    return [n for n in cg.net_names() if n not in ("VDD", "GND")]


def _pick_signal_net(cg: CircuitGraph) -> str:
    """
    Pick an existing signal net with remaining fan-out capacity, or create a
    new one if none are available.
    """
    candidates = [n for n in _existing_signal_nets(cg)
                  if len(cg.devices_on_net(n)) < _MAX_FANOUT]
    if candidates:
        return random.choice(candidates)
    return cg.new_signal_net()


def _add_mos_device(cg: CircuitGraph, dev_id: str,
                    dtype: DeviceType, domain: VoltageDomain,
                    gate_net: str, drain_net: str, source_net: str,
                    body_tied: bool = False):
    cg.add_device(dev_id, dtype, domain, body_tied=body_tied)
    cg.connect(dev_id, PortType.GATE,   gate_net)
    cg.connect(dev_id, PortType.DRAIN,  drain_net)
    cg.connect(dev_id, PortType.SOURCE, source_net)
    body_net = source_net if body_tied else ("GND" if dtype == DeviceType.NMOS else "VDD")
    cg.connect(dev_id, PortType.BODY, body_net)


def _add_passive_device(cg: CircuitGraph, dev_id: str,
                        dtype: DeviceType, domain: VoltageDomain,
                        net_a: str, net_b: str):
    """
    Passive devices (Capacitor/Resistor/Inductor/Diode) use only DRAIN and
    SOURCE ports for the two terminals; GATE and BODY default to a safe net.
    """
    cg.add_device(dev_id, dtype, domain)
    cg.connect(dev_id, PortType.DRAIN,  net_a)
    cg.connect(dev_id, PortType.SOURCE, net_b)
    # Tie Gate and Body to one terminal (neutral connection)
    cg.connect(dev_id, PortType.GATE,  net_a)
    cg.connect(dev_id, PortType.BODY,  net_b)


def _next_dev_id(cg: CircuitGraph, prefix: str = "X") -> str:
    return "%s%d" % (prefix, cg.num_devices())


def _add_topology_pattern(cg: CircuitGraph, pattern: str):
    """Add one topology sub-pattern to the circuit graph."""
    sig_nets = _existing_signal_nets(cg)

    if pattern == "inverter_pair":
        in_net  = _pick_signal_net(cg)
        out_net = cg.new_signal_net()
        mp = _next_dev_id(cg, "MP")
        cg.add_device(mp, DeviceType.PMOS, VoltageDomain.CVD)
        cg.connect(mp, PortType.GATE,   in_net)
        cg.connect(mp, PortType.DRAIN,  out_net)
        cg.connect(mp, PortType.SOURCE, "VDD")
        cg.connect(mp, PortType.BODY,   "VDD")
        mn = _next_dev_id(cg, "MN")
        cg.add_device(mn, DeviceType.NMOS, VoltageDomain.CVD)
        cg.connect(mn, PortType.GATE,   in_net)
        cg.connect(mn, PortType.DRAIN,  out_net)
        cg.connect(mn, PortType.SOURCE, "GND")
        cg.connect(mn, PortType.BODY,   "GND")

    elif pattern == "current_mirror_branch":
        ref_net  = _pick_signal_net(cg)
        out_net  = cg.new_signal_net()
        tail_net = cg.new_signal_net()
        dtype    = random.choice([DeviceType.NMOS, DeviceType.PMOS])
        supply   = "GND" if dtype == DeviceType.NMOS else "VDD"
        # Reference branch (diode-connected)
        mr = _next_dev_id(cg, "MR")
        _add_mos_device(cg, mr, dtype, VoltageDomain.CVD,
                        gate_net=ref_net, drain_net=ref_net, source_net=supply)
        # Mirror branch
        mb = _next_dev_id(cg, "MB")
        _add_mos_device(cg, mb, dtype, VoltageDomain.CVD,
                        gate_net=ref_net, drain_net=out_net, source_net=supply)

    elif pattern == "bypass_cap":
        sig_net = _pick_signal_net(cg)
        cx = _next_dev_id(cg, "C")
        _add_passive_device(cg, cx, DeviceType.CAPACITOR, VoltageDomain.CVD,
                            sig_net, "GND")

    elif pattern == "series_resistor":
        net_a = _pick_signal_net(cg)
        net_b = cg.new_signal_net()
        rx = _next_dev_id(cg, "R")
        _add_passive_device(cg, rx, DeviceType.RESISTOR, VoltageDomain.CVD,
                            net_a, net_b)

    elif pattern == "cascode_stage":
        in_net    = _pick_signal_net(cg)
        mid_net   = cg.new_signal_net()
        out_net   = cg.new_signal_net()
        bias_net  = cg.new_signal_net()
        dtype     = DeviceType.NMOS
        m1 = _next_dev_id(cg, "MC1")
        _add_mos_device(cg, m1, dtype, VoltageDomain.CVD,
                        gate_net=in_net, drain_net=mid_net, source_net="GND")
        m2 = _next_dev_id(cg, "MC2")
        _add_mos_device(cg, m2, dtype, VoltageDomain.CVD,
                        gate_net=bias_net, drain_net=out_net, source_net=mid_net)

    else:  # simple_branch: single MOS driving a new net
        in_net  = _pick_signal_net(cg)
        out_net = cg.new_signal_net()
        dtype   = _sample_device_type()
        if dtype in MOS_TYPES:
            mx = _next_dev_id(cg, "M")
            _add_mos_device(cg, mx, dtype, _sample_voltage_domain(),
                            gate_net=in_net, drain_net=out_net,
                            source_net="GND" if dtype == DeviceType.NMOS else "VDD")
        else:
            px = _next_dev_id(cg, "P")
            _add_passive_device(cg, px, dtype, _sample_voltage_domain(),
                                in_net, out_net)


def expand_nodes(cg: CircuitGraph, target_device_count: int):
    """
    Phase 2: add devices sampled from the industrial distribution until
    *target_device_count* is reached, using topology patterns.
    """
    while cg.num_devices() < target_device_count:
        pattern = random.choices(_TOPO_PATTERNS, weights=_TOPO_WEIGHTS, k=1)[0]
        _add_topology_pattern(cg, pattern)


# ──────────────────────────────────────────────────────────────────────────────
# 12. Phase 3 – Topology completion
# ──────────────────────────────────────────────────────────────────────────────

def complete_topology(cg: CircuitGraph):
    """
    Phase 3: ensure that every device has all four ports connected.
    Floating ports are connected to appropriate nets (no floating gates).
    """
    for dev_id in cg.device_ids():
        connected_ports = {pt for pt, _ in cg.ports_of_device(dev_id)}
        dtype = DeviceType(cg.device_vertex(dev_id)["device_type"])

        # Determine which ports are needed
        required = {PortType.GATE, PortType.DRAIN, PortType.SOURCE, PortType.BODY}

        for port in required:
            if port in connected_ports:
                continue

            # Choose a safe net for each missing port
            if port == PortType.BODY:
                # Default body connection
                s_nets = {pt: n for pt, n in cg.ports_of_device(dev_id)}
                if PortType.SOURCE in s_nets:
                    body_net = s_nets[PortType.SOURCE]
                elif dtype in MOS_TYPES:
                    body_net = "GND" if dtype == DeviceType.NMOS else "VDD"
                else:
                    body_net = "GND"
                cg.connect(dev_id, PortType.BODY, body_net)

            elif port == PortType.GATE:
                # Gate must connect to a net with ≥1 other connection (no floating gate)
                # Prefer an existing signal net
                candidates = [n for n in _existing_signal_nets(cg)
                              if len(cg.devices_on_net(n)) >= 1]
                if not candidates:
                    gate_net = cg.new_signal_net()
                    # Ensure net is non-trivial: connect a diode-connected device
                    dummy_id = _next_dev_id(cg, "MDG")
                    _add_mos_device(cg, dummy_id, DeviceType.NMOS,
                                    VoltageDomain.CVD,
                                    gate_net=gate_net, drain_net=gate_net,
                                    source_net="GND")
                else:
                    gate_net = random.choice(candidates)
                cg.connect(dev_id, PortType.GATE, gate_net)

            elif port == PortType.DRAIN:
                drain_net = _pick_signal_net(cg)
                cg.connect(dev_id, PortType.DRAIN, drain_net)

            elif port == PortType.SOURCE:
                if dtype == DeviceType.NMOS:
                    cg.connect(dev_id, PortType.SOURCE, "GND")
                elif dtype == DeviceType.PMOS:
                    cg.connect(dev_id, PortType.SOURCE, "VDD")
                else:
                    src_net = _pick_signal_net(cg)
                    cg.connect(dev_id, PortType.SOURCE, src_net)


# ──────────────────────────────────────────────────────────────────────────────
# 13. Physical validity checks (Rule 1 and Rule 2)
# ──────────────────────────────────────────────────────────────────────────────

def check_vdd_gnd_isolation(cg: CircuitGraph) -> bool:
    """
    Rule 1: VDD and GND must not be connected to the same device via its two
    primary current-carrying terminals (Drain/Source) simultaneously, which
    would represent a potential short circuit.

    We check for any device whose {Drain, Source} nets include both VDD and GND.
    Also verifies that VDD and GND are not the same net object.
    """
    if "VDD" not in cg._net_index or "GND" not in cg._net_index:
        return True  # No power nets yet – trivially valid

    for dev_id in cg.device_ids():
        nets = cg.nets_of_device(dev_id)
        current_nets = {nets.get(PortType.DRAIN), nets.get(PortType.SOURCE)}
        current_nets.discard(None)
        if "VDD" in current_nets and "GND" in current_nets:
            return False  # Short-circuit path detected
    return True


def check_no_floating_gate(cg: CircuitGraph) -> bool:
    """
    Rule 2: For each MOS/BJT device, the net connected to the Gate port must
    have degree ≥ 2 (i.e., it is connected to at least one other device port
    besides this Gate connection, preventing a truly floating gate).
    """
    for dev_id in cg.device_ids():
        dtype = DeviceType(cg.device_vertex(dev_id)["device_type"])
        if dtype not in ACTIVE_TYPES:
            continue
        nets = cg.nets_of_device(dev_id)
        gate_net = nets.get(PortType.GATE)
        if gate_net is None:
            return False  # Missing gate connection
        net_vid = cg._net_index[gate_net]
        # Count all edges on the gate net
        degree = len(cg.g.incident(net_vid))
        if degree < 2:
            return False  # Only this one Gate edge – truly floating
    return True


def check_bipartite_consistency(cg: CircuitGraph) -> bool:
    """
    Verify device-net bipartite consistency:
      - Every device vertex connects only to net vertices (and vice-versa).
      - No device has dangling (unconnected) ports.
    """
    # Build set of device vertex ids and net vertex ids
    dev_vids = set(cg._dev_index.values())
    net_vids = set(cg._net_index.values())

    for edge in cg.g.es:
        s, t = edge.source, edge.target
        # One endpoint must be device, the other net
        if not ((s in dev_vids and t in net_vids) or
                (s in net_vids and t in dev_vids)):
            return False

    # Every device must have all four ports connected
    for dev_id in cg.device_ids():
        connected = {pt for pt, _ in cg.ports_of_device(dev_id)}
        required  = {PortType.GATE, PortType.DRAIN, PortType.SOURCE, PortType.BODY}
        if not required.issubset(connected):
            return False

    return True


def validate(cg: CircuitGraph) -> Tuple[bool, List[str]]:
    """
    Run all physical validity checks.
    Returns (is_valid, list_of_violations).
    """
    violations = []
    if not check_vdd_gnd_isolation(cg):
        violations.append("Rule1: direct VDD-GND short-circuit path detected")
    if not check_no_floating_gate(cg):
        violations.append("Rule2: floating gate detected on one or more MOS devices")
    if not check_bipartite_consistency(cg):
        violations.append("BipartiteConsistency: dangling port or non-bipartite edge")
    return len(violations) == 0, violations


# ──────────────────────────────────────────────────────────────────────────────
# 14. Main generator class
# ──────────────────────────────────────────────────────────────────────────────

class AnalogCircuitGenerator:
    """
    Three-phase analog circuit graph generator.

    Usage::

        gen = AnalogCircuitGenerator(tech_name="180nm")
        cg  = gen.generate(target_devices=100)
        ok, violations = gen.validate(cg)
    """

    def __init__(self,
                 tech_name:    str = "180nm",
                 min_devices:  int = 50,
                 max_devices:  int = 600,
                 max_attempts: int = 10,
                 seed:         Optional[int] = None):
        if tech_name not in TECH_NODES:
            raise ValueError("Unknown tech node '%s'. Choose from: %s"
                             % (tech_name, list(TECH_NODES.keys())))
        self.tech         = TECH_NODES[tech_name]
        self.min_devices  = min_devices
        self.max_devices  = max_devices
        self.max_attempts = max_attempts
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ── public API ───────────────────────────────────────────────────────

    def generate(self,
                 target_devices: Optional[int] = None,
                 template: Optional[Dict]       = None) -> CircuitGraph:
        """
        Full three-phase generation with physical validity enforcement.

        Returns a valid *CircuitGraph*.  Raises RuntimeError if a valid
        circuit cannot be produced within *max_attempts* retries.
        """
        if target_devices is None:
            target_devices = random.randint(self.min_devices, self.max_devices)

        for attempt in range(self.max_attempts):
            cg = CircuitGraph(self.tech)

            # Ensure power rails exist
            cg.add_net("VDD")
            cg.add_net("GND")

            # Phase 1: seed injection
            hints = inject_seed(cg, template)

            # Apply sizing hints from the seed template
            self._apply_seed_sizing(cg, hints)

            # Phase 2: node expansion
            expand_nodes(cg, target_devices)

            # Phase 3: topology completion
            complete_topology(cg)

            # Apply structural sizing constraints
            apply_all_sizing_constraints(cg)

            # Validate
            ok, violations = validate(cg)
            if ok:
                return cg

            # Otherwise retry (violations should be rare after completion)

        raise RuntimeError(
            "Could not generate a valid circuit after %d attempts. "
            "Last violations: %s" % (self.max_attempts, violations)
        )

    def validate(self, cg: CircuitGraph) -> Tuple[bool, List[str]]:
        return validate(cg)

    # ── private helpers ───────────────────────────────────────────────────

    def _apply_seed_sizing(self, cg: CircuitGraph, hints: Dict):
        """Apply sizing rules guided by template hints."""
        # Differential pairs
        for d1, d2 in hints.get("diff_pairs", []):
            if d1 in cg._dev_index and d2 in cg._dev_index:
                W = _rand_W(self.tech)
                L = _rand_L(self.tech)
                _update_device_size(cg, d1, W=W, L=L)
                _update_device_size(cg, d2, W=W, L=L)

        # Current mirrors
        for group in hints.get("current_mirrors", []):
            group = [d for d in group if d in cg._dev_index]
            if len(group) >= 2:
                ref_L  = _rand_L(self.tech)
                base_W = _rand_W(self.tech)
                for dev_id in group:
                    ratio = random.randint(1, 8)
                    _update_device_size(cg, dev_id,
                                       W=base_W * ratio,
                                       L=ref_L,
                                       multiplier=ratio)

        # Inverter pairs
        for pmos_id, nmos_id in hints.get("inverter_pairs", []):
            if pmos_id in cg._dev_index and nmos_id in cg._dev_index:
                Wn    = _rand_W(self.tech)
                ratio = random.uniform(2.0, 3.0)
                L     = _rand_L(self.tech)
                _update_device_size(cg, nmos_id, W=Wn,        L=L)
                _update_device_size(cg, pmos_id, W=Wn * ratio, L=L)


# ──────────────────────────────────────────────────────────────────────────────
# 15. igraph export helpers
# ──────────────────────────────────────────────────────────────────────────────

def circuit_graph_to_igraph(cg: CircuitGraph) -> ig.Graph:
    """
    Return the underlying igraph.Graph with all attributes serialised to
    GML-compatible types (int/float/str).
    """
    g = cg.g.copy()

    # Serialise feature_vector list to a JSON string for GML compatibility
    for v in g.vs:
        fv = v["feature_vector"]
        v["feature_vector"] = json.dumps([round(x, 6) for x in fv])
        # Ensure all attributes are GML-safe
        v["node_type"]      = str(v["node_type"])
        v["net_name"]       = str(v["net_name"])
        v["device_type"]    = int(v["device_type"])
        v["voltage_domain"] = int(v["voltage_domain"])
        v["W"]              = float(v["W"])
        v["L"]              = float(v["L"])
        v["multiplier"]     = int(v["multiplier"])
        v["num_fingers"]    = int(v["num_fingers"])
        v["body_tied"]      = int(v["body_tied"])  # GML has no bool

    for e in g.es:
        e["port_type"] = int(e["port_type"])
        e["port_name"] = str(e["port_name"])

    return g


def save_circuit_graph(cg: CircuitGraph, path: str):
    """Save circuit graph to a GML file."""
    g = circuit_graph_to_igraph(cg)
    g.write(path)


# ──────────────────────────────────────────────────────────────────────────────
# 16. CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Physics-aware analog circuit graph generator")
    parser.add_argument("--tech",           type=str, default="180nm",
                        choices=list(TECH_NODES.keys()),
                        help="Technology node")
    parser.add_argument("--num_graphs",     type=int, default=5,
                        help="Number of graphs to generate")
    parser.add_argument("--min_devices",    type=int, default=50)
    parser.add_argument("--max_devices",    type=int, default=600)
    parser.add_argument("--target_devices", type=int, default=None,
                        help="Fixed device count (overrides min/max)")
    parser.add_argument("--template",       type=str, default=None,
                        choices=[t["name"] for t in TEMPLATES],
                        help="Force a specific seed template")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--save_dir",       type=str, default="./data/analog",
                        help="Directory for output GML files")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Optionally look up template dict by name
    forced_template = None
    if args.template:
        forced_template = next(t for t in TEMPLATES if t["name"] == args.template)

    gen = AnalogCircuitGenerator(
        tech_name=args.tech,
        min_devices=args.min_devices,
        max_devices=args.max_devices,
        seed=args.seed,
    )

    for i in range(args.num_graphs):
        cg = gen.generate(target_devices=args.target_devices,
                          template=forced_template)
        ok, violations = gen.validate(cg)

        out_path = os.path.join(
            args.save_dir,
            "circuit_%s_N%d_%d.gml" % (args.tech, cg.num_devices(), i))
        save_circuit_graph(cg, out_path)

        print("[%d] devices=%d  nets=%d  edges=%d  valid=%s  path=%s"
              % (i, cg.num_devices(), cg.num_nets(),
                 cg.g.ecount(), ok, out_path))
        if not ok:
            print("    violations:", violations)
