"""Geometry controllers for closed-loop steering in latent-space decoding."""

from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return maximum if value > maximum else minimum if value < minimum else value


class GeometryController:
    """Base geometry controller interface."""

    mode = "base"

    def step(self, forward_kl: float) -> dict[str, float | int | bool]:
        raise NotImplementedError


@dataclass
class LegacyGeometryController(GeometryController):
    """Proportional band controller matching prior ad-hoc adaptation behavior."""

    mode: str = "legacy"
    eta: float = 0.05
    target_forward_kl: float = 0.06
    kl_tolerance: float = 0.5
    eta_min: float = 0.01
    eta_max: float = 0.5
    eta_growth: float = 1.06
    eta_decay: float = 0.85

    def step(self, forward_kl: float) -> dict[str, float | int | bool]:
        eta_before = self.eta

        if self.target_forward_kl > 0:
            low = max(0.0, self.target_forward_kl * (1.0 - self.kl_tolerance))
            high = self.target_forward_kl * (1.0 + self.kl_tolerance)
        else:
            low = 0.0
            high = max(0.0, self.kl_tolerance)

        if forward_kl > high:
            self.eta = max(self.eta_min, self.eta * self.eta_decay)
        elif forward_kl < low:
            self.eta = min(self.eta_max, self.eta * self.eta_growth)

        self.eta = _clamp(self.eta, self.eta_min, self.eta_max)
        return {
            "mode": self.mode,
            "eta_before": float(eta_before),
            "eta_after": float(self.eta),
            "eta_delta": float(self.eta - eta_before),
            "target_forward_kl": float(self.target_forward_kl),
            "low_band": float(low),
            "high_band": float(high),
            "forward_kl_error": float((self.target_forward_kl - forward_kl) if self.target_forward_kl else (0.0 - forward_kl)),
            "within_band": 1 if (low <= forward_kl <= high) else 0,
        }


@dataclass
class PIDGeometryController(GeometryController):
    """PID-like controller for smoother eta updates."""

    mode: str = "pid"
    eta: float = 0.05
    target_forward_kl: float = 0.06
    kl_tolerance: float = 0.5
    eta_min: float = 0.01
    eta_max: float = 0.5
    eta_growth: float = 1.06
    eta_decay: float = 0.85
    kp: float = 0.25
    ki: float = 0.0
    kd: float = 0.0
    ema_alpha: float = 0.2
    integral: float = 0.0
    prev_error: float = 0.0
    smoothed_error: float = 0.0

    def step(self, forward_kl: float) -> dict[str, float | int | bool]:
        eta_before = self.eta

        if self.target_forward_kl > 0:
            low = max(0.0, self.target_forward_kl * (1.0 - self.kl_tolerance))
            high = self.target_forward_kl * (1.0 + self.kl_tolerance)
            normalized_error = (self.target_forward_kl - forward_kl) / max(self.target_forward_kl, 1e-6)
        else:
            low = 0.0
            high = max(0.0, self.kl_tolerance)
            normalized_error = (-forward_kl) / max(high if high > 0 else 1e-6, 1e-6)

        self.smoothed_error = float(self.ema_alpha * normalized_error + (1.0 - self.ema_alpha) * self.smoothed_error)
        self.integral = _clamp(self.integral + self.smoothed_error, -5.0, 5.0)
        derivative = self.smoothed_error - self.prev_error
        self.prev_error = self.smoothed_error

        step_ratio = 1.0 + self.kp * self.smoothed_error + self.ki * self.integral + self.kd * derivative
        proposed_eta = self.eta * step_ratio
        proposed_eta = _clamp(proposed_eta, self.eta_min, self.eta_max)

        if forward_kl > high:
            self.eta = min(proposed_eta, self.eta * self.eta_decay)
        elif forward_kl < low:
            self.eta = max(proposed_eta, self.eta * self.eta_growth)
        else:
            self.eta = proposed_eta

        self.eta = _clamp(self.eta, self.eta_min, self.eta_max)
        return {
            "mode": self.mode,
            "eta_before": float(eta_before),
            "eta_after": float(self.eta),
            "eta_delta": float(self.eta - eta_before),
            "target_forward_kl": float(self.target_forward_kl),
            "low_band": float(low),
            "high_band": float(high),
            "normalized_error": float(self.smoothed_error),
            "integral_error": float(self.integral),
            "derivative_error": float(derivative),
            "within_band": 1 if (low <= forward_kl <= high) else 0,
        }


def build_geometry_controller(
    *,
    mode: str,
    steering_eta: float,
    target_forward_kl: float,
    kl_tolerance: float,
    eta_min: float,
    eta_max: float,
    eta_growth: float,
    eta_decay: float,
    kp: float = 0.0,
    ki: float = 0.0,
    kd: float = 0.0,
    ema_alpha: float = 0.2,
) -> GeometryController:
    """Create a geometry controller from compact configuration."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode in {"legacy", "band", "adaptive", "old"}:
        return LegacyGeometryController(
            eta=steering_eta,
            target_forward_kl=target_forward_kl,
            kl_tolerance=kl_tolerance,
            eta_min=eta_min,
            eta_max=eta_max,
            eta_growth=eta_growth,
            eta_decay=eta_decay,
        )
    if normalized_mode in {"pid", "pid_plus", "smoothed"}:
        return PIDGeometryController(
            eta=steering_eta,
            target_forward_kl=target_forward_kl,
            kl_tolerance=kl_tolerance,
            eta_min=eta_min,
            eta_max=eta_max,
            eta_growth=eta_growth,
            eta_decay=eta_decay,
            kp=kp,
            ki=ki,
            kd=kd,
            ema_alpha=ema_alpha,
        )
    raise ValueError(
        "Unsupported geometry_controller. Supported values: legacy, pid"
    )
