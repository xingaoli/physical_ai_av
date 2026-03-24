#!/usr/bin/env python3
"""
Shared configuration for meta-action detection and visualization.

This module contains all thresholds and semantic group definitions
used by both the annotation script and the visualization script.
"""

from dataclasses import dataclass, field


@dataclass
class MetaActionConfig:
    """Configuration thresholds for meta-action detection."""

    # Longitudinal thresholds (m/s²)
    strong_accel_threshold: float = 2.0
    gentle_accel_threshold: float = 1.0
    gentle_decel_threshold: float = -1.0
    strong_decel_threshold: float = -2.0
    stop_speed_threshold: float = 0.5  # m/s

    # Lateral thresholds - Curvature based (1/m, recommended)
    # Curvature = 1/turning_radius, coordinate-system independent
    # |κ| < 0.01: straight (radius > 100m)
    # 0.01 ≤ |κ| < 0.05: gentle turn (20m-100m radius)
    # |κ| ≥ 0.05: sharp turn (radius < 20m)
    sharp_steer_threshold_curvature: float = 0.05   # 1/m
    gentle_steer_threshold_curvature: float = 0.01  # 1/m

    # Lateral thresholds - Yaw rate based (rad/s, noisy)
    # Only used if use_yaw_rate = True
    sharp_steer_threshold_yaw_rate: float = 0.1    # ~5.7 deg/s
    gentle_steer_threshold_yaw_rate: float = 0.02   # ~1.1 deg/s

    # Smoothing parameters for raw sensor data
    acceleration_window: int = 5  # frames for median filter
    curvature_window: int = 5     # frames for median filter

    # Short state filtering: removes noise and insignificant state changes
    # Uses forward propagation to absorb short states into previous state
    # - 8 frames (0.8s) for longitudinal: filters noise, keeps real braking events
    # - 5 frames (0.5s) for lateral: filters noise, keeps real lane changes
    min_state_duration_long: int = 8
    min_state_duration_lat: int = 5

    # Use curvature for lateral detection (recommended, more stable than yaw_rate)
    use_yaw_rate: bool = False    # Set to True to use yaw_rate instead (noisy)

    # Self-terminating states: entry is a decision point, exit is not
    # When transitioning from these states back to terminal state, the exit keyframe is suppressed
    self_terminating_long: tuple = ('decelerating', 'accelerating')
    self_terminating_lat: tuple = ('turning_left', 'turning_right')

    # Terminal states to return to (exit points from self-terminating states are suppressed)
    terminal_return_long: str = 'cruise'
    terminal_return_lat: str = 'straight'

    # Whitelist for suppressing accompanying/secondary keyframes
    # Key: the dimension that stays unchanged
    # Value: changes in the other dimension that should be suppressed

    # When lateral state is unchanged, suppress these longitudinal changes
    suppress_if_lat_unchanged: dict = field(default_factory=lambda: {
        'turning_left': {'accelerating', 'cruise'},   # 左转时，加速或回匀速是伴随
        'turning_right': {'accelerating', 'cruise'},   # 右转时，加速或回匀速是伴随
        # 当巡航跟车发生了减速等关键帧，再加速时会被抑制；全程巡航偶有一个加速也会被抑制；
        # 但像停车起步，转弯后直行（这个直行转换关键帧会被自抑制）一段再加速不会被抑制
        'straight': {'accelerating'},                  
    })

    # When longitudinal state is unchanged, suppress these lateral changes
    # Note: turn_left/turn_right are NOT included here - starting a turn is a new decision!
    suppress_if_long_unchanged: dict = field(default_factory=dict)
    # suppress_if_long_unchanged: dict = field(default_factory=lambda: {
    #     'decelerating': {'turning_left', 'turning_right'},   # 减速时，左转和右转是伴随
    # })

    # Minimum gap between keyframes to merge nearby transitions
    # When longitudinal and lateral transitions occur within this window,
    # they are treated as the same decision-making moment
    min_transition_gap: int = 5  # frames (0.5s)

# Semantic group mapping for keyframe detection
# Keyframes are detected only when there's a semantic group transition
# This prevents Gentle decel → Strong decel from creating spurious keyframes
LONG_SEMANTIC_GROUP = {
    'Strong accelerate': 'accelerating',
    'Gentle accelerate': 'accelerating',  # Same intent
    'Maintain speed':    'cruise',
    'Gentle decelerate': 'decelerating',
    'Strong decelerate': 'decelerating',  # Same intent
    'Stop':              'stopped',
    'Reverse':           'reverse',
}

LAT_SEMANTIC_GROUP = {
    'Sharp steer left':   'turning_left',
    'Steer left':         'turning_left',   # Same intent
    'Go straight':        'straight',
    'Sharp steer right':  'turning_right',
    'Steer right':        'turning_right',  # Same intent
    'Reverse left':       'reverse_left',
    'Reverse right':      'reverse_right',
    'Reverse straight':   'reverse_straight',
}


# Color maps for visualization
LONG_COLORS = {
    'Stop': '#FF6B6B',
    'Maintain speed': '#95A5A6',
    'Gentle accelerate': '#90EE90',
    'Strong accelerate': '#2ECC71',
    'Gentle decelerate': '#FFB6C1',
    'Strong decelerate': '#E74C3C',
    'Reverse': '#9B59B6',
}

LAT_COLORS = {
    'Go straight': '#95A5A6',
    'Steer left': '#ADD8E6',
    'Steer right': '#FFFACD',
    'Sharp steer left': '#3498DB',
    'Sharp steer right': '#F1C40F',
    'Reverse left': '#9B59B6',
    'Reverse right': '#E67E22',
    'Reverse straight': '#8E44AD',
}


# Semantic group colors for visualization (coarse-grained)
LONG_SEMANTIC_COLORS = {
    'accelerating': '#2ECC71',   # Green shades
    'cruise': '#95A5A6',         # Gray (maintaining speed)
    'decelerating': '#E74C3C',   # Red shades
    'stopped': '#FF6B6B',        # Bright red
    'reverse': '#9B59B6',        # Purple
}

LAT_SEMANTIC_COLORS = {
    'turning_left': '#3498DB',   # Blue
    'straight': '#95A5A6',       # Gray
    'turning_right': '#F1C40F',  # Yellow/orange
    'reverse_left': '#9B59B6',   # Purple
    'reverse_right': '#E67E22',  # Orange
    'reverse_straight': '#8E44AD', # Dark purple
}
