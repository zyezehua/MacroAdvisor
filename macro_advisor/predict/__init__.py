"""Prediction layer (Phase 2): label construction + walk-forward OOS engine.

All labels are computed from data available at decision time only; validation uses
expanding/rolling walk-forward with purged + embargoed splits to prevent leakage.
"""
