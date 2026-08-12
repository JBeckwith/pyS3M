#!/usr/bin/env python3
"""
Full coverage tests for pyS3M.simulation.diffusion -- 2D Langevin diffusion,
Gillespie binding/unbinding kinetics, MSD analysis, and the CameraAdapter
bridge to camera-image rendering.

Deliberately tiny throughout: 2-4 molecules, 2-5 simulation steps, small
areas -- these are unit tests for branch coverage, not statistically
meaningful diffusion/kinetics validation.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import pyS3M.simulation.diffusion as diff
from pyS3M.simulation.diffusion import (
    BindingKinetics,
    CameraAdapter,
    DiffusionSimulator2D,
    LangevinDiffusion2D,
    Molecule,
    compute_msd_from_trajectory,
    estimate_D_from_msd,
)


# ======================================================================
# Molecule
# ======================================================================

class TestMolecule:
    def test_default_profile_known_colors(self):
        for color in ["R", "G", "B"]:
            mol = Molecule(0, color, np.array([1.0, 2.0]), D_free=1.0)
            assert set(mol.spectral_profile.keys()) == {"A_R", "A_G", "A_B"}
        assert Molecule(0, "R", np.array([0.0, 0.0]), 1.0).spectral_profile["A_R"] == 0.80

    def test_default_profile_unknown_color(self):
        mol = Molecule(0, "X", np.array([0.0, 0.0]), D_free=1.0)
        assert mol.spectral_profile["A_R"] == pytest.approx(0.33)

    def test_explicit_profile_not_overwritten(self):
        profile = {"A_R": 0.1, "A_G": 0.2, "A_B": 0.7}
        mol = Molecule(0, "R", np.array([0.0, 0.0]), 1.0, spectral_profile=profile)
        assert mol.spectral_profile == profile

    def test_trajectory_auto_initialised(self):
        pos = np.array([5.0, 6.0])
        mol = Molecule(0, "R", pos, 1.0)
        assert len(mol.trajectory) == 1
        assert mol.times == [0.0]
        np.testing.assert_array_equal(mol.trajectory[0], pos)

    def test_trajectory_not_overwritten_if_provided(self):
        pos = np.array([5.0, 6.0])
        mol = Molecule(0, "R", pos, 1.0, trajectory=[pos.copy(), pos.copy()], times=[0.0, 1.0])
        assert len(mol.trajectory) == 2

    def test_get_current_D(self):
        mol = Molecule(0, "R", np.array([0.0, 0.0]), D_free=1.0, D_bound=0.1)
        assert mol.get_current_D() == 1.0
        mol.is_bound = True
        assert mol.get_current_D() == 0.1

    def test_add_position(self):
        mol = Molecule(0, "R", np.array([0.0, 0.0]), D_free=1.0)
        new_pos = np.array([9.0, 9.0])
        mol.add_position(new_pos, time=5.0)
        assert len(mol.trajectory) == 2
        assert mol.times[-1] == 5.0
        np.testing.assert_array_equal(mol.position, new_pos)


# ======================================================================
# BindingKinetics
# ======================================================================

def _bk(colors=("R", "G"), k_on=1.0, k_off=1.0, binding_radius=100.0):
    n = len(colors)
    k_on_matrix = np.full((n, n), k_on)
    k_off_matrix = np.full((n, n), k_off)
    return BindingKinetics(list(colors), k_on_matrix=k_on_matrix,
                            k_off_matrix=k_off_matrix, binding_radius=binding_radius)


class TestBindingKineticsInit:
    def test_missing_matrices_raises(self):
        with pytest.raises(ValueError, match="requires"):
            BindingKinetics(["R", "G"])

    def test_shape_mismatch_asserts(self):
        with pytest.raises(AssertionError):
            BindingKinetics(["R", "G"], k_on_matrix=np.ones((3, 3)), k_off_matrix=np.ones((2, 2)))

    def test_symmetrised(self):
        k_on = np.array([[0.0, 2.0], [0.0, 0.0]])
        k_off = np.array([[0.0, 4.0], [0.0, 0.0]])
        bk = BindingKinetics(["R", "G"], k_on_matrix=k_on, k_off_matrix=k_off)
        assert bk.k_on_matrix[0, 1] == bk.k_on_matrix[1, 0] == 1.0
        assert bk.k_off_matrix[0, 1] == bk.k_off_matrix[1, 0] == 2.0


class TestBindingKineticsCanBind:
    def test_both_unbound_nonzero_rate(self):
        bk = _bk()
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0)
        m2 = Molecule(1, "G", np.array([1.0, 1.0]), 1.0)
        assert bk.can_bind(m1, m2) == True

    def test_one_bound_returns_false(self):
        bk = _bk()
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0, is_bound=True)
        m2 = Molecule(1, "G", np.array([1.0, 1.0]), 1.0)
        assert bk.can_bind(m1, m2) == False

    def test_zero_rate_returns_false(self):
        bk = BindingKinetics(["R", "G"], k_on_matrix=np.zeros((2, 2)), k_off_matrix=np.ones((2, 2)))
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0)
        m2 = Molecule(1, "G", np.array([1.0, 1.0]), 1.0)
        assert bk.can_bind(m1, m2) == False


class TestBindingKineticsRates:
    def test_get_binding_and_unbinding_rate(self):
        bk = _bk(k_on=3.0, k_off=7.0)
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0)
        m2 = Molecule(1, "G", np.array([1.0, 1.0]), 1.0)
        assert bk.get_binding_rate(m1, m2) == 3.0
        assert bk.get_unbinding_rate(m1, m2) == 7.0


class TestFindPotentialBindingPairs:
    def test_fewer_than_two_unbound_returns_empty(self):
        bk = _bk()
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0, is_bound=True)
        assert bk.find_potential_binding_pairs([m1]) == []

    def test_within_radius_and_can_bind(self):
        bk = _bk(binding_radius=10.0)
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0)
        m2 = Molecule(1, "G", np.array([5.0, 0.0]), 1.0)
        m3 = Molecule(2, "G", np.array([500.0, 0.0]), 1.0)  # too far
        pairs = bk.find_potential_binding_pairs([m1, m2, m3])
        assert len(pairs) == 1
        assert pairs[0][:2] == (m1, m2)


class TestCalculatePropensities:
    def test_empty_when_no_pairs_or_bound(self):
        bk = _bk()
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0)
        bind, unbind, total = bk.calculate_propensities([m1], area=1e6)
        assert bind == [] and unbind == [] and total == 0.0

    def test_binding_and_unbinding_propensities(self):
        bk = _bk(binding_radius=10.0)
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0)
        m2 = Molecule(1, "G", np.array([5.0, 0.0]), 1.0)
        m3 = Molecule(2, "R", np.array([0.0, 0.0]), 1.0, is_bound=True)
        m4 = Molecule(3, "G", np.array([0.0, 0.0]), 1.0, is_bound=True)
        m3.bound_partner = m4
        m4.bound_partner = m3
        bind, unbind, total = bk.calculate_propensities([m1, m2, m3, m4], area=1e6)
        assert len(bind) == 1
        assert len(unbind) == 1
        assert total > 0


class TestGetBoundPairs:
    def test_dedups_pairs(self):
        bk = _bk()
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0, is_bound=True)
        m2 = Molecule(1, "G", np.array([0.0, 0.0]), 1.0, is_bound=True)
        m1.bound_partner, m2.bound_partner = m2, m1
        pairs = bk._get_bound_pairs([m1, m2])
        assert len(pairs) == 1

    def test_unbound_or_no_partner_excluded(self):
        bk = _bk()
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0)
        m2 = Molecule(1, "G", np.array([0.0, 0.0]), 1.0, is_bound=True, bound_partner=None)
        assert bk._get_bound_pairs([m1, m2]) == []


class TestChooseEvent:
    def test_zero_total_returns_none(self):
        bk = _bk()
        assert bk.choose_event([], [], 0.0) is None

    def test_binding_event_chosen(self, monkeypatch):
        bk = _bk()
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0)
        m2 = Molecule(1, "G", np.array([1.0, 1.0]), 1.0)
        monkeypatch.setattr(np.random, "uniform", lambda a, b: 0.5)
        event = bk.choose_event([(m1, m2, 1.0)], [], 1.0)
        assert event == ("bind", m1, m2)

    def test_unbinding_event_chosen(self, monkeypatch):
        bk = _bk()
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0)
        m2 = Molecule(1, "G", np.array([1.0, 1.0]), 1.0)
        monkeypatch.setattr(np.random, "uniform", lambda a, b: 1.5)
        event = bk.choose_event([(m1, m2, 1.0)], [(m1, m2, 1.0)], 2.0)
        assert event == ("unbind", m1, m2)

    def test_falls_through_returns_none(self, monkeypatch):
        bk = _bk()
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0)
        m2 = Molecule(1, "G", np.array([1.0, 1.0]), 1.0)
        # r drawn exactly at (beyond) the summed propensities -- no branch's
        # cumsum ever exceeds it, falling through both loops.
        monkeypatch.setattr(np.random, "uniform", lambda a, b: 10.0)
        event = bk.choose_event([(m1, m2, 1.0)], [(m1, m2, 1.0)], 2.0)
        assert event is None


class TestExecuteBindingUnbinding:
    def test_execute_binding(self):
        bk = _bk()
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0)
        m2 = Molecule(1, "G", np.array([2.0, 2.0]), 1.0)
        bk.execute_binding(m1, m2, time=1.0)
        assert m1.is_bound and m2.is_bound
        assert m1.bound_partner is m2 and m2.bound_partner is m1
        assert len(bk.binding_events) == 1

    def test_execute_unbinding(self):
        bk = _bk()
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0, is_bound=True)
        m2 = Molecule(1, "G", np.array([2.0, 2.0]), 1.0, is_bound=True)
        m1.bound_partner, m2.bound_partner = m2, m1
        bk.execute_unbinding(m1, m2, time=2.0)
        assert not m1.is_bound and not m2.is_bound
        assert m1.bound_partner is None and m2.bound_partner is None
        assert len(bk.unbinding_events) == 1


class TestProcessEvents:
    def test_zero_propensity_returns_zero(self):
        bk = BindingKinetics(["R", "G"], k_on_matrix=np.zeros((2, 2)), k_off_matrix=np.zeros((2, 2)))
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0)
        m2 = Molecule(1, "G", np.array([1.0, 1.0]), 1.0)
        assert bk.process_events([m1, m2], dt=10.0, area=1e6, current_time=0.0) == 0

    def test_events_occur_within_timestep(self):
        # Very high k_on and tiny area -> huge propensity -> tau << dt,
        # so at least one binding event should occur.
        bk = _bk(k_on=1e12, k_off=1e-6, binding_radius=1000.0)
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0)
        m2 = Molecule(1, "G", np.array([1.0, 1.0]), 1.0)
        n_events = bk.process_events([m1, m2], dt=1000.0, area=1.0, current_time=0.0)
        assert n_events >= 1
        assert m1.is_bound and m2.is_bound

    def test_no_events_when_tau_exceeds_dt(self):
        bk = _bk(k_on=1e-9, k_off=1e-9, binding_radius=1000.0)
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0)
        m2 = Molecule(1, "G", np.array([1.0, 1.0]), 1.0)
        n_events = bk.process_events([m1, m2], dt=0.001, area=1e12, current_time=0.0)
        assert n_events == 0

    def test_choose_event_none_breaks_loop(self, monkeypatch):
        bk = _bk(k_on=1e12, k_off=1e-6, binding_radius=1000.0)
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0)
        m2 = Molecule(1, "G", np.array([1.0, 1.0]), 1.0)
        monkeypatch.setattr(bk, "choose_event", lambda *a, **kw: None)
        n_events = bk.process_events([m1, m2], dt=1000.0, area=1.0, current_time=0.0)
        assert n_events == 0

    def test_unbind_event_within_timestep(self):
        # Start already bound with a huge k_off -> unbinding fires quickly.
        bk = _bk(k_on=1e-9, k_off=1e12, binding_radius=1000.0)
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0, is_bound=True)
        m2 = Molecule(1, "G", np.array([1.0, 1.0]), 1.0, is_bound=True)
        m1.bound_partner, m2.bound_partner = m2, m1
        n_events = bk.process_events([m1, m2], dt=1000.0, area=1e12, current_time=0.0)
        assert n_events >= 1
        assert not m1.is_bound and not m2.is_bound

    def test_total_propensity_drops_to_zero_after_event(self):
        # Only two molecules, k_off=0 -> after they bind there is nothing
        # left to bind (no other unbound molecules) and nothing to unbind
        # (k_off=0), so total_prop hits exactly 0 right after the one event.
        bk = _bk(k_on=1e12, k_off=0.0, binding_radius=1000.0)
        m1 = Molecule(0, "R", np.array([0.0, 0.0]), 1.0)
        m2 = Molecule(1, "G", np.array([1.0, 1.0]), 1.0)
        n_events = bk.process_events([m1, m2], dt=1000.0, area=1.0, current_time=0.0)
        assert n_events == 1
        assert m1.is_bound and m2.is_bound


# ======================================================================
# LangevinDiffusion2D
# ======================================================================

class TestLangevinDiffusion2D:
    def test_compute_dynamic_localization_error(self):
        ld = LangevinDiffusion2D(sigma0=10.0, s0=100.0)
        sigma = ld.compute_dynamic_localization_error(D=0.0, t_exposure=10.0)
        assert sigma == pytest.approx(10.0)
        sigma2 = ld.compute_dynamic_localization_error(D=1000.0, t_exposure=10.0)
        assert sigma2 > sigma

    def test_generate_displacement_1D_short(self):
        ld = LangevinDiffusion2D(sigma0=10.0, s0=100.0)
        out = ld.generate_displacement_1D(D=1.0, N=1, dt=1.0, t_exposure=1.0)
        np.testing.assert_array_equal(out, [0.0])

    def test_generate_displacement_1D_normal(self):
        ld = LangevinDiffusion2D(sigma0=10.0, s0=100.0)
        out = ld.generate_displacement_1D(D=1000.0, N=5, dt=1.0, t_exposure=1.0)
        assert out.shape == (5,)
        assert out[0] == 0.0

    def test_generate_trajectory_2D(self):
        ld = LangevinDiffusion2D(sigma0=10.0, s0=100.0)
        traj = ld.generate_trajectory_2D(
            D=1000.0, N=4, dt=1.0, t_exposure=1.0, starting_position=np.array([5.0, 5.0]),
        )
        assert traj.shape == (4, 2)
        np.testing.assert_allclose(traj[0], [5.0, 5.0])


# ======================================================================
# DiffusionSimulator2D
# ======================================================================

class TestDiffusionSimulator2DBasics:
    def _sim(self, boundary="reflective", binding_kinetics=None):
        return DiffusionSimulator2D(
            area=(1000.0, 1000.0), dt=1.0, t_exposure=1.0,
            sigma0=5.0, s0=100.0, boundary=boundary, binding_kinetics=binding_kinetics,
        )

    def test_add_molecule(self):
        sim = self._sim()
        mol = sim.add_molecule("R", np.array([10.0, 10.0]), D_free=1000.0)
        assert mol.molecule_id == 0
        assert len(sim.molecules) == 1

    def test_add_molecules_random(self):
        sim = self._sim()
        mols = sim.add_molecules_random(3, "G", D_free=500.0)
        assert len(mols) == 3
        assert len(sim.molecules) == 3

    def test_boundary_periodic(self):
        sim = self._sim(boundary="periodic")
        pos = sim._apply_boundary_condition(np.array([-10.0, 1010.0]))
        assert 0 <= pos[0] < 1000.0
        assert 0 <= pos[1] < 1000.0

    def test_boundary_reflective_both_sides(self):
        sim = self._sim(boundary="reflective")
        pos = sim._apply_boundary_condition(np.array([-10.0, 1010.0]))
        np.testing.assert_allclose(pos, [10.0, 990.0])

    def test_boundary_absorbing_noop(self):
        sim = self._sim(boundary="absorbing")
        pos = sim._apply_boundary_condition(np.array([-10.0, 1010.0]))
        np.testing.assert_allclose(pos, [-10.0, 1010.0])

    def test_update_bound_pair_positions(self):
        sim = self._sim()
        m1 = sim.add_molecule("R", np.array([0.0, 0.0]), 1.0)
        m2 = sim.add_molecule("G", np.array([1.0, 1.0]), 1.0)
        new_pos = np.array([5.0, 5.0])
        sim._update_bound_pair_positions(m1, m2, new_pos)
        np.testing.assert_array_equal(m1.position, new_pos)
        np.testing.assert_array_equal(m2.position, new_pos)

    def test_get_trajectory_and_all(self):
        sim = self._sim()
        sim.add_molecule("R", np.array([10.0, 10.0]), D_free=1000.0)
        sim.run(n_steps=3, enable_binding=False)
        positions, times = sim.get_trajectory(0)
        assert positions.shape[0] == 4  # initial + 3 steps
        assert len(times) == 4
        all_traj = sim.get_all_trajectories()
        assert 0 in all_traj

    def test_reset(self):
        sim = self._sim()
        sim.add_molecule("R", np.array([10.0, 10.0]), D_free=1000.0)
        sim.run(n_steps=2, enable_binding=False)
        sim.reset()
        assert sim.current_time == 0.0
        assert len(sim.molecules[0].trajectory) == 1
        assert sim.molecules[0].is_bound is False


class TestDiffusionSimulator2DRun:
    def test_run_no_binding_disabled_flag(self):
        sim = DiffusionSimulator2D(area=(1000.0, 1000.0), dt=1.0, t_exposure=1.0, sigma0=5.0, s0=100.0)
        sim.add_molecule("R", np.array([10.0, 10.0]), D_free=1000.0)
        sim.run(n_steps=3, enable_binding=False)
        assert sim.current_time == 3.0
        assert len(sim.molecules[0].trajectory) == 4

    def test_run_no_binding_kinetics_object(self):
        sim = DiffusionSimulator2D(area=(1000.0, 1000.0), dt=1.0, t_exposure=1.0, sigma0=5.0, s0=100.0)
        sim.add_molecule("R", np.array([10.0, 10.0]), D_free=1000.0)
        sim.run(n_steps=2, enable_binding=True)  # binding_kinetics is None -> fast path
        assert sim.current_time == 2.0

    def test_run_with_binding_unbound_and_bound_pairs(self):
        bk = _bk(k_on=1e12, k_off=1e-9, binding_radius=1000.0)
        sim = DiffusionSimulator2D(
            area=(1000.0, 1000.0), dt=1.0, t_exposure=1.0, sigma0=5.0, s0=100.0,
            binding_kinetics=bk,
        )
        sim.add_molecule("R", np.array([10.0, 10.0]), D_free=1000.0)
        sim.add_molecule("G", np.array([12.0, 12.0]), D_free=1000.0)
        sim.add_molecule("R", np.array([500.0, 500.0]), D_free=1000.0)  # stays unbound
        # chunk_size=2 with n_steps=4 -> two chunks; forces binding to occur
        # in the first chunk (unbound branch) then the pair to move together
        # in the second chunk (bound-pair branch + processed_pairs skip).
        sim.run(n_steps=4, enable_binding=True, chunk_size=2)
        assert sim.current_time == 4.0
        assert all(len(mol.trajectory) == 5 for mol in sim.molecules)


# ======================================================================
# compute_msd_from_trajectory / estimate_D_from_msd
# ======================================================================

class TestMsdFunctions:
    def test_compute_msd_default_max_tau(self):
        traj = np.cumsum(np.random.default_rng(0).normal(0, 1, size=(40, 2)), axis=0)
        tau, msd = compute_msd_from_trajectory(traj)
        assert len(tau) == len(msd) == 10  # 40 // 4

    def test_compute_msd_explicit_max_tau_clipped(self):
        traj = np.cumsum(np.random.default_rng(0).normal(0, 1, size=(5, 2)), axis=0)
        tau, msd = compute_msd_from_trajectory(traj, max_tau=100)
        assert tau[-1] == 4  # clipped to N - 1

    def test_estimate_D_from_msd(self):
        tau = np.arange(1, 11)
        D_true = 2.0
        msd = 4 * D_true * tau  # 2D: MSD = 4D*tau
        D_est = estimate_D_from_msd(tau, msd, dt=1.0)
        assert D_est == pytest.approx(D_true, rel=1e-6)

    def test_estimate_D_from_msd_fit_points_clipped(self):
        tau = np.arange(1, 4)
        msd = 4.0 * tau
        D_est = estimate_D_from_msd(tau, msd, dt=1.0, fit_points=100)
        assert np.isfinite(D_est)


# ======================================================================
# CameraAdapter.prepare_localisations_for_imaging
# ======================================================================

class TestPrepareLocalisationsForImaging:
    def _sim_with_molecules(self):
        sim = DiffusionSimulator2D(area=(1000.0, 1000.0), dt=1.0, t_exposure=1.0, sigma0=5.0, s0=100.0)
        sim.add_molecule("R", np.array([10.0, 10.0]), D_free=500.0)
        sim.add_molecule("G", np.array([20.0, 20.0]), D_free=500.0)
        sim.run(n_steps=3, enable_binding=False)
        return sim

    def test_all_frames_default(self):
        sim = self._sim_with_molecules()
        adapter = CameraAdapter(sim)
        x0y0, n_photons, profiles = adapter.prepare_localisations_for_imaging(
            n_photons_per_dye={"R": 1000.0, "G": 800.0},
            random_state=np.random.default_rng(0),
        )
        assert set(x0y0.keys()) == {"dye_R", "dye_G"}
        assert x0y0["dye_R"].shape == (4, 2, 1)
        assert profiles["dye_R"].shape == (1, 3)

    def test_explicit_frame_indices(self):
        sim = self._sim_with_molecules()
        adapter = CameraAdapter(sim)
        x0y0, n_photons, profiles = adapter.prepare_localisations_for_imaging(
            n_photons_per_dye={"R": 1000.0, "G": 800.0},
            frame_indices=np.array([0, 2]),
            random_state=np.random.default_rng(0),
        )
        assert x0y0["dye_R"].shape == (2, 2, 1)

    def test_out_of_range_frame_index_skipped(self):
        sim = self._sim_with_molecules()
        adapter = CameraAdapter(sim)
        x0y0, n_photons, profiles = adapter.prepare_localisations_for_imaging(
            n_photons_per_dye={"R": 1000.0, "G": 800.0},
            frame_indices=np.array([0, 999]),
            random_state=np.random.default_rng(0),
        )
        # traj_idx=999 is out of range -> that row stays zero.
        assert np.all(x0y0["dye_R"][1] == 0.0)
        assert np.all(n_photons["dye_R"][1] == 0.0)

    def test_blinking_probability_visible_and_invisible(self, monkeypatch):
        sim = self._sim_with_molecules()
        adapter = CameraAdapter(sim)

        class _FakeRng:
            def __init__(self):
                self.calls = 0
            def random(self):
                self.calls += 1
                return 0.0 if self.calls % 2 == 1 else 1.0  # alternate visible/invisible
            def poisson(self, mean):
                return mean

        x0y0, n_photons, profiles = adapter.prepare_localisations_for_imaging(
            n_photons_per_dye={"R": 1000.0, "G": 800.0},
            blinking_probability={"R": 0.5},
            random_state=_FakeRng(),
        )
        assert n_photons["dye_R"].shape == (4, 1)

    def test_poisson_brightness_false_deterministic(self):
        sim = self._sim_with_molecules()
        adapter = CameraAdapter(sim)
        x0y0, n_photons, profiles = adapter.prepare_localisations_for_imaging(
            n_photons_per_dye={"R": 1000.0, "G": 800.0},
            poisson_brightness=False,
            random_state=np.random.default_rng(0),
        )
        assert np.all(n_photons["dye_R"] == 1000.0)

    def test_default_random_state(self):
        sim = self._sim_with_molecules()
        adapter = CameraAdapter(sim)
        x0y0, n_photons, profiles = adapter.prepare_localisations_for_imaging(
            n_photons_per_dye={"R": 1000.0, "G": 800.0},
        )
        assert "dye_R" in x0y0


# ======================================================================
# CameraAdapter.generate_ground_truth_rgb_video
# ======================================================================

class TestGenerateGroundTruthRgbVideo:
    def _sim_with_molecules(self, area=(200.0, 200.0)):
        sim = DiffusionSimulator2D(area=area, dt=1.0, t_exposure=1.0, sigma0=5.0, s0=100.0)
        sim.add_molecule("R", np.array([100.0, 100.0]), D_free=0.0)
        sim.add_molecule("B", np.array([50.0, 50.0]), D_free=0.0)
        sim.run(n_steps=2, enable_binding=False)
        return sim

    def test_spectral_colormap_no_save(self, tmp_path):
        sim = self._sim_with_molecules()
        adapter = CameraAdapter(sim)
        video = adapter.generate_ground_truth_rgb_video(
            output_path=str(tmp_path / "gt.tif"), pixel_size_nm=20.0, save_video=False,
        )
        assert video.dtype == np.uint8
        assert video.shape[0] == 3  # initial + 2 steps

    def test_direct_colormap(self, tmp_path):
        sim = self._sim_with_molecules()
        adapter = CameraAdapter(sim)
        video = adapter.generate_ground_truth_rgb_video(
            output_path=str(tmp_path / "gt.tif"), pixel_size_nm=20.0,
            colormap="direct", save_video=False,
        )
        assert video.shape[0] == 3

    def test_spectral_colormap_all_bands(self, tmp_path):
        # Default "G" molecule lands exactly on the Green->Yellow band
        # (spectral_pos=0.50); a custom profile lands on Cyan->Green
        # (spectral_pos=0.4). Covers both intermediate gradient branches
        # ("R"/"B" defaults only ever land in the outer two bands).
        sim = DiffusionSimulator2D(area=(200.0, 200.0), dt=1.0, t_exposure=1.0, sigma0=5.0, s0=100.0)
        sim.add_molecule("G", np.array([100.0, 100.0]), D_free=0.0)
        sim.add_molecule(
            "R", np.array([50.0, 50.0]), D_free=0.0,
            spectral_profile={"A_R": 0.2, "A_G": 0.4, "A_B": 0.4},
        )
        sim.run(n_steps=1, enable_binding=False)
        adapter = CameraAdapter(sim)
        video = adapter.generate_ground_truth_rgb_video(
            output_path=str(tmp_path / "gt.tif"), pixel_size_nm=20.0, save_video=False,
        )
        assert video.shape[0] == 2

    def test_molecule_out_of_fov_on_y_axis_only(self, tmp_path):
        # x within the FOV but y far outside it -> exercises the y-axis
        # out-of-view check specifically (the x-axis check alone doesn't
        # reach it, since it `continue`s first).
        sim = DiffusionSimulator2D(area=(200.0, 200.0), dt=1.0, t_exposure=1.0, sigma0=5.0, s0=100.0,
                                    boundary="absorbing")
        sim.add_molecule("R", np.array([100.0, 1.0e6]), D_free=0.0)
        sim.run(n_steps=1, enable_binding=False)
        adapter = CameraAdapter(sim)
        video = adapter.generate_ground_truth_rgb_video(
            output_path=str(tmp_path / "gt.tif"), pixel_size_nm=20.0, save_video=False,
        )
        assert np.all(video == 10)  # background only -- molecule never drawn

    def test_unknown_colormap_raises(self, tmp_path):
        sim = self._sim_with_molecules()
        adapter = CameraAdapter(sim)
        with pytest.raises(ValueError, match="Unknown colormap"):
            adapter.generate_ground_truth_rgb_video(
                output_path=str(tmp_path / "gt.tif"), colormap="bogus", save_video=False,
            )

    def test_explicit_frame_indices_and_image_size(self, tmp_path):
        sim = self._sim_with_molecules()
        adapter = CameraAdapter(sim)
        video = adapter.generate_ground_truth_rgb_video(
            output_path=str(tmp_path / "gt.tif"),
            frame_indices=np.array([0, 1]),
            image_size_nm=(150.0, 150.0),
            pixel_size_nm=20.0, save_video=False,
        )
        assert video.shape[0] == 2

    def test_molecule_out_of_trajectory_range_and_out_of_fov_skipped(self, tmp_path):
        sim = DiffusionSimulator2D(area=(50.0, 50.0), dt=1.0, t_exposure=1.0, sigma0=5.0, s0=100.0)
        sim.add_molecule("R", np.array([25.0, 25.0]), D_free=0.0)
        sim.run(n_steps=1, enable_binding=False)
        # A second molecule with a shorter trajectory (added after run, never
        # advanced) exercises the `traj_idx >= len(mol.trajectory)` skip when
        # frame_indices asks for frame 1.
        sim.add_molecule("B", np.array([40.0, 40.0]), D_free=0.0)
        adapter = CameraAdapter(sim)
        video = adapter.generate_ground_truth_rgb_video(
            output_path=str(tmp_path / "gt.tif"),
            frame_indices=np.array([0, 1]),
            pixel_size_nm=20.0, save_video=False,
        )
        assert video.shape[0] == 2

    def test_scale_intensity_absolute_max(self, tmp_path):
        sim = self._sim_with_molecules()
        adapter = CameraAdapter(sim)
        video = adapter.generate_ground_truth_rgb_video(
            output_path=str(tmp_path / "gt.tif"), pixel_size_nm=20.0,
            save_video=False, intensity_percentile=100.0,
        )
        assert video.dtype == np.uint8

    def test_scale_intensity_false(self, tmp_path):
        sim = self._sim_with_molecules()
        adapter = CameraAdapter(sim)
        video = adapter.generate_ground_truth_rgb_video(
            output_path=str(tmp_path / "gt.tif"), pixel_size_nm=20.0,
            save_video=False, scale_intensity=False,
        )
        assert video.dtype == np.uint8

    def test_all_zero_intensity_no_scale_factor_applied(self, tmp_path):
        # No molecules in FOV at all -> rgb_video_float stays all-zero ->
        # scale_target == 0 -> the `if scale_target > 0` branch is skipped.
        # boundary="absorbing" is required: the default "reflective" boundary
        # clamps add_molecule's initial position into the area, so a huge
        # coordinate wouldn't actually land outside the FOV.
        sim = DiffusionSimulator2D(area=(50.0, 50.0), dt=1.0, t_exposure=1.0, sigma0=5.0, s0=100.0,
                                    boundary="absorbing")
        sim.add_molecule("R", np.array([1e9, 1e9]), D_free=0.0)
        sim.run(n_steps=1, enable_binding=False)
        adapter = CameraAdapter(sim)
        video = adapter.generate_ground_truth_rgb_video(
            output_path=str(tmp_path / "gt.tif"), pixel_size_nm=20.0, save_video=False,
        )
        assert np.all(video == 10)  # just the background_value

    def test_save_video_tifffile_path(self, tmp_path):
        sim = self._sim_with_molecules()
        adapter = CameraAdapter(sim)
        out = tmp_path / "gt.tif"
        adapter.generate_ground_truth_rgb_video(
            output_path=str(out), pixel_size_nm=20.0, save_video=True,
        )
        assert out.exists()

    def test_save_video_tifffile_import_error_fallback(self, tmp_path, monkeypatch):
        sim = self._sim_with_molecules()
        adapter = CameraAdapter(sim)
        monkeypatch.setitem(sys.modules, "tifffile", None)
        out = tmp_path / "gt.tif"
        adapter.generate_ground_truth_rgb_video(
            output_path=str(out), pixel_size_nm=20.0, save_video=True,
        )
        assert out.exists()

    def test_save_video_tifffile_fallback_inserts_src_path(self, tmp_path, monkeypatch):
        # src/ is already on sys.path from module import time, so the
        # `if _dir not in sys.path: sys.path.insert(...)` guard is normally
        # a no-op -- remove it first so the insert branch actually runs.
        from pathlib import Path as _Path
        _dir = str(_Path(diff.__file__).parent.parent)
        sim = self._sim_with_molecules()
        adapter = CameraAdapter(sim)
        monkeypatch.setitem(sys.modules, "tifffile", None)
        monkeypatch.setattr(sys, "path", [p for p in sys.path if p != _dir])
        out = tmp_path / "gt.tif"
        adapter.generate_ground_truth_rgb_video(
            output_path=str(out), pixel_size_nm=20.0, save_video=True,
        )
        assert out.exists()
        assert _dir in sys.path
