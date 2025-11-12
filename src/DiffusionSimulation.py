#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multicolor Diffusion-Binding Simulation

Simulates molecules of different colors diffusing on a 2D surface with
binding/unbinding kinetics. Designed for testing pyBayerSMLM analysis pipeline.

Based on:
- Michalet, X.; Berglund, A. J. Phys. Rev. E 2012, 85 (6), 061916.
  (Realistic Brownian motion with camera effects)
- Gillespie, D. T. J. Phys. Chem. 1977, 81 (25), 2340-2361.
  (Stochastic simulation algorithm)

@author: jbeckwith
Created: 2025-11-12
"""

import numpy as np
from scipy.sparse import diags_array
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from numba import jit


@dataclass
class Molecule:
    """
    Single molecule with color and diffusion properties.

    Attributes:
        molecule_id: Unique identifier for this molecule
        color: Color label ('R', 'G', or 'B' for red/green/blue)
        position: Current (x, y) coordinates in nm
        D_free: Diffusion coefficient when unbound (nm²/ms)
        D_bound: Diffusion coefficient when bound (nm²/ms)
        is_bound: Current binding state
        bound_partner: Reference to binding partner molecule
        trajectory: History of positions [(x, y), ...]
        times: History of timepoints [t0, t1, ...]
        spectral_profile: Dictionary with A_R, A_G, A_B values
    """
    molecule_id: int
    color: str
    position: np.ndarray
    D_free: float
    D_bound: float = 0.0
    is_bound: bool = False
    bound_partner: Optional['Molecule'] = None
    trajectory: List[np.ndarray] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    spectral_profile: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize spectral profiles based on color."""
        if not self.spectral_profile:
            # Default spectral profiles for R, G, B dyes
            profiles = {
                'R': {'A_R': 0.80, 'A_G': 0.15, 'A_B': 0.05},
                'G': {'A_R': 0.10, 'A_G': 0.80, 'A_B': 0.10},
                'B': {'A_R': 0.05, 'A_G': 0.15, 'A_B': 0.80},
            }
            self.spectral_profile = profiles.get(self.color,
                                                  {'A_R': 0.33, 'A_G': 0.33, 'A_B': 0.34})

        # Initialize trajectory with starting position
        if not self.trajectory:
            self.trajectory = [self.position.copy()]
            self.times = [0.0]

    def get_current_D(self) -> float:
        """Get current diffusion coefficient based on binding state."""
        return self.D_bound if self.is_bound else self.D_free

    def add_position(self, position: np.ndarray, time: float):
        """Add a position to the trajectory."""
        self.trajectory.append(position.copy())
        self.times.append(time)
        self.position = position.copy()


class LangevinDiffusion2D:
    """
    2D Brownian diffusion with realistic camera imaging effects.

    Implements the model from Michalet & Berglund (2012):
    - Motion blur from finite exposure time
    - Dynamic localization error
    - Covariance structure for realistic trajectories
    """

    def __init__(self, sigma0: float, s0: float, R: float = 1.0/6):
        """
        Initialize diffusion simulator.

        Args:
            sigma0: Static localization error (nm)
            s0: Standard deviation of PSF (nm)
            R: Motion blur coefficient (default 1/6 for typical camera)
        """
        self.sigma0 = sigma0
        self.s0 = s0
        self.R = R

    def compute_dynamic_localization_error(self, D: float, t_exposure: float) -> float:
        """
        Compute dynamic localization error including motion blur.

        From Michalet & Berglund Eq. 3:
        σ = σ₀√(1 + D·tₑ/s₀²)

        Args:
            D: Diffusion coefficient (nm²/ms)
            t_exposure: Camera exposure time (ms)

        Returns:
            sigma: Dynamic localization error (nm)
        """
        return self.sigma0 * np.sqrt(1.0 + (D * t_exposure) / (self.s0**2))

    def generate_displacement_1D(self, D: float, N: int, dt: float,
                                  t_exposure: float) -> np.ndarray:
        """
        Generate realistic 1D Brownian displacements.

        Uses covariance matrix from Michalet & Berglund Eqs. 2-5:
        - Diagonal: 2D·Δt·(1-2R) + 2σ²
        - Off-diagonal: 2R·D·Δt - σ²

        Args:
            D: Diffusion coefficient (nm²/ms)
            N: Number of steps
            dt: Time step between data points (ms)
            t_exposure: Camera exposure duration (ms)

        Returns:
            positions: Array of N positions (starting from 0)
        """
        if N < 2:
            return np.array([0.0])

        # Compute dynamic localization error
        sigma = self.compute_dynamic_localization_error(D, t_exposure)

        # Build covariance matrix (tridiagonal)
        diagonal = 2 * D * dt * (1 - 2 * self.R) + 2 * sigma**2
        off_diagonal = 2 * self.R * D * dt - sigma**2

        # Create sparse tridiagonal matrix
        stack = np.vstack([
            np.full(N - 1, off_diagonal),
            np.full(N - 1, diagonal),
            np.full(N - 1, off_diagonal),
        ])

        cov = diags_array(stack, offsets=[-1, 0, 1],
                          shape=(N - 1, N - 1)).toarray()

        # Generate correlated random displacements
        rng = np.random.default_rng()
        displacements = rng.multivariate_normal(np.zeros(N - 1), cov)

        # Return cumulative positions (starting from 0)
        positions = np.zeros(N)
        positions[1:] = np.cumsum(displacements)

        return positions

    def generate_trajectory_2D(self, D: float, N: int, dt: float,
                               t_exposure: float,
                               starting_position: np.ndarray) -> np.ndarray:
        """
        Generate realistic 2D Brownian trajectory.

        Args:
            D: Diffusion coefficient (nm²/ms)
            N: Number of steps
            dt: Time step (ms)
            t_exposure: Camera exposure time (ms)
            starting_position: Initial (x, y) position (nm)

        Returns:
            trajectory: Array of shape (N, 2) with (x, y) positions
        """
        # Generate independent x and y displacements
        x_disp = self.generate_displacement_1D(D, N, dt, t_exposure)
        y_disp = self.generate_displacement_1D(D, N, dt, t_exposure)

        # Combine and add starting position
        trajectory = np.column_stack([x_disp, y_disp])
        trajectory += starting_position

        return trajectory


class DiffusionSimulator2D:
    """
    Main simulator for 2D diffusion of multiple molecules.

    Features:
    - Multiple molecules with different diffusion coefficients
    - Boundary conditions (periodic, reflective, absorbing)
    - Realistic camera imaging effects
    - Compatible with pyBayerSMLM analysis
    """

    def __init__(self, area: Tuple[float, float], dt: float, t_exposure: float,
                 sigma0: float, s0: float, R: float = 1.0/6,
                 boundary: str = 'reflective'):
        """
        Initialize 2D diffusion simulator.

        Args:
            area: (width, height) of simulation area in nm
            dt: Timestep for simulation (ms)
            t_exposure: Camera exposure time (ms)
            sigma0: Static localization error (nm)
            s0: PSF standard deviation (nm)
            R: Motion blur coefficient (default 1/6)
            boundary: Boundary condition ('periodic', 'reflective', 'absorbing')
        """
        self.area = np.array(area)
        self.dt = dt
        self.t_exposure = t_exposure
        self.boundary = boundary
        self.current_time = 0.0

        # Diffusion engine
        self.diffusion_engine = LangevinDiffusion2D(sigma0, s0, R)

        # Molecule storage
        self.molecules: List[Molecule] = []
        self.molecule_counter = 0

    def add_molecule(self, color: str, position: np.ndarray, D_free: float,
                     D_bound: float = 0.0,
                     spectral_profile: Optional[Dict[str, float]] = None) -> Molecule:
        """
        Add a molecule to the simulation.

        Args:
            color: Color label ('R', 'G', or 'B')
            position: Initial (x, y) position in nm
            D_free: Diffusion coefficient when unbound (nm²/ms)
            D_bound: Diffusion coefficient when bound (nm²/ms)
            spectral_profile: Optional custom spectral profile

        Returns:
            molecule: The created Molecule object
        """
        # Ensure position is within bounds
        position = self._apply_boundary_condition(position)

        mol = Molecule(
            molecule_id=self.molecule_counter,
            color=color,
            position=position,
            D_free=D_free,
            D_bound=D_bound,
            spectral_profile=spectral_profile or {}
        )

        self.molecules.append(mol)
        self.molecule_counter += 1

        return mol

    def add_molecules_random(self, n_molecules: int, color: str,
                            D_free: float, D_bound: float = 0.0) -> List[Molecule]:
        """
        Add multiple molecules with random starting positions.

        Args:
            n_molecules: Number of molecules to add
            color: Color label
            D_free: Diffusion coefficient when unbound
            D_bound: Diffusion coefficient when bound

        Returns:
            molecules: List of created molecules
        """
        molecules = []
        for _ in range(n_molecules):
            x = np.random.uniform(0, self.area[0])
            y = np.random.uniform(0, self.area[1])
            position = np.array([x, y])

            mol = self.add_molecule(color, position, D_free, D_bound)
            molecules.append(mol)

        return molecules

    def _apply_boundary_condition(self, position: np.ndarray) -> np.ndarray:
        """
        Apply boundary conditions to a position.

        Args:
            position: (x, y) position

        Returns:
            corrected_position: Position after applying boundary
        """
        pos = position.copy()

        if self.boundary == 'periodic':
            # Wrap around boundaries
            pos = np.mod(pos, self.area)

        elif self.boundary == 'reflective':
            # Reflect at boundaries
            for i in range(2):
                if pos[i] < 0:
                    pos[i] = -pos[i]
                elif pos[i] > self.area[i]:
                    pos[i] = 2 * self.area[i] - pos[i]

            # Clamp to ensure within bounds (in case of multiple reflections)
            pos = np.clip(pos, 0, self.area)

        elif self.boundary == 'absorbing':
            # Mark molecules that leave as invalid (handled elsewhere)
            pass

        return pos

    def run(self, n_steps: int):
        """
        Run simulation for n_steps.

        Generates full trajectories for all molecules at once to preserve
        the proper covariance structure from Michalet & Berglund (2012).

        Args:
            n_steps: Number of timesteps to simulate
        """
        for mol in self.molecules:
            # Get current D (depends on binding state)
            D = mol.get_current_D()

            # Generate full trajectory at once (N steps from current position)
            # This preserves the covariance structure
            trajectory = self.diffusion_engine.generate_trajectory_2D(
                D=D,
                N=n_steps,
                dt=self.dt,
                t_exposure=self.t_exposure,
                starting_position=mol.position
            )

            # Apply boundary conditions to each position
            for i in range(n_steps):
                new_position = self._apply_boundary_condition(trajectory[i])
                time = mol.times[-1] + (i + 1) * self.dt
                mol.add_position(new_position, time)

        # Update current time
        self.current_time += n_steps * self.dt

    def get_trajectory(self, molecule_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get trajectory for a specific molecule.

        Args:
            molecule_id: ID of molecule

        Returns:
            positions: Array of shape (N, 2) with (x, y) positions
            times: Array of timepoints
        """
        mol = self.molecules[molecule_id]
        positions = np.array(mol.trajectory)
        times = np.array(mol.times)

        return positions, times

    def get_all_trajectories(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Get all molecule trajectories.

        Returns:
            trajectories: Dictionary mapping molecule_id -> (positions, times)
        """
        trajectories = {}
        for mol in self.molecules:
            positions = np.array(mol.trajectory)
            times = np.array(mol.times)
            trajectories[mol.molecule_id] = (positions, times)

        return trajectories

    def reset(self):
        """Reset simulation to initial state."""
        self.current_time = 0.0

        for mol in self.molecules:
            # Keep only first position
            mol.trajectory = [mol.trajectory[0].copy()]
            mol.times = [0.0]
            mol.position = mol.trajectory[0].copy()
            mol.is_bound = False
            mol.bound_partner = None


def compute_msd_from_trajectory(trajectory: np.ndarray,
                                 max_tau: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean squared displacement from a trajectory.

    MSD(τ) = ⟨[r(t+τ) - r(t)]²⟩

    Args:
        trajectory: Array of shape (N, 2) with (x, y) positions
        max_tau: Maximum lag time to compute (default: N//4)

    Returns:
        tau_array: Array of lag times (in units of timesteps)
        msd_array: Array of MSD values
    """
    N = len(trajectory)

    if max_tau is None:
        max_tau = N // 4  # Use first quarter to ensure good statistics

    max_tau = min(max_tau, N - 1)

    msd = np.zeros(max_tau)

    for tau in range(1, max_tau + 1):
        displacements = trajectory[tau:] - trajectory[:-tau]
        squared_displacements = np.sum(displacements**2, axis=1)
        msd[tau - 1] = np.mean(squared_displacements)

    tau_array = np.arange(1, max_tau + 1)

    return tau_array, msd


def estimate_D_from_msd(tau_array: np.ndarray, msd_array: np.ndarray,
                        dt: float, n_d: int = 2, fit_points: int = 10) -> float:
    """
    Estimate diffusion coefficient from MSD using linear fit.

    For 2D: MSD(τ) = 4D·τ + offset

    Args:
        tau_array: Array of lag times (in timesteps)
        msd_array: Array of MSD values
        dt: Timestep duration (ms)
        n_d: Number of dimensions (default 2)
        fit_points: Number of initial points to use for linear fit

    Returns:
        D: Estimated diffusion coefficient (nm²/ms)
    """
    # Use first few points for linear fit
    fit_points = min(fit_points, len(tau_array))

    # Convert tau to time
    t = tau_array[:fit_points] * dt
    msd = msd_array[:fit_points]

    # Linear fit: MSD = 2·n_d·D·t + offset
    # Slope = 2·n_d·D
    coeffs = np.polyfit(t, msd, deg=1)
    slope = coeffs[0]

    D = slope / (2 * n_d)

    return D


# =============================================================================
# OLSF MSD Analysis (from pyDiffusion_LeeLab)
# =============================================================================

def autocorrFFT(x: np.ndarray) -> np.ndarray:
    """
    Compute the autocorrelation of a 1D signal using FFT.

    Args:
        x: Input signal

    Returns:
        res: Autocorrelation of the input signal
    """
    N = len(x)
    F = np.fft.fft(x, n=2 * N)  # 2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real  # now we have the autocorrelation in convention B
    n = N * np.ones(N) - np.arange(0, N)  # divide res(m) by (N-m)
    return res / n  # this is the autocorrelation in convention A


def msd_fft(r: np.ndarray) -> np.ndarray:
    """
    Compute the mean squared displacement (MSD) using FFT.

    Args:
        r: Trajectory data, shape (N, n_d) with positions at each timepoint

    Returns:
        S: MSD computed for each time step
    """
    N = len(r)
    D = np.square(r).sum(axis=1)
    D = np.append(D, 0)
    S2 = sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    S = S1 - 2 * S2
    return S[1:]


@jit(nopython=True)
def PMin_XM(x: float, N: int) -> Tuple[int, int]:
    """
    Calculate optimal fit point from Michalet (2010).

    Args:
        x: Input value
        N: Number of trajectory points

    Returns:
        pa: Optimal fit point for parameter 'a'
        pb: Optimal fit point for parameter 'b'
    """
    fa = 2 + 1.6 * (x**0.51)
    La = 3 + ((4.5 * (N**0.4) - 8.5) ** 1.2)

    fb = 2 + 1.35 * x**0.6
    Lb = 0.8 + (0.564 * N)

    if np.isinf(x):
        pa = int(np.floor(La))
        pb = int(np.floor(Lb))
    else:
        pa = int(np.floor(fa * La / (fa**3 + La**3) ** 0.33))
        pb = min(
            int(np.floor(Lb)), int(np.floor(fb * Lb / (fb**3 + Lb**3) ** 0.33))
        )

    # Make sure nothing is zero
    pa = max(2, pa)
    pb = max(2, pb)

    return pa, pb


def estimate_D_OLSF(trajectory: np.ndarray, dt: float, R: float = 1.0/6,
                    n_d: int = 2, maxiter: int = 100,
                    min_points: int = 10) -> Tuple[float, float]:
    """
    Estimate diffusion coefficient using Optimal Least Squares Fit (OLSF).

    This is the optimal method from Michalet & Berglund (2012) that properly
    accounts for localization error and finite trajectory length.

    Args:
        trajectory: Array of shape (N, n_d) with positions
        dt: Timestep (ms)
        R: Motion blur coefficient (default 1/6)
        n_d: Number of dimensions (default 2)
        maxiter: Maximum iterations (default 100)
        min_points: Minimum points required (default 10)

    Returns:
        D: Diffusion coefficient estimate (nm²/ms)
        var: Localization variance estimate (nm²)
    """
    coordinates = trajectory

    if n_d > 1:
        if coordinates.shape[1] != n_d:
            print(f"Error: Dimension mismatch. Expected {n_d}, got {coordinates.shape[1]}")
            return np.nan, np.nan

    if coordinates.shape[0] < min_points:
        print(f"Error: Not enough points ({coordinates.shape[0]} < {min_points})")
        return np.nan, np.nan

    NXM = len(coordinates.ravel())
    pa = np.array([int(np.floor(NXM / 10))])
    pb = np.array([int(np.floor(NXM / 10))])
    donea = False
    doneb = False
    iter_count = 1
    rho = []

    D = np.nan
    var = np.nan

    while iter_count <= maxiter and (not donea or not doneb):
        if iter_count == maxiter:
            donea = True
            doneb = True
            D = np.nan
            var = np.nan
            print(f"OLSF did not converge in {maxiter} iterations")
            break

        if np.max(np.hstack([pa, pb])) > len(rho):
            rho = msd_fft(coordinates)
        rho_subset = rho[: np.max(np.hstack([pa, pb]))]

        if not donea:
            A = np.vstack((np.ones(pa[-1]), np.arange(1, pa[-1] + 1))).T
            B = np.asarray(rho_subset[: A.shape[0]])
            if B.shape[0] != A.shape[0]:
                donea = True
                doneb = True
                D = np.nan
                var = np.nan
                print(f"OLSF did not converge in {maxiter} iterations")
                break

            X = np.linalg.lstsq(A, B, rcond=None)[0]
            aa, ba = X
            xa = 0 if aa < 0 else np.inf if ba < 0 else aa / ba
            newpa, _ = PMin_XM(xa, NXM)
            if np.any(np.isin(newpa, pa)):
                Da = ba / (2 * dt * n_d)
                var = (aa + 4 * Da * R * dt) / (2 * n_d)
                donea = True
            pa = np.hstack([pa, newpa])

        if not doneb:
            A = np.vstack((np.ones(pb[-1]), np.arange(1, pb[-1] + 1))).T
            B = np.asarray(rho_subset[: int(pb[-1])])
            if B.shape[0] != A.shape[0]:
                donea = True
                doneb = True
                D = np.nan
                var = np.nan
                print(f"OLSF did not converge in {maxiter} iterations")
                break

            X = np.linalg.lstsq(A, B, rcond=None)[0]
            ab, bb = X
            xb = 0 if ab < 0 else np.inf if bb < 0 else ab / bb
            _, newpb = PMin_XM(xb, NXM)
            if np.any(np.isin(newpb, pb)):
                D = bb / (2 * dt * n_d)
                var = (ab + 4 * D * R * dt) / (2 * n_d)
                doneb = True
            pb = np.hstack([pb, newpb])

        iter_count += 1

    return D, var
