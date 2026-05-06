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
from scipy.spatial.distance import cdist
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass, field
from numba import jit
from Constants import DriftConstants


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


class BindingKinetics:
    """
    Handles binding/unbinding kinetics between molecules using Gillespie algorithm.

    Two modes:
    1. LEGACY: Uses macroscopic k_on/k_off matrices (concentration-dependent)
    2. MICROSCOPIC: Uses microscopic k, γ, ρ parameters (Fange et al. 2010)
       - Automatically calculates scale-dependent mesoscopic rates q_a(h), q_d(h)

    The Gillespie algorithm ensures correct stochastic kinetics.
    """

    def __init__(self, colors: List[str], k_on_matrix: np.ndarray = None,
                 k_off_matrix: np.ndarray = None, binding_radius: float = 100.0,
                 reaction_radius: float = None, k_micro_matrix: np.ndarray = None,
                 gamma_matrix: np.ndarray = None, use_microscopic: bool = False):
        """
        Initialize binding kinetics.

        LEGACY MODE (use_microscopic=False):
            k_on_matrix: 2D array of binding rates (1/(nM·s)) between color pairs
            k_off_matrix: 2D array of unbinding rates (1/s) for each pair
            binding_radius: Distance threshold for binding (nm)

        MICROSCOPIC MODE (use_microscopic=True, recommended):
            reaction_radius: ρ (nm) - contact distance for binding
            k_micro_matrix: 2D array of intrinsic on-rates k (1/s) at contact
            gamma_matrix: 2D array of intrinsic off-rates γ (1/s)
            binding_radius: Distance threshold for finding pairs (nm)
                           Should be 3-5× reaction_radius

        Args:
            colors: List of color labels (e.g., ['R', 'G', 'B'])
            use_microscopic: If True, use Fange et al. 2010 framework
        """
        self.colors = colors
        self.color_to_idx = {c: i for i, c in enumerate(colors)}
        self.binding_radius = binding_radius
        self.use_microscopic = use_microscopic

        n_colors = len(colors)

        if use_microscopic:
            # Microscopic framework (Fange et al. 2010)
            if reaction_radius is None or k_micro_matrix is None or gamma_matrix is None:
                raise ValueError("Microscopic mode requires: reaction_radius, k_micro_matrix, gamma_matrix")

            self.reaction_radius = reaction_radius  # ρ (nm)
            self.k_micro_matrix = np.array(k_micro_matrix)  # k (1/s)
            self.gamma_matrix = np.array(gamma_matrix)      # γ (1/s)

            # Validate
            assert self.k_micro_matrix.shape == (n_colors, n_colors), \
                f"k_micro_matrix shape {self.k_micro_matrix.shape} != ({n_colors}, {n_colors})"
            assert self.gamma_matrix.shape == (n_colors, n_colors), \
                f"gamma_matrix shape {self.gamma_matrix.shape} != ({n_colors}, {n_colors})"

            # Make symmetric
            self.k_micro_matrix = (self.k_micro_matrix + self.k_micro_matrix.T) / 2
            self.gamma_matrix = (self.gamma_matrix + self.gamma_matrix.T) / 2

            # Legacy matrices will be calculated on-demand via calculate_mesoscopic_rates()
            self.k_on_matrix = None
            self.k_off_matrix = None

        else:
            # Legacy macroscopic framework
            if k_on_matrix is None or k_off_matrix is None:
                raise ValueError("Legacy mode requires: k_on_matrix, k_off_matrix")

            self.k_on_matrix = np.array(k_on_matrix)
            self.k_off_matrix = np.array(k_off_matrix)
            self.reaction_radius = None
            self.k_micro_matrix = None
            self.gamma_matrix = None

            # Validate
            assert self.k_on_matrix.shape == (n_colors, n_colors), \
                f"k_on_matrix shape {self.k_on_matrix.shape} != ({n_colors}, {n_colors})"
            assert self.k_off_matrix.shape == (n_colors, n_colors), \
                f"k_off_matrix shape {self.k_off_matrix.shape} != ({n_colors}, {n_colors})"

            # Make symmetric
            self.k_on_matrix = (self.k_on_matrix + self.k_on_matrix.T) / 2
            self.k_off_matrix = (self.k_off_matrix + self.k_off_matrix.T) / 2

        # Track binding events for analysis
        self.binding_events: List[Dict] = []
        self.unbinding_events: List[Dict] = []

    def calculate_mesoscopic_rates(self, lattice_spacing: float,
                                   diffusion_coeff: float,
                                   dimensionality: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate scale-dependent mesoscopic rates q_a(h) and q_d(h).

        Based on Fange et al. (2010) PNAS 107(46):19820-19825, Eqs. 1-2 and 7-8.
        This fixes the RDME divergence problem at fine spatial discretization.

        The mesoscopic rates account for:
        - Spatial discretization effects (lattice spacing h)
        - Diffusion-controlled vs reaction-controlled kinetics (parameter α)
        - Reactions between neighboring subvolumes (Cartesian grids)
        - Microscopic rebinding after dissociation

        Args:
            lattice_spacing: h (nm) - size of spatial discretization (voxel side length)
            diffusion_coeff: D (nm²/ms) - combined diffusion D_A + D_B
            dimensionality: 2 or 3 for 2D/3D systems

        Returns:
            q_a_matrix: Mesoscopic association rates
                       Units: 1/ms (propensity per pair in contact)
            q_d_matrix: Mesoscopic dissociation rates
                       Units: 1/ms

        Notes:
            - For 3D: Rates converge to macroscopic values at h >> ρ
            - For 2D: Rates are scale-dependent at ALL discretizations
            - Satisfies detailed balance: q_a/q_d = K (equilibrium constant)
            - Include neighbor reactions: h_eff calculated from 7ℓ³ (3D) or 5ℓ² (2D)

        References:
            Fange et al. (2010) doi:10.1073/pnas.1006565107
        """
        if not self.use_microscopic:
            raise RuntimeError("calculate_mesoscopic_rates() requires use_microscopic=True mode")

        if dimensionality not in [2, 3]:
            raise ValueError("dimensionality must be 2 or 3")

        n_colors = len(self.colors)
        q_a_matrix = np.zeros((n_colors, n_colors))
        q_d_matrix = np.zeros((n_colors, n_colors))

        # Calculate effective reaction radius from lattice spacing + neighbors
        # This accounts for reactions between molecules in neighboring voxels
        if dimensionality == 3:
            # Central voxel + 6 neighbors: 4π(h+ρ)³/3 = 7ℓ³
            # Solve for h: (h+ρ)³ = 7ℓ³ × 3/(4π)
            h_eff = ((7 * lattice_spacing**3 * 3 / (4*np.pi)))**(1/3) - self.reaction_radius
        else:  # 2D
            # Central voxel + 4 neighbors: π(h+ρ)² = 5ℓ²
            # Solve for h: (h+ρ)² = 5ℓ²/π
            h_eff = np.sqrt(5 * lattice_spacing**2 / np.pi) - self.reaction_radius

        # For very fine discretization, h_eff can be negative
        # This is OK - it means β → 1 and rates → microscopic k, γ
        # Only warn if lattice is MUCH smaller than expected
        if lattice_spacing < 0.1 * self.reaction_radius:
            import warnings
            warnings.warn(f"Lattice spacing {lattice_spacing:.1f} nm is very fine "
                         f"compared to reaction radius {self.reaction_radius:.1f} nm. "
                         f"Results may be inaccurate.")

        # Clip h_eff to avoid numerical issues, but allow negative values
        # (negative h_eff just means very fine grid, β close to 1)
        h_eff = max(h_eff, -0.9 * self.reaction_radius)

        # Discretization parameter β = ρ/(ρ+h)
        # β → 1: fine discretization (h → 0), rates → microscopic k, γ
        # β → 0: coarse discretization (h → ∞), rates → macroscopic k_a, k_d
        beta = self.reaction_radius / (self.reaction_radius + h_eff)

        # Ensure β is in valid range [0, 1]
        # If h_eff is very negative, β can exceed 1, which is non-physical
        # Clip to ensure well-defined behavior
        beta = np.clip(beta, 0.0, 0.999)  # Don't allow exactly 1 to avoid division issues

        for i in range(n_colors):
            for j in range(n_colors):
                k_micro = self.k_micro_matrix[i, j]  # Intrinsic on-rate (1/s)
                gamma = self.gamma_matrix[i, j]       # Intrinsic off-rate (1/s)

                if k_micro == 0:
                    # No binding allowed
                    continue

                # Degree of diffusion control α
                # α → 0: reaction-limited (slow k, fast diffusion)
                # α → ∞: diffusion-limited (fast k, slow diffusion)
                # Note: k_micro is in 1/s, diffusion_coeff is in nm²/ms
                # Convert D to nm²/s for consistent units
                D_nm2_per_s = diffusion_coeff * 1000.0
                if dimensionality == 3:
                    alpha = k_micro / (4 * np.pi * self.reaction_radius * D_nm2_per_s)
                else:  # 2D
                    alpha = k_micro / (2 * np.pi * D_nm2_per_s)

                # Calculate mesoscopic association rate q_a(h)
                if dimensionality == 3:
                    # Fange Eq. 1 (approximate form, excellent agreement)
                    # Exact form is Eq. 7 (more complex, see SI)
                    q_a = k_micro / (1 + alpha * (1-beta) * (1 - 0.58*beta))
                else:  # 2D
                    # Fange Eq. 2
                    # In 2D, no macroscopic limit exists (always scale-dependent)
                    if beta < 0.999:  # Avoid log(0)
                        q_a = k_micro / (1 + alpha * np.log(1 + 0.544*(1-beta)/beta))
                    else:
                        q_a = k_micro  # Very fine discretization

                # Mesoscopic dissociation rate q_d(h)
                # Detailed balance requires q_a/q_d = k/γ = K
                K = k_micro / gamma  # Equilibrium constant
                q_d = q_a / K

                # Convert from 1/s to 1/ms
                q_a_matrix[i, j] = q_a / 1000.0
                q_d_matrix[i, j] = q_d / 1000.0

        return q_a_matrix, q_d_matrix

    def can_bind(self, mol1: Molecule, mol2: Molecule) -> bool:
        """
        Check if two molecules can bind.

        Args:
            mol1: First molecule
            mol2: Second molecule

        Returns:
            can_bind: True if binding is allowed
        """
        # Can't bind if either is already bound
        if mol1.is_bound or mol2.is_bound:
            return False

        # Check if binding rate is non-zero
        idx1 = self.color_to_idx[mol1.color]
        idx2 = self.color_to_idx[mol2.color]

        if self.use_microscopic:
            return self.k_micro_matrix[idx1, idx2] > 0
        else:
            return self.k_on_matrix[idx1, idx2] > 0

    def get_binding_rate(self, mol1: Molecule, mol2: Molecule) -> float:
        """
        Get binding rate constant for two molecules.

        Args:
            mol1: First molecule
            mol2: Second molecule

        Returns:
            k_on: Binding rate (1/(nM·s))
        """
        idx1 = self.color_to_idx[mol1.color]
        idx2 = self.color_to_idx[mol2.color]
        return self.k_on_matrix[idx1, idx2]

    def get_unbinding_rate(self, mol1: Molecule, mol2: Molecule) -> float:
        """
        Get unbinding rate constant for a bound pair.

        Args:
            mol1: First molecule
            mol2: Second molecule

        Returns:
            k_off: Unbinding rate (1/s)
        """
        idx1 = self.color_to_idx[mol1.color]
        idx2 = self.color_to_idx[mol2.color]
        return self.k_off_matrix[idx1, idx2]

    def find_potential_binding_pairs(self, molecules: List[Molecule]) -> List[Tuple[Molecule, Molecule, float]]:
        """
        Find all molecule pairs within binding radius.

        Args:
            molecules: List of all molecules

        Returns:
            pairs: List of (mol1, mol2, distance) tuples within binding radius
        """
        # Get positions of unbound molecules
        unbound_mols = [m for m in molecules if not m.is_bound]

        if len(unbound_mols) < 2:
            return []

        # Compute pairwise distances
        positions = np.array([m.position for m in unbound_mols])
        distances = cdist(positions, positions)

        pairs = []
        for i in range(len(unbound_mols)):
            for j in range(i + 1, len(unbound_mols)):
                mol1 = unbound_mols[i]
                mol2 = unbound_mols[j]
                dist = distances[i, j]

                if dist <= self.binding_radius and self.can_bind(mol1, mol2):
                    pairs.append((mol1, mol2, dist))

        return pairs

    def calculate_propensities(self, molecules: List[Molecule], area: float,
                              lattice_spacing: float = None,
                              diffusion_coeff: float = None) -> Tuple[List, List, float]:
        """
        Calculate all binding and unbinding propensities.

        LEGACY MODE:
            Propensity = k_on × effective_concentration × volume
            (concentration-dependent, volume-based calculation)

        MICROSCOPIC MODE:
            Propensity = q_a(h) or q_d(h)
            (intrinsic rates, no volume dependence)
            Requires lattice_spacing and diffusion_coeff parameters

        Args:
            molecules: List of all molecules
            area: Simulation area (nm²) - only used in legacy mode
            lattice_spacing: h (nm) - spatial discretization (microscopic mode only)
            diffusion_coeff: D (nm²/ms) - combined diffusion (microscopic mode only)

        Returns:
            binding_propensities: List of (mol1, mol2, propensity) for binding events
            unbinding_propensities: List of (mol1, mol2, propensity) for unbinding events
            total_propensity: Sum of all propensities
        """
        binding_propensities = []
        unbinding_propensities = []

        # Find potential binding pairs
        pairs = self.find_potential_binding_pairs(molecules)

        if self.use_microscopic:
            # MICROSCOPIC MODE: Use mesoscopic rates q_a(h), q_d(h)
            if lattice_spacing is None or diffusion_coeff is None:
                raise ValueError("Microscopic mode requires lattice_spacing and diffusion_coeff")

            # Calculate mesoscopic rates for current discretization
            q_a_matrix, q_d_matrix = self.calculate_mesoscopic_rates(
                lattice_spacing, diffusion_coeff, dimensionality=3
            )

            # Calculate binding propensities
            for mol1, mol2, dist in pairs:
                idx1 = self.color_to_idx[mol1.color]
                idx2 = self.color_to_idx[mol2.color]
                q_a = q_a_matrix[idx1, idx2]  # Already in 1/ms

                if q_a > 0:
                    # Propensity is just the mesoscopic rate
                    # No volume normalization! Molecules are already in contact
                    propensity = q_a  # 1/ms
                    binding_propensities.append((mol1, mol2, propensity))

            # Calculate unbinding propensities
            bound_pairs = self._get_bound_pairs(molecules)
            for mol1, mol2 in bound_pairs:
                idx1 = self.color_to_idx[mol1.color]
                idx2 = self.color_to_idx[mol2.color]
                q_d = q_d_matrix[idx1, idx2]  # Already in 1/ms
                unbinding_propensities.append((mol1, mol2, q_d))

        else:
            # LEGACY MODE: Use macroscopic k_on, k_off with volume-based calculation
            # Calculate binding propensities
            # For surface diffusion: effective concentration ~ 1/area
            # Propensity = k_on × (effective concentration)
            effective_volume = area * 1.0  # nm² × 1 nm height = nm³
            nM_to_nm3 = 1e-9 / 6.022e23 * 1e24  # Convert nM to molecules/nm³

            for mol1, mol2, dist in pairs:
                k_on = self.get_binding_rate(mol1, mol2)
                # Propensity increases as molecules get closer
                # Use simple model: full rate within binding radius
                propensity = k_on * nM_to_nm3 * effective_volume
                binding_propensities.append((mol1, mol2, propensity))

            # Calculate unbinding propensities
            bound_pairs = self._get_bound_pairs(molecules)
            for mol1, mol2 in bound_pairs:
                k_off = self.get_unbinding_rate(mol1, mol2)
                # Convert from 1/s to 1/ms
                unbinding_propensities.append((mol1, mol2, k_off / 1000.0))

        # Total propensity
        total = sum(p[2] for p in binding_propensities) + \
                sum(p[2] for p in unbinding_propensities)

        return binding_propensities, unbinding_propensities, total

    def _get_bound_pairs(self, molecules: List[Molecule]) -> List[Tuple[Molecule, Molecule]]:
        """
        Get all bound molecule pairs (helper to avoid code duplication).

        Returns:
            pairs: List of (mol1, mol2) tuples for bound pairs
        """
        bound_pairs_set = set()
        for mol in molecules:
            if mol.is_bound and mol.bound_partner is not None:
                # Store pairs as sorted tuple to avoid duplicates
                pair = tuple(sorted([mol.molecule_id, mol.bound_partner.molecule_id]))
                bound_pairs_set.add(pair)

        # Convert to list of molecule pairs
        pairs = []
        for pair_ids in bound_pairs_set:
            mol1 = next(m for m in molecules if m.molecule_id == pair_ids[0])
            mol2 = next(m for m in molecules if m.molecule_id == pair_ids[1])
            pairs.append((mol1, mol2))

        return pairs

    def choose_event(self, binding_propensities: List, unbinding_propensities: List,
                    total_propensity: float) -> Optional[Tuple[str, Molecule, Molecule]]:
        """
        Choose next binding/unbinding event stochastically.

        Uses Gillespie algorithm: probability ∝ propensity

        Args:
            binding_propensities: List of binding events
            unbinding_propensities: List of unbinding events
            total_propensity: Sum of all propensities

        Returns:
            event: ('bind', mol1, mol2) or ('unbind', mol1, mol2) or None
        """
        if total_propensity == 0:
            return None

        # Choose event with probability proportional to propensity
        r = np.random.uniform(0, total_propensity)

        cumsum = 0
        # Check binding events
        for mol1, mol2, prop in binding_propensities:
            cumsum += prop
            if r < cumsum:
                return ('bind', mol1, mol2)

        # Check unbinding events
        for mol1, mol2, prop in unbinding_propensities:
            cumsum += prop
            if r < cumsum:
                return ('unbind', mol1, mol2)

        return None

    def execute_binding(self, mol1: Molecule, mol2: Molecule, time: float):
        """
        Execute a binding event.

        Args:
            mol1: First molecule
            mol2: Second molecule
            time: Current simulation time
        """
        mol1.is_bound = True
        mol2.is_bound = True
        mol1.bound_partner = mol2
        mol2.bound_partner = mol1

        # Record event
        self.binding_events.append({
            'time': time,
            'mol1_id': mol1.molecule_id,
            'mol2_id': mol2.molecule_id,
            'mol1_color': mol1.color,
            'mol2_color': mol2.color,
            'position': (mol1.position + mol2.position) / 2
        })

    def execute_unbinding(self, mol1: Molecule, mol2: Molecule, time: float):
        """
        Execute an unbinding event.

        Args:
            mol1: First molecule
            mol2: Second molecule
            time: Current simulation time
        """
        mol1.is_bound = False
        mol2.is_bound = False
        mol1.bound_partner = None
        mol2.bound_partner = None

        # Record event
        self.unbinding_events.append({
            'time': time,
            'mol1_id': mol1.molecule_id,
            'mol2_id': mol2.molecule_id,
            'mol1_color': mol1.color,
            'mol2_color': mol2.color,
            'position': (mol1.position + mol2.position) / 2
        })

    def process_events(self, molecules: List[Molecule], dt: float,
                      area: float, current_time: float,
                      lattice_spacing: Optional[float] = None,
                      diffusion_coeff: Optional[float] = None) -> int:
        """
        Process binding/unbinding events during a timestep.

        Uses Gillespie algorithm to determine if events occur.

        Args:
            molecules: List of all molecules
            dt: Timestep duration (ms)
            area: Simulation area (nm²)
            current_time: Current simulation time (ms)
            lattice_spacing: Lattice spacing for microscopic mode (nm)
            diffusion_coeff: Combined diffusion coefficient for microscopic mode (nm²/ms)

        Returns:
            n_events: Number of events that occurred
        """
        n_events = 0

        # Calculate propensities (pass lattice_spacing and diffusion_coeff for microscopic mode)
        bind_props, unbind_props, total_prop = self.calculate_propensities(
            molecules, area, lattice_spacing=lattice_spacing, diffusion_coeff=diffusion_coeff
        )

        if total_prop == 0:
            return 0

        # Time to next event (exponential distribution)
        # Convert rates: k_on is 1/(nM·s), k_off is 1/s
        # Need to convert to 1/ms
        total_prop_per_ms = total_prop / 1000.0  # Convert /s to /ms

        tau = np.random.exponential(1.0 / total_prop_per_ms) if total_prop_per_ms > 0 else np.inf

        # Process events that occur within this timestep
        time_elapsed = 0
        while time_elapsed + tau < dt:
            # Choose which event occurs
            event = self.choose_event(bind_props, unbind_props, total_prop)

            if event is None:
                break

            event_type, mol1, mol2 = event
            event_time = current_time + time_elapsed + tau

            if event_type == 'bind':
                self.execute_binding(mol1, mol2, event_time)
            else:  # unbind
                self.execute_unbinding(mol1, mol2, event_time)

            n_events += 1
            time_elapsed += tau

            # Recalculate propensities after event
            bind_props, unbind_props, total_prop = self.calculate_propensities(
                molecules, area, lattice_spacing=lattice_spacing, diffusion_coeff=diffusion_coeff
            )

            if total_prop == 0:
                break

            total_prop_per_ms = total_prop / 1000.0
            tau = np.random.exponential(1.0 / total_prop_per_ms) if total_prop_per_ms > 0 else np.inf

        return n_events


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
                 boundary: str = 'reflective',
                 binding_kinetics: Optional[BindingKinetics] = None,
                 lattice_spacing: float = 50.0):
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
            binding_kinetics: Optional BindingKinetics object for binding/unbinding
            lattice_spacing: Lattice spacing for microscopic binding rates (nm, default: 50.0)
        """
        self.area = np.array(area)
        self.dt = dt
        self.t_exposure = t_exposure
        self.boundary = boundary
        self.current_time = 0.0
        self.lattice_spacing = lattice_spacing

        # Diffusion engine
        self.diffusion_engine = LangevinDiffusion2D(sigma0, s0, R)

        # Binding kinetics
        self.binding_kinetics = binding_kinetics

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

    def _update_bound_pair_positions(self, mol1: Molecule, mol2: Molecule, new_position: np.ndarray):
        """
        Update positions of bound pair to move together.

        Bound molecules move to their center of mass.

        Args:
            mol1: First molecule in bound pair
            mol2: Second molecule in bound pair
            new_position: New position for the pair
        """
        mol1.position = new_position.copy()
        mol2.position = new_position.copy()

    def run(self, n_steps: int, enable_binding: bool = True, chunk_size: int = 10):
        """
        Run simulation for n_steps with optional binding kinetics.

        With binding disabled: Generates full trajectories for all molecules at once
        to preserve proper covariance structure from Michalet & Berglund (2012).

        With binding enabled: Uses chunk-based approach:
        - Generates trajectories in chunks
        - Processes binding/unbinding events between chunks
        - Bound pairs move together at D_bound

        Args:
            n_steps: Number of timesteps to simulate
            enable_binding: Whether to process binding events (default True)
            chunk_size: Steps per chunk when binding enabled (default 10)
        """
        if not enable_binding or self.binding_kinetics is None:
            # Fast path: no binding, generate full trajectories at once
            for mol in self.molecules:
                D = mol.get_current_D()
                trajectory = self.diffusion_engine.generate_trajectory_2D(
                    D=D,
                    N=n_steps,
                    dt=self.dt,
                    t_exposure=self.t_exposure,
                    starting_position=mol.position
                )

                for i in range(n_steps):
                    new_position = self._apply_boundary_condition(trajectory[i])
                    time = mol.times[-1] + (i + 1) * self.dt
                    mol.add_position(new_position, time)

            self.current_time += n_steps * self.dt
            return

        # With binding: process in chunks
        steps_remaining = n_steps
        while steps_remaining > 0:
            current_chunk = min(chunk_size, steps_remaining)

            # Process binding/unbinding events
            area_nm2 = self.area[0] * self.area[1]

            # Calculate combined diffusion coefficient for microscopic mode
            # Use average of all unbound molecules' D_free (typical approach)
            if self.binding_kinetics.use_microscopic:
                D_values = [mol.D_free for mol in self.molecules if not mol.is_bound]
                if D_values:
                    # Combined D ≈ D1 + D2 for two molecules
                    # Use 2× mean D as representative combined diffusion
                    combined_D = 2.0 * np.mean(D_values)
                else:
                    # Fallback if all molecules are bound
                    combined_D = np.mean([mol.D_free for mol in self.molecules]) * 2.0
            else:
                combined_D = None

            self.binding_kinetics.process_events(
                self.molecules, current_chunk * self.dt, area_nm2, self.current_time,
                lattice_spacing=self.lattice_spacing,
                diffusion_coeff=combined_D
            )

            # Generate trajectories for this chunk
            # Track which molecules are bound to avoid double-processing
            processed_pairs = set()

            for mol in self.molecules:
                # Skip if this molecule was already processed as part of a bound pair
                if mol.molecule_id in processed_pairs:
                    continue

                D = mol.get_current_D()

                if mol.is_bound and mol.bound_partner is not None:
                    # Generate trajectories for both partners
                    partner = mol.bound_partner

                    traj1 = self.diffusion_engine.generate_trajectory_2D(
                        D=D, N=current_chunk, dt=self.dt,
                        t_exposure=self.t_exposure,
                        starting_position=mol.position
                    )

                    traj2 = self.diffusion_engine.generate_trajectory_2D(
                        D=partner.get_current_D(), N=current_chunk, dt=self.dt,
                        t_exposure=self.t_exposure,
                        starting_position=partner.position
                    )

                    # Move bound pair to center of mass at each step
                    for i in range(current_chunk):
                        pos1 = self._apply_boundary_condition(traj1[i])
                        pos2 = self._apply_boundary_condition(traj2[i])
                        com_pos = (pos1 + pos2) / 2

                        time = self.current_time + (i + 1) * self.dt
                        mol.add_position(com_pos, time)
                        partner.add_position(com_pos, time)

                    processed_pairs.add(mol.molecule_id)
                    processed_pairs.add(partner.molecule_id)

                else:
                    # Unbound molecule moves independently
                    trajectory = self.diffusion_engine.generate_trajectory_2D(
                        D=D, N=current_chunk, dt=self.dt,
                        t_exposure=self.t_exposure,
                        starting_position=mol.position
                    )

                    for i in range(current_chunk):
                        new_position = self._apply_boundary_condition(trajectory[i])
                        time = self.current_time + (i + 1) * self.dt
                        mol.add_position(new_position, time)

                    processed_pairs.add(mol.molecule_id)

            self.current_time += current_chunk * self.dt
            steps_remaining -= current_chunk

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


# =============================================================================
# Camera Imaging Adapter - Converts diffusion trajectories to TIFF movies
# =============================================================================

class CameraAdapter:
    """
    Adapter to convert diffusion simulation trajectories to camera images.

    This class bridges DiffusionSimulator2D output to Multicolour_Simulation_Functions
    for generating realistic TIFF movies with Bayer filtering, PSF, and camera noise.

    Features:
    - Poisson brightness sampling per frame
    - Spectral profile handling for multicolor imaging
    - Blinking support (future extension)
    - Compatible with existing pyBayerSMLM analysis pipeline
    """

    def __init__(self, simulator: DiffusionSimulator2D):
        """
        Initialize camera adapter with diffusion simulator.

        Args:
            simulator: DiffusionSimulator2D instance with run trajectories
        """
        self.simulator = simulator

    def prepare_localisations_for_imaging(
        self,
        n_photons_per_dye: Dict[str, float],
        frame_indices: Optional[np.ndarray] = None,
        blinking_probability: Optional[Dict[str, float]] = None,
        poisson_brightness: bool = True,
        random_state: Optional[np.random.Generator] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Convert trajectories to localisation format for image generation.

        This prepares molecule positions, photon counts, and spectral profiles
        for each frame in the format expected by gen_camera_image_stack.

        Args:
            n_photons_per_dye: Mean photon count per dye color {'R': 1000, 'G': 800, ...}
            frame_indices: Frame indices to extract (default: all frames)
            blinking_probability: Probability of blinking off per frame (default: None = always on)
            poisson_brightness: Sample photons from Poisson distribution (default: True)
            random_state: Random generator for reproducibility

        Returns:
            x0y0: Dict[dye_name, positions array (n_frames, 2, n_molecules)]
            n_photons: Dict[dye_name, photon counts (n_frames, n_molecules)]
            spectral_profiles: Dict[dye_name, (A_R, A_G, A_B) arrays (n_molecules, 3)]
        """
        if random_state is None:
            random_state = np.random.default_rng()

        # Determine which frames to extract
        if frame_indices is None:
            # Use all frames from simulation
            n_frames = len(self.simulator.molecules[0].trajectory)
            frame_indices = np.arange(n_frames)
        else:
            n_frames = len(frame_indices)

        # Group molecules by color
        molecules_by_color: Dict[str, List[Molecule]] = {}
        for mol in self.simulator.molecules:
            color = mol.color
            if color not in molecules_by_color:
                molecules_by_color[color] = []
            molecules_by_color[color].append(mol)

        # Prepare output dictionaries
        x0y0 = {}
        n_photons = {}
        spectral_profiles = {}

        for color, molecules in molecules_by_color.items():
            n_molecules = len(molecules)

            # Initialize position array: (n_frames, 2, n_molecules)
            positions = np.zeros((n_frames, 2, n_molecules))

            # Initialize photon counts: (n_frames, n_molecules)
            photons = np.zeros((n_frames, n_molecules))

            # Get spectral profiles (constant per molecule)
            profiles = np.zeros((n_molecules, 3))  # (A_R, A_G, A_B)

            for mol_idx, mol in enumerate(molecules):
                # Extract positions at requested frames
                for frame_idx, traj_idx in enumerate(frame_indices):
                    if traj_idx < len(mol.trajectory):
                        positions[frame_idx, 0, mol_idx] = mol.trajectory[traj_idx][0]  # x
                        positions[frame_idx, 1, mol_idx] = mol.trajectory[traj_idx][1]  # y

                        # Sample photons for this frame
                        mean_photons = n_photons_per_dye[color]

                        # Check blinking
                        is_visible = True
                        if blinking_probability is not None and color in blinking_probability:
                            if random_state.random() < blinking_probability[color]:
                                is_visible = False

                        if is_visible:
                            if poisson_brightness:
                                # Poisson sampling for realistic shot noise
                                photons[frame_idx, mol_idx] = random_state.poisson(mean_photons)
                            else:
                                # Deterministic brightness
                                photons[frame_idx, mol_idx] = mean_photons
                        else:
                            # Blinked off
                            photons[frame_idx, mol_idx] = 0

                # Get spectral profile
                profiles[mol_idx, 0] = mol.spectral_profile.get('A_R', 0.33)
                profiles[mol_idx, 1] = mol.spectral_profile.get('A_G', 0.33)
                profiles[mol_idx, 2] = mol.spectral_profile.get('A_B', 0.34)

            # Store in dictionaries
            dye_name = f"dye_{color}"
            x0y0[dye_name] = positions
            n_photons[dye_name] = photons
            spectral_profiles[dye_name] = profiles

        return x0y0, n_photons, spectral_profiles

    def generate_tiff_stack(
        self,
        camera_parameters: Dict,
        wavelength: np.ndarray,
        n_photons_per_dye: Dict[str, float],
        smoothing_function,
        output_path: str,
        frame_indices: Optional[np.ndarray] = None,
        blinking_probability: Optional[Dict[str, float]] = None,
        poisson_brightness: bool = True,
        background_photons: float = 40.0,
        background_colour: List[float] = None,
        NA: float = 1.49,
        pixel_size: float = DriftConstants.XIMEA_PIXEL_SIZE_NM,
        save_tiff: bool = True,
        random_state: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate TIFF movie from diffusion trajectories using Multicolour_Simulation_Functions.

        This is the main method that converts diffusion simulation output to camera images.

        Args:
            camera_parameters: Camera calibration dict (gain, offset, variance, masks, etc.)
            wavelength: Wavelength array for spectral calculations (nm)
            n_photons_per_dye: Mean photon count per dye {'R': 1000, 'G': 800, ...}
            smoothing_function: Smoothing function for PSF (from Multicolour_Simulation_Functions)
            output_path: Path to save TIFF file
            frame_indices: Frames to render (default: all)
            blinking_probability: Per-color blinking probability (default: None)
            poisson_brightness: Use Poisson brightness sampling (default: True)
            background_photons: Background photons per pixel (default: 40)
            background_colour: RGB background weights (default: [1,1,1])
            NA: Numerical aperture (default: 1.49)
            pixel_size: Camera pixel size in nm (default: 69)
            save_tiff: Whether to save TIFF file (default: True)
            random_state: Random generator for reproducibility

        Returns:
            bayer_image: Raw Bayer-filtered image stack (n_frames, w, h)
            smoothed_image: Smoothed image stack (n_frames, w, h)
        """
        # Import required module
        _dir = str(Path(__file__).parent)
        if _dir not in sys.path:
            sys.path.insert(0, _dir)

        import Multicolour_Simulation_Functions as MSF
        import IOFunctions

        # Initialize simulation functions
        sim_funcs = MSF.MultiC_Sim_Funcs()
        io_funcs = IOFunctions.IO_Functions()

        if background_colour is None:
            background_colour = [1, 1, 1]

        if random_state is None:
            random_state = np.random.default_rng()

        # Prepare localisation data
        x0y0, n_photons, spectral_profiles = self.prepare_localisations_for_imaging(
            n_photons_per_dye=n_photons_per_dye,
            frame_indices=frame_indices,
            blinking_probability=blinking_probability,
            poisson_brightness=poisson_brightness,
            random_state=random_state,
        )

        # Calculate average emission wavelengths and pixel efficiencies per dye
        # For now, use simple weighted average based on spectral profiles
        # More sophisticated: could use SpectralFunctions.get_pixel_fractions_rawspectra

        average_emission_wavelengths = {}
        dye_pixel_efficiencies = {}

        for dye_name, profiles in spectral_profiles.items():
            # Average spectral profile across molecules
            avg_profile = np.mean(profiles, axis=0)  # (A_R, A_G, A_B)

            # Store as pixel efficiency
            dye_pixel_efficiencies[dye_name] = avg_profile

            # Estimate average wavelength (rough approximation)
            # R ~ 630nm, G ~ 530nm, B ~ 470nm
            wavelength_map = np.array([630.0, 530.0, 470.0])  # R, G, B
            avg_wavelength = np.sum(avg_profile * wavelength_map) / np.sum(avg_profile)
            average_emission_wavelengths[dye_name] = avg_wavelength

        # Convert to format expected by gen_camera_image_stack
        # Need single dye_pixel_efficiency array if all dyes have same color ratios
        # Otherwise, need to handle multiple dyes separately

        # For simplicity, merge all dyes into single "mixture"
        # by concatenating molecules along last axis

        # Determine total number of molecules and frames
        n_frames = len(frame_indices) if frame_indices is not None else len(self.simulator.molecules[0].trajectory)

        # Merge x0y0 and n_photons across dyes
        all_positions = []
        all_photons = []
        all_wavelengths = []
        all_efficiencies = []

        for dye_name in x0y0.keys():
            n_mols_this_dye = x0y0[dye_name].shape[2]

            all_positions.append(x0y0[dye_name])  # (n_frames, 2, n_mols)
            all_photons.append(n_photons[dye_name])  # (n_frames, n_mols)

            # Repeat wavelength and efficiency for each molecule
            all_wavelengths.extend([average_emission_wavelengths[dye_name]] * n_mols_this_dye)
            all_efficiencies.append(np.tile(dye_pixel_efficiencies[dye_name], (n_mols_this_dye, 1)))

        # Concatenate across molecules
        merged_positions = np.concatenate(all_positions, axis=2)  # (n_frames, 2, total_mols)
        merged_photons = np.concatenate(all_photons, axis=1)  # (n_frames, total_mols)
        merged_efficiencies = np.vstack(all_efficiencies)  # (total_mols, 3)

        # Format for gen_camera_image_stack expects:
        # x0y0: Dict[str, array(n_frames, 2, n_molecules)]
        # n_photons: Dict[str, array(n_frames)]  -- BUT we need per-molecule!
        # dye_pixel_efficiency: array(n_molecules, 3) or (3,)

        # Since gen_camera_image_stack expects single photon count per dye per frame,
        # we need to call it frame-by-frame with individual molecules

        print(f"Generating {n_frames} frames with {merged_positions.shape[2]} molecules...")

        w, h = camera_parameters['gain'].shape
        bayer_image_stack = np.zeros((n_frames, w, h))
        smoothed_image_stack = np.zeros((n_frames, w, h))

        # Process frame by frame
        for frame_idx in range(n_frames):
            # For this frame, create x0y0 and n_photons for each molecule
            # Each "molecule" is treated as a separate "dye"

            frame_x0y0 = {}
            frame_n_photons = {}

            for mol_idx in range(merged_positions.shape[2]):
                mol_name = f"mol_{mol_idx}"

                # Position for this molecule: (2, 1)
                frame_x0y0[mol_name] = merged_positions[frame_idx, :, mol_idx:mol_idx+1]

                # Photons for this molecule: scalar
                frame_n_photons[mol_name] = merged_photons[frame_idx, mol_idx]

            # Call gen_camera_image_stack for this frame
            # Use merged_efficiencies (different per molecule)
            bayer_frame, smoothed_frame, _ = sim_funcs.gen_camera_image_stack(
                camera_calibration=camera_parameters,
                wavelength=wavelength,
                average_emission_wavelengths=np.array(all_wavelengths),
                dye_pixel_efficiency=merged_efficiencies,
                n_photons=frame_n_photons,
                x0y0=frame_x0y0,
                smoothing_function=smoothing_function,
                background_photons=background_photons,
                background_colour=background_colour,
                NA=NA,
                pixel_size=pixel_size,
                return_normal_image=False,
            )

            bayer_image_stack[frame_idx] = bayer_frame
            smoothed_image_stack[frame_idx] = smoothed_frame

            if frame_idx % 10 == 0:
                print(f"  Frame {frame_idx}/{n_frames} complete", end='\r')

        print(f"\nImage generation complete!")

        # Save TIFF if requested
        if save_tiff:
            io_funcs.write_tiff(bayer_image_stack.astype(np.uint16), output_path)
            print(f"Saved TIFF stack to: {output_path}")

        return bayer_image_stack, smoothed_image_stack

    def generate_ground_truth_rgb_video(
        self,
        output_path: str,
        frame_indices: Optional[np.ndarray] = None,
        image_size_nm: Optional[Tuple[float, float]] = None,
        pixel_size_nm: float = DriftConstants.XIMEA_PIXEL_SIZE_NM,
        gaussian_width_nm: float = 50.0,
        colormap: str = 'spectral',
        save_video: bool = True,
        background_value: int = 10,
        scale_intensity: bool = True,
        max_intensity: int = 255,
        intensity_percentile: float = 99.5,
    ) -> np.ndarray:
        """
        Generate ground truth RGB video showing molecules as colored Gaussians.

        This creates a "perfect" visualization where each molecule is rendered as a
        Gaussian spot with color determined by its spectral profile. Useful for
        comparing Bayer-filtered analysis results to ground truth.

        Args:
            output_path: Path to save video (TIFF stack)
            frame_indices: Frames to render (default: all)
            image_size_nm: (width, height) in nm (default: use simulator area)
            pixel_size_nm: Pixel size for video (default: 69 nm, matching camera)
            gaussian_width_nm: Width (sigma) of Gaussian PSFs (default: 50 nm)
            colormap: How to map spectral profiles to RGB:
                      'spectral' - Blue (B-dominant) → Red (R-dominant)
                      'direct' - Use (A_R, A_G, A_B) directly as RGB
            save_video: Whether to save TIFF file
            background_value: Background pixel value (default: 10)
            scale_intensity: Whether to auto-scale intensity per frame
            max_intensity: Maximum pixel value (default: 255 for uint8)
            intensity_percentile: Percentile for intensity scaling (default: 99.5).
                                 Lower values (e.g., 95) give brighter videos by allowing
                                 bright spots to saturate. 100 = use absolute max (dimmest).

        Returns:
            rgb_video: RGB video stack (n_frames, height, width, 3) uint8
        """
        # Determine image size
        if image_size_nm is None:
            image_size_nm = self.simulator.area

        width_pixels = int(np.ceil(image_size_nm[0] / pixel_size_nm))
        height_pixels = int(np.ceil(image_size_nm[1] / pixel_size_nm))

        # Determine frames
        if frame_indices is None:
            n_frames = len(self.simulator.molecules[0].trajectory)
            frame_indices = np.arange(n_frames)
        else:
            n_frames = len(frame_indices)

        # Initialize RGB video
        rgb_video = np.zeros((n_frames, height_pixels, width_pixels, 3), dtype=np.uint8)

        # Add background
        rgb_video[:, :, :, :] = background_value

        # Create coordinate grids (in pixels)
        x_pixels = np.arange(width_pixels)
        y_pixels = np.arange(height_pixels)
        X, Y = np.meshgrid(x_pixels, y_pixels)

        # Convert Gaussian width to pixels
        sigma_pixels = gaussian_width_nm / pixel_size_nm

        print(f"Generating ground truth RGB video: {n_frames} frames, {width_pixels}×{height_pixels} pixels")

        # First pass: render all frames to float arrays
        rgb_video_float = np.zeros((n_frames, height_pixels, width_pixels, 3), dtype=np.float32)

        for frame_idx, traj_idx in enumerate(frame_indices):
            # Accumulate RGB channels
            r_channel = np.zeros((height_pixels, width_pixels), dtype=np.float32)
            g_channel = np.zeros((height_pixels, width_pixels), dtype=np.float32)
            b_channel = np.zeros((height_pixels, width_pixels), dtype=np.float32)

            # Render each molecule
            for mol in self.simulator.molecules:
                if traj_idx >= len(mol.trajectory):
                    continue

                # Get position in nm
                pos_nm = mol.trajectory[traj_idx]
                x_nm, y_nm = pos_nm[0], pos_nm[1]

                # Convert to pixels
                x_px = x_nm / pixel_size_nm
                y_px = y_nm / pixel_size_nm

                # Check if molecule is in field of view
                if x_px < -3*sigma_pixels or x_px > width_pixels + 3*sigma_pixels:
                    continue
                if y_px < -3*sigma_pixels or y_px > height_pixels + 3*sigma_pixels:
                    continue

                # Generate 2D Gaussian
                gaussian = np.exp(-((X - x_px)**2 + (Y - y_px)**2) / (2 * sigma_pixels**2))

                # Get color based on spectral profile
                if colormap == 'spectral':
                    # Map spectral profile to blue→red gradient
                    # Calculate "redness" metric: high A_R = red, high A_B = blue
                    A_R = mol.spectral_profile.get('A_R', 0.33)
                    A_G = mol.spectral_profile.get('A_G', 0.33)
                    A_B = mol.spectral_profile.get('A_B', 0.34)

                    # Compute spectral position (0=blue, 1=red)
                    # Weight: R=1, G=0.5, B=0
                    spectral_pos = (A_R * 1.0 + A_G * 0.5 + A_B * 0.0)

                    # Map to RGB using smooth gradient
                    # Blue (0, 0, 1) → Cyan (0, 1, 1) → Green (0, 1, 0) → Yellow (1, 1, 0) → Red (1, 0, 0)
                    if spectral_pos < 0.25:
                        # Blue → Cyan
                        t = spectral_pos / 0.25
                        rgb = np.array([0.0, t, 1.0])
                    elif spectral_pos < 0.5:
                        # Cyan → Green
                        t = (spectral_pos - 0.25) / 0.25
                        rgb = np.array([0.0, 1.0, 1.0 - t])
                    elif spectral_pos < 0.75:
                        # Green → Yellow
                        t = (spectral_pos - 0.5) / 0.25
                        rgb = np.array([t, 1.0, 0.0])
                    else:
                        # Yellow → Red
                        t = (spectral_pos - 0.75) / 0.25
                        rgb = np.array([1.0, 1.0 - t, 0.0])

                elif colormap == 'direct':
                    # Use spectral profile directly as RGB
                    A_R = mol.spectral_profile.get('A_R', 0.33)
                    A_G = mol.spectral_profile.get('A_G', 0.33)
                    A_B = mol.spectral_profile.get('A_B', 0.34)
                    rgb = np.array([A_R, A_G, A_B])
                    # Normalize to max = 1
                    rgb = rgb / np.max(rgb) if np.max(rgb) > 0 else rgb

                else:
                    raise ValueError(f"Unknown colormap: {colormap}")

                # Add to RGB channels (Gaussian peak = 1.0)
                r_channel += gaussian * rgb[0]
                g_channel += gaussian * rgb[1]
                b_channel += gaussian * rgb[2]

            # Store in float video
            rgb_video_float[frame_idx, :, :, 0] = r_channel
            rgb_video_float[frame_idx, :, :, 1] = g_channel
            rgb_video_float[frame_idx, :, :, 2] = b_channel

            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx}/{n_frames} complete", end='\r')

        print(f"\nRendering complete, applying intensity scaling...")

        # Second pass: apply global intensity scaling
        if scale_intensity:
            # Use percentile-based scaling for better brightness
            # This allows rare bright spots to saturate while boosting typical intensities
            if intensity_percentile >= 100.0:
                # Use absolute max (original behavior)
                scale_target = rgb_video_float.max()
                print(f"  Using absolute max intensity: {scale_target:.6f}")
            else:
                # Use percentile (makes video brighter by allowing some saturation)
                scale_target = np.percentile(rgb_video_float, intensity_percentile)
                actual_max = rgb_video_float.max()
                print(f"  Using {intensity_percentile}th percentile: {scale_target:.6f} (max: {actual_max:.6f})")

            if scale_target > 0:
                # Scale so target intensity → (max_intensity - background_value)
                scale_factor = (max_intensity - background_value) / scale_target
                rgb_video_float = rgb_video_float * scale_factor
                print(f"  Applied scale factor: {scale_factor:.2f}")
        else:
            # No scaling: use raw Gaussian intensities
            # Multiply by a reasonable factor to make visible
            # (Each Gaussian has peak ~1.0, need to scale to 0-255 range)
            scale_factor = max_intensity - background_value
            rgb_video_float = rgb_video_float * scale_factor
            print(f"  No auto-scaling: using fixed scale factor {scale_factor:.2f}")

        # Convert to uint8 with background
        rgb_video = np.clip(rgb_video_float + background_value, 0, max_intensity).astype(np.uint8)

        print(f"Ground truth video generation complete!")

        # Save if requested
        if save_video:
            # Save as proper RGB TIFF using tifffile for ImageJ compatibility
            try:
                import tifffile
                # tifffile expects shape (T, Y, X, C) for RGB time series
                # Our rgb_video is already in this format
                tifffile.imwrite(
                    output_path,
                    rgb_video,
                    photometric='rgb',
                    metadata={'axes': 'TYXC'}
                )
                print(f"Saved ground truth RGB video to: {output_path}")
                print(f"Format: RGB TIFF, {n_frames} frames × {height_pixels}×{width_pixels} pixels")
                print(f"  → Open in ImageJ/Fiji as RGB composite")
            except ImportError:
                # Fallback: save as hyperstack with separate channels
                print("Warning: tifffile not available, using fallback format")
                _dir = str(Path(__file__).parent)
                if _dir not in sys.path:
                    sys.path.insert(0, _dir)

                import IOFunctions
                io_funcs = IOFunctions.IO_Functions()

                # Save as ImageJ hyperstack: TZCYXS order
                # For RGB composite in ImageJ: (n_frames, 3, height, width)
                tiff_stack = np.zeros((n_frames, 3, height_pixels, width_pixels), dtype=np.uint8)
                for i in range(n_frames):
                    tiff_stack[i, 0] = rgb_video[i, :, :, 0]  # R
                    tiff_stack[i, 1] = rgb_video[i, :, :, 1]  # G
                    tiff_stack[i, 2] = rgb_video[i, :, :, 2]  # B

                # Reshape to (n_frames * 3, height, width) for write_tiff
                tiff_stack = tiff_stack.reshape(n_frames * 3, height_pixels, width_pixels)
                io_funcs.write_tiff(tiff_stack, output_path)
                print(f"Saved ground truth RGB video to: {output_path}")
                print(f"Format: {n_frames} frames × 3 channels = {n_frames * 3} slices")
                print(f"  → In ImageJ: Image → Color → Make Composite, set Display Mode to Composite")

        return rgb_video
