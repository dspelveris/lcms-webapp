"""Analysis functions for LC-MS data processing."""

import numpy as np
from scipy import signal
from scipy import integrate
from typing import Optional

from data_reader import SampleData


def smooth_data(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to data.

    Args:
        data: Input array to smooth
        window_size: Size of smoothing window (must be odd)

    Returns:
        Smoothed array
    """
    if data is None or len(data) < window_size:
        return data

    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Window must be smaller than data length
    if window_size >= len(data):
        window_size = len(data) - 1 if len(data) % 2 == 0 else len(data) - 2
        if window_size < 3:
            return data

    try:
        return signal.savgol_filter(data, window_size, polyorder=2)
    except Exception:
        # Fall back to simple moving average
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def extract_eic(sample: SampleData, target_mz: float, window: float = 0.5) -> Optional[np.ndarray]:
    """
    Extract ion chromatogram (EIC) for a given m/z value.

    Args:
        sample: SampleData object with loaded MS data
        target_mz: Target m/z value
        window: m/z window (+/- this value)

    Returns:
        Array of intensities at each time point, or None if no MS data
    """
    if sample.ms_scans is None or sample.ms_times is None:
        return None

    mz_min = target_mz - window
    mz_max = target_mz + window

    # Check if we have a shared m/z axis (1D scans)
    if sample.ms_mz_axis is not None:
        # Find indices within m/z window
        mask = (sample.ms_mz_axis >= mz_min) & (sample.ms_mz_axis <= mz_max)
        if not np.any(mask):
            return np.zeros(len(sample.ms_scans))

        eic = []
        for scan in sample.ms_scans:
            if scan is None:
                eic.append(0)
            elif isinstance(scan, np.ndarray) and scan.ndim == 1:
                # 1D array - use mask on shared m/z axis
                eic.append(np.sum(scan[mask]))
            else:
                eic.append(0)
        return np.array(eic)

    # Fall back to per-scan m/z extraction (2D scans)
    eic = []
    for scan in sample.ms_scans:
        if scan is None:
            eic.append(0)
            continue

        # Get m/z and intensity from scan
        try:
            mz = None
            intensity = None

            # Try different attribute names
            if hasattr(scan, 'mz') and hasattr(scan, 'intensity'):
                mz = np.array(scan.mz)
                intensity = np.array(scan.intensity)
            elif hasattr(scan, 'masses') and hasattr(scan, 'intensities'):
                mz = np.array(scan.masses)
                intensity = np.array(scan.intensities)
            elif hasattr(scan, 'mzs') and hasattr(scan, 'ints'):
                mz = np.array(scan.mzs)
                intensity = np.array(scan.ints)
            elif isinstance(scan, dict):
                mz = np.array(scan.get('mz', scan.get('masses', [])))
                intensity = np.array(scan.get('intensity', scan.get('intensities', [])))
            elif isinstance(scan, np.ndarray):
                if scan.ndim == 2 and scan.shape[1] >= 2:
                    mz = scan[:, 0]
                    intensity = scan[:, 1]
                elif scan.ndim == 1:
                    # 1D without m/z axis - can't extract EIC
                    eic.append(0)
                    continue
            elif isinstance(scan, (list, tuple)) and len(scan) >= 2:
                mz = np.array(scan[0])
                intensity = np.array(scan[1])

            if mz is None or intensity is None or len(mz) == 0:
                eic.append(0)
                continue

            # Sum intensities within m/z window
            scan_mask = (mz >= mz_min) & (mz <= mz_max)
            eic.append(np.sum(intensity[scan_mask]) if np.any(scan_mask) else 0)

        except Exception as e:
            eic.append(0)

    return np.array(eic)


def calculate_tic(sample: SampleData) -> Optional[np.ndarray]:
    """
    Calculate Total Ion Chromatogram from MS data.

    Args:
        sample: SampleData object with loaded MS data

    Returns:
        Array of total ion intensities, or None if no MS data
    """
    if sample.tic is not None:
        return sample.tic

    if sample.ms_scans is None:
        return None

    tic = []
    for scan in sample.ms_scans:
        if scan is None:
            tic.append(0)
            continue

        try:
            if hasattr(scan, 'intensity'):
                tic.append(np.sum(scan.intensity))
            elif isinstance(scan, np.ndarray):
                if scan.ndim == 2:
                    tic.append(np.sum(scan[:, 1]))
                else:
                    tic.append(np.sum(scan))
            else:
                tic.append(0)
        except Exception:
            tic.append(0)

    return np.array(tic)


def calculate_peak_area(times: np.ndarray, intensities: np.ndarray,
                        start_time: Optional[float] = None,
                        end_time: Optional[float] = None) -> float:
    """
    Calculate peak area using trapezoidal integration.

    Args:
        times: Time array
        intensities: Intensity array
        start_time: Start of integration window (optional)
        end_time: End of integration window (optional)

    Returns:
        Integrated area
    """
    if times is None or intensities is None or len(times) != len(intensities):
        return 0.0

    # Apply time window if specified
    if start_time is not None or end_time is not None:
        mask = np.ones(len(times), dtype=bool)
        if start_time is not None:
            mask &= times >= start_time
        if end_time is not None:
            mask &= times <= end_time
        times = times[mask]
        intensities = intensities[mask]

    if len(times) < 2:
        return 0.0

    return float(integrate.trapezoid(intensities, times))


def find_peaks(times: np.ndarray, intensities: np.ndarray,
               height_threshold: float = 0.1,
               prominence: float = 0.05) -> list[dict]:
    """
    Find peaks in chromatogram.

    Args:
        times: Time array
        intensities: Intensity array
        height_threshold: Minimum height as fraction of max
        prominence: Minimum prominence as fraction of max

    Returns:
        List of peak info dicts with time, intensity, area
    """
    if times is None or intensities is None or len(times) < 3:
        return []

    max_intensity = np.max(intensities)
    if max_intensity == 0:
        return []

    min_height = max_intensity * height_threshold
    min_prominence = max_intensity * prominence

    try:
        peak_indices, properties = signal.find_peaks(
            intensities,
            height=min_height,
            prominence=min_prominence
        )
    except Exception:
        return []

    peaks = []
    for i, idx in enumerate(peak_indices):
        # Estimate peak boundaries for area calculation
        left_base = properties.get('left_bases', [idx - 1])[i] if 'left_bases' in properties else max(0, idx - 5)
        right_base = properties.get('right_bases', [idx + 1])[i] if 'right_bases' in properties else min(len(times) - 1, idx + 5)

        area = calculate_peak_area(
            times[left_base:right_base + 1],
            intensities[left_base:right_base + 1]
        )

        peaks.append({
            'time': times[idx],
            'intensity': intensities[idx],
            'area': area,
            'index': idx
        })

    return peaks


def normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize data to 0-1 range."""
    if data is None or len(data) == 0:
        return data

    data_min = np.min(data)
    data_max = np.max(data)

    if data_max - data_min == 0:
        return np.zeros_like(data)

    return (data - data_min) / (data_max - data_min)


def baseline_correction(data: np.ndarray, percentile: float = 5) -> np.ndarray:
    """
    Simple baseline correction using percentile.

    Args:
        data: Input array
        percentile: Percentile to use as baseline estimate

    Returns:
        Baseline-corrected array
    """
    if data is None or len(data) == 0:
        return data

    baseline = np.percentile(data, percentile)
    corrected = data - baseline
    corrected[corrected < 0] = 0

    return corrected


def sum_spectra_in_range(sample: 'SampleData', start_time: float, end_time: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Sum mass spectra within a time range.

    Args:
        sample: SampleData object with loaded MS data
        start_time: Start time in minutes
        end_time: End time in minutes

    Returns:
        Tuple of (mz_array, intensity_array)
    """
    if sample.ms_scans is None or sample.ms_times is None:
        return np.array([]), np.array([])

    # Find time indices within range
    time_mask = (sample.ms_times >= start_time) & (sample.ms_times <= end_time)
    scan_indices = np.where(time_mask)[0]

    if len(scan_indices) == 0:
        return np.array([]), np.array([])

    # If we have a shared m/z axis
    if sample.ms_mz_axis is not None:
        mz_axis = sample.ms_mz_axis
        summed_intensities = np.zeros(len(mz_axis))

        for idx in scan_indices:
            scan = sample.ms_scans[idx]
            if scan is not None and isinstance(scan, np.ndarray):
                summed_intensities += scan

        return mz_axis, summed_intensities

    # Otherwise, need to bin the spectra
    # Collect all m/z and intensities first
    all_mz = []
    all_int = []

    for idx in scan_indices:
        scan = sample.ms_scans[idx]
        if scan is None:
            continue

        mz, intensity = None, None

        if hasattr(scan, 'mz') and hasattr(scan, 'intensity'):
            mz = np.array(scan.mz)
            intensity = np.array(scan.intensity)
        elif hasattr(scan, 'masses') and hasattr(scan, 'intensities'):
            mz = np.array(scan.masses)
            intensity = np.array(scan.intensities)
        elif isinstance(scan, np.ndarray) and scan.ndim == 2:
            mz = scan[:, 0]
            intensity = scan[:, 1]

        if mz is not None and intensity is not None and len(mz) > 0:
            all_mz.extend(mz)
            all_int.extend(intensity)

    if len(all_mz) == 0:
        return np.array([]), np.array([])

    # Bin the data
    mz_min, mz_max = min(all_mz), max(all_mz)
    bin_width = 0.1  # 0.1 Da bins
    bins = np.arange(mz_min, mz_max + bin_width, bin_width)

    binned_intensity, bin_edges = np.histogram(all_mz, bins=bins, weights=all_int)
    mz_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return mz_centers, binned_intensity


def centroid_peak(mz: np.ndarray, intensity: np.ndarray, peak_idx: int, window: int = 3) -> float:
    """
    Calculate centroid m/z for a peak using intensity-weighted average.

    Args:
        mz: m/z array
        intensity: intensity array
        peak_idx: Index of peak maximum
        window: Number of points on each side to include

    Returns:
        Centroid m/z value
    """
    left = max(0, peak_idx - window)
    right = min(len(mz), peak_idx + window + 1)

    mz_window = mz[left:right]
    int_window = intensity[left:right]

    # Use only the top portion of the peak (above 50% of max in window)
    max_int = np.max(int_window)
    threshold = max_int * 0.5
    mask = int_window >= threshold

    if not np.any(mask):
        return mz[peak_idx]

    mz_filtered = mz_window[mask]
    int_filtered = int_window[mask]

    # Subtract baseline
    baseline = np.min(int_filtered)
    int_corrected = int_filtered - baseline

    total_int = np.sum(int_corrected)
    if total_int == 0:
        return mz[peak_idx]

    # Intensity-weighted centroid
    centroid = np.sum(mz_filtered * int_corrected) / total_int
    return centroid


def find_spectrum_peaks(mz: np.ndarray, intensity: np.ndarray,
                        height_threshold: float = 0.01,
                        min_distance: int = 3,
                        use_centroid: bool = True) -> list[dict]:
    """
    Find peaks in a mass spectrum.

    Args:
        mz: m/z array
        intensity: intensity array
        height_threshold: Minimum height as fraction of max
        min_distance: Minimum distance between peaks in data points
        use_centroid: If True, calculate centroid m/z for better precision

    Returns:
        List of peak dicts with mz and intensity
    """
    if len(mz) == 0 or len(intensity) == 0:
        return []

    max_int = np.max(intensity)
    if max_int == 0:
        return []

    min_height = max_int * height_threshold

    try:
        peak_indices, properties = signal.find_peaks(
            intensity,
            height=min_height,
            distance=min_distance
        )
    except Exception:
        return []

    peaks = []
    for idx in peak_indices:
        if use_centroid:
            peak_mz = centroid_peak(mz, intensity, idx)
        else:
            peak_mz = mz[idx]

        peaks.append({
            'mz': peak_mz,
            'intensity': intensity[idx],
            'index': idx
        })

    # Sort by intensity descending
    peaks.sort(key=lambda x: x['intensity'], reverse=True)
    return peaks


def deconvolute_protein(mz: np.ndarray, intensity: np.ndarray,
                        min_charge: int = 5, max_charge: int = 60,
                        mass_tolerance: float = 1.0,
                        min_peaks: int = 3,
                        max_peaks: int = 50) -> list[dict]:
    """
    Perform protein mass deconvolution from multiply-charged ions.

    Algorithm:
    For each pair of adjacent peaks, calculate potential charge state:
    z = m2 / (m1 - m2) where m1 > m2
    Then M = m1 * z - z * 1.00784 (proton mass)

    Args:
        mz: m/z array
        intensity: intensity array
        min_charge: Minimum charge state to consider
        max_charge: Maximum charge state to consider
        mass_tolerance: Mass tolerance in Da for grouping
        min_peaks: Minimum number of charge states to confirm a mass

    Returns:
        List of deconvoluted mass results
    """
    PROTON_MASS = 1.00784

    # Find peaks
    peaks = find_spectrum_peaks(mz, intensity, height_threshold=0.01, min_distance=2)

    if len(peaks) < min_peaks:
        return []

    # Get top peaks (limit to max_peaks to control processing)
    peaks = peaks[:max_peaks]
    peak_mzs = np.array([p['mz'] for p in peaks])
    peak_ints = np.array([p['intensity'] for p in peaks])

    # Determine data resolution
    if len(mz) > 1:
        resolution = np.median(np.diff(mz))
    else:
        resolution = 1.0

    # Adjust charge tolerance based on resolution
    charge_tolerance = 0.3 if resolution < 0.5 else 0.5

    # Calculate potential masses from all peak pairs
    candidate_masses = []

    for i, mz1 in enumerate(peak_mzs):
        for j, mz2 in enumerate(peak_mzs):
            if i >= j or mz1 <= mz2:
                continue

            # Calculate charge from adjacent peaks
            delta_mz = mz1 - mz2
            if delta_mz < 0.5 or delta_mz > 200:  # Reasonable range for proteins
                continue

            z = mz2 / delta_mz
            z_rounded = round(z)

            if z_rounded < min_charge or z_rounded > max_charge:
                continue

            # Check if z is close to an integer (more tolerant for low-res data)
            if abs(z - z_rounded) > charge_tolerance:
                continue

            # Calculate neutral mass from both peaks
            mass1 = (mz1 * z_rounded) - (z_rounded * PROTON_MASS)
            mass2 = (mz2 * (z_rounded + 1)) - ((z_rounded + 1) * PROTON_MASS)

            # Average the two mass estimates
            mass_avg = (mass1 + mass2) / 2

            candidate_masses.append({
                'mass': mass_avg,
                'charge': z_rounded,
                'mz1': mz1,
                'mz2': mz2,
                'intensity': (peak_ints[i] + peak_ints[j]) / 2
            })

    if len(candidate_masses) == 0:
        return []

    # Group masses within tolerance (scale tolerance with mass for large proteins)
    masses = np.array([c['mass'] for c in candidate_masses])

    # Use percentage-based tolerance for larger masses
    def get_tolerance(mass):
        base_tol = mass_tolerance
        # Add 0.01% of mass for larger proteins
        return base_tol + mass * 0.0005

    # Cluster similar masses
    results = []
    used = set()

    # Sort by mass for consistent clustering
    sorted_indices = np.argsort(masses)

    for i in sorted_indices:
        if i in used:
            continue

        mass = masses[i]
        tol = get_tolerance(mass)

        # Find all masses within tolerance
        group_indices = np.where(np.abs(masses - mass) < tol)[0]

        if len(group_indices) < min_peaks:
            continue

        for idx in group_indices:
            used.add(idx)

        # Calculate median mass (more robust than mean) and collect charge states
        group_masses = [candidate_masses[idx]['mass'] for idx in group_indices]
        group_charges = [candidate_masses[idx]['charge'] for idx in group_indices]
        group_intensities = [candidate_masses[idx]['intensity'] for idx in group_indices]

        median_mass = np.median(group_masses)
        std_mass = np.std(group_masses)
        unique_charges = sorted(set(group_charges))
        total_intensity = sum(group_intensities)

        results.append({
            'mass': median_mass,
            'mass_std': std_mass,
            'charge_states': unique_charges,
            'num_charges': len(unique_charges),
            'intensity': total_intensity,
            'peaks_found': len(group_indices)
        })

    # Sort by number of charge states first, then intensity
    results.sort(key=lambda x: (x['num_charges'], x['intensity']), reverse=True)

    return results


def get_theoretical_mz(mass: float, charge_states: list[int]) -> list[dict]:
    """
    Calculate theoretical m/z values for a given mass and charge states.

    Args:
        mass: Neutral mass in Da
        charge_states: List of charge states

    Returns:
        List of dicts with charge and mz
    """
    PROTON_MASS = 1.00784

    result = []
    for z in charge_states:
        mz = (mass + z * PROTON_MASS) / z
        result.append({'charge': z, 'mz': mz})

    return result
