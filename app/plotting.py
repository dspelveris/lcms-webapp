"""Plotting functions for LC-MS data visualization."""

import io
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Set smaller default font sizes
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 10
})

import config
from data_reader import SampleData
from analysis import smooth_data, extract_eic


def create_single_panel(
    ax: plt.Axes,
    times: np.ndarray,
    data: np.ndarray,
    label: str = "",
    color: str = "#1f77b4",
    xlabel: str = "Time (min)",
    ylabel: str = "Intensity",
    smoothing: int = 0,
    line_width: float = 0.8,
    show_grid: bool = False,
    y_scale: str = "linear"
) -> None:
    """
    Create a single chromatogram panel.

    Args:
        ax: Matplotlib axes to plot on
        times: Time array
        data: Intensity array
        label: Legend label
        color: Line color
        xlabel: X-axis label
        ylabel: Y-axis label
        smoothing: Smoothing window size (0 = no smoothing)
        line_width: Width of the plot line
        show_grid: Whether to show grid
        y_scale: Y-axis scale ('linear' or 'log')
    """
    if times is None or data is None:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
        return

    plot_data = smooth_data(data, smoothing) if smoothing > 0 else data

    ax.plot(times, plot_data, color=color, label=label, linewidth=line_width)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if label:
        ax.legend(loc='upper right')

    if show_grid:
        ax.grid(True, alpha=0.3)

    if y_scale == 'log':
        ax.set_yscale('log')
    else:
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 3))

    ax.set_xlim(times[0], times[-1])


def create_overlay_panel(
    ax: plt.Axes,
    samples: list[tuple[str, np.ndarray, np.ndarray]],
    colors: Optional[list[str]] = None,
    xlabel: str = "Time (min)",
    ylabel: str = "Intensity",
    smoothing: int = 0,
    normalize: bool = False,
    line_width: float = 0.8,
    show_grid: bool = False,
    y_scale: str = "linear"
) -> None:
    """
    Create an overlay plot with multiple traces.

    Args:
        ax: Matplotlib axes to plot on
        samples: List of (label, times, data) tuples
        colors: Optional list of colors for each trace
        xlabel: X-axis label
        ylabel: Y-axis label
        smoothing: Smoothing window size
        normalize: Whether to normalize each trace
        line_width: Width of plot lines
        show_grid: Whether to show grid
        y_scale: Y-axis scale ('linear' or 'log')
    """
    if not samples:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
        return

    if colors is None:
        colors = config.EIC_COLORS

    x_min, x_max = float('inf'), float('-inf')

    for i, (label, times, data) in enumerate(samples):
        if times is None or data is None:
            continue

        color = colors[i % len(colors)]
        plot_data = smooth_data(data, smoothing) if smoothing > 0 else data

        if normalize and np.max(plot_data) > 0:
            plot_data = plot_data / np.max(plot_data)

        ax.plot(times, plot_data, color=color, label=label, linewidth=line_width)

        x_min = min(x_min, times[0])
        x_max = max(x_max, times[-1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right', fontsize='small')

    if x_min < x_max:
        ax.set_xlim(x_min, x_max)

    if show_grid:
        ax.grid(True, alpha=0.3)

    if y_scale == 'log':
        ax.set_yscale('log')
    else:
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 3))


def create_time_progression_figure(
    samples: list[SampleData],
    labels: list[str],
    uv_wavelength: float = config.UV_WAVELENGTH,
    eic_targets: Optional[list[float]] = None,
    style: Optional[dict] = None,
    mz_window: float = config.DEFAULT_MZ_WINDOW,
    uv_smoothing: int = config.UV_SMOOTHING_WINDOW,
    eic_smoothing: int = config.EIC_SMOOTHING_WINDOW
) -> matplotlib.figure.Figure:
    """
    Create a multi-panel figure comparing samples across timepoints.

    Args:
        samples: List of SampleData objects (should be 2-3)
        labels: Labels for each sample (e.g., ["Initial", "Mid", "Overnight"])
        uv_wavelength: UV wavelength to extract
        eic_targets: List of m/z values for EIC extraction
        style: Dictionary with style settings (colors, line_width, show_grid, etc.)
        mz_window: m/z window for EIC extraction
        uv_smoothing: Smoothing window for UV
        eic_smoothing: Smoothing window for EIC

    Returns:
        Matplotlib Figure object
    """
    # Default style settings
    if style is None:
        style = {}
    fig_width = style.get('fig_width', 10)
    fig_height_per_panel = style.get('fig_height_per_panel', 3)
    line_width = style.get('line_width', 0.8)
    show_grid = style.get('show_grid', False)
    y_scale = style.get('y_scale', 'linear')
    custom_colors = style.get('colors', {})
    labels_config = style.get('labels', {})

    # Get label templates
    title = labels_config.get('title_progression', 'Time Progression Analysis')
    x_label = labels_config.get('x_label', 'Time (min)')
    y_label_uv_template = labels_config.get('y_label_uv', 'UV {wavelength}nm (mAU)')
    y_label_tic = labels_config.get('y_label_tic', 'TIC Intensity')
    y_label_eic = labels_config.get('y_label_eic', 'EIC Intensity')
    panel_title_uv_template = labels_config.get('panel_title_uv', 'UV Chromatogram ({wavelength} nm)')
    panel_title_tic = labels_config.get('panel_title_tic', 'Total Ion Chromatogram (TIC)')
    panel_title_eic_template = labels_config.get('panel_title_eic', 'EIC m/z {mz} (±{window})')

    if eic_targets is None:
        eic_targets = config.DEFAULT_MZ_VALUES

    n_eics = len(eic_targets)
    n_rows = 2 + n_eics  # UV + TIC + EICs

    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height_per_panel * n_rows))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=1.005)

    # Define colors based on number of samples
    color_initial = custom_colors.get('initial', config.TIME_COLORS["initial"])
    color_mid = custom_colors.get('mid', config.TIME_COLORS["mid"])
    color_final = custom_colors.get('final', config.TIME_COLORS["final"])

    if len(samples) == 2:
        colors = [color_initial, color_final]
    else:
        colors = [color_initial, color_mid, color_final]

    # Format label templates
    y_label_uv = y_label_uv_template.format(wavelength=uv_wavelength)
    panel_title_uv = panel_title_uv_template.format(wavelength=uv_wavelength)

    # Panel 1: UV overlay
    uv_traces = []
    for i, sample in enumerate(samples):
        uv_data = sample.get_uv_at_wavelength(uv_wavelength)
        if uv_data is not None and sample.uv_times is not None:
            uv_traces.append((labels[i], sample.uv_times, uv_data))

    create_overlay_panel(
        axes[0], uv_traces, colors=colors,
        xlabel=x_label,
        ylabel=y_label_uv,
        smoothing=uv_smoothing,
        line_width=line_width,
        show_grid=show_grid,
        y_scale=y_scale
    )
    axes[0].set_title(panel_title_uv)

    # Panel 2: TIC overlay
    tic_traces = []
    for i, sample in enumerate(samples):
        if sample.tic is not None and sample.ms_times is not None:
            tic_traces.append((labels[i], sample.ms_times, sample.tic))

    create_overlay_panel(
        axes[1], tic_traces, colors=colors,
        xlabel=x_label,
        ylabel=y_label_tic,
        smoothing=eic_smoothing,
        line_width=line_width,
        show_grid=show_grid,
        y_scale=y_scale
    )
    axes[1].set_title(panel_title_tic)

    # Panels 3+: EIC for each target m/z
    for j, target_mz in enumerate(eic_targets):
        eic_traces = []
        for i, sample in enumerate(samples):
            eic = extract_eic(sample, target_mz, mz_window)
            if eic is not None and sample.ms_times is not None:
                eic_traces.append((labels[i], sample.ms_times, eic))

        panel_title_eic = panel_title_eic_template.format(mz=f"{target_mz:.2f}", window=mz_window)

        create_overlay_panel(
            axes[2 + j], eic_traces, colors=colors,
            xlabel=x_label,
            ylabel=y_label_eic,
            smoothing=eic_smoothing,
            line_width=line_width,
            show_grid=show_grid,
            y_scale=y_scale
        )
        axes[2 + j].set_title(panel_title_eic)

    plt.tight_layout()
    return fig


def create_single_sample_figure(
    sample: SampleData,
    uv_wavelength: float = config.UV_WAVELENGTH,
    uv_wavelengths: Optional[list[float]] = None,
    eic_targets: Optional[list[float]] = None,
    style: Optional[dict] = None,
    mz_window: float = config.DEFAULT_MZ_WINDOW,
    uv_smoothing: int = config.UV_SMOOTHING_WINDOW,
    eic_smoothing: int = config.EIC_SMOOTHING_WINDOW
) -> matplotlib.figure.Figure:
    """
    Create a multi-panel figure for a single sample.

    Args:
        sample: SampleData object
        uv_wavelength: UV wavelength to extract (deprecated, use uv_wavelengths)
        uv_wavelengths: List of UV wavelengths to extract
        eic_targets: List of m/z values for EIC extraction
        style: Dictionary with style settings
        mz_window: m/z window for EIC extraction
        uv_smoothing: Smoothing window for UV
        eic_smoothing: Smoothing window for EIC

    Returns:
        Matplotlib Figure object
    """
    # Default style settings
    if style is None:
        style = {}
    fig_width = style.get('fig_width', 10)
    fig_height_per_panel = style.get('fig_height_per_panel', 3)
    line_width = style.get('line_width', 0.8)
    show_grid = style.get('show_grid', False)
    y_scale = style.get('y_scale', 'linear')
    labels_config = style.get('labels', {})

    # Get label templates
    title_template = labels_config.get('title_single', 'Sample: {name}')
    x_label = labels_config.get('x_label', 'Time (min)')
    y_label_uv_template = labels_config.get('y_label_uv', 'UV {wavelength}nm (mAU)')
    y_label_tic = labels_config.get('y_label_tic', 'TIC Intensity')
    y_label_eic = labels_config.get('y_label_eic', 'EIC Intensity')
    panel_title_uv_template = labels_config.get('panel_title_uv', 'UV Chromatogram ({wavelength} nm)')
    panel_title_tic = labels_config.get('panel_title_tic', 'Total Ion Chromatogram (TIC)')
    panel_title_eic_template = labels_config.get('panel_title_eic', 'EIC m/z {mz} (±{window})')

    # Format label templates
    title = title_template.format(name=sample.name)

    # Support both single wavelength (legacy) and multiple wavelengths
    if uv_wavelengths is None or len(uv_wavelengths) == 0:
        uv_wavelengths = [uv_wavelength]

    if eic_targets is None:
        eic_targets = []

    n_uvs = len(uv_wavelengths)
    n_eics = len(eic_targets)
    n_rows = n_uvs + 1 + n_eics  # UV panels + TIC + EICs

    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height_per_panel * n_rows))
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=10, fontweight='bold', y=1.005)

    # UV panels - one for each wavelength
    for i, wl in enumerate(uv_wavelengths):
        uv_data = sample.get_uv_at_wavelength(wl)
        y_label_uv = y_label_uv_template.format(wavelength=wl)
        panel_title_uv = panel_title_uv_template.format(wavelength=wl)
        create_single_panel(
            axes[i],
            sample.uv_times, uv_data,
            xlabel=x_label,
            ylabel=y_label_uv,
            color="#1f77b4",
            smoothing=uv_smoothing,
            line_width=line_width,
            show_grid=show_grid,
            y_scale=y_scale
        )
        axes[i].set_title(panel_title_uv)

    # TIC panel (after all UV panels)
    tic_idx = n_uvs
    create_single_panel(
        axes[tic_idx],
        sample.ms_times, sample.tic,
        xlabel=x_label,
        ylabel=y_label_tic,
        color="#ff7f0e",
        smoothing=eic_smoothing,
        line_width=line_width,
        show_grid=show_grid,
        y_scale=y_scale
    )
    axes[tic_idx].set_title(panel_title_tic)

    # EIC panels
    for j, target_mz in enumerate(eic_targets):
        eic = extract_eic(sample, target_mz, mz_window)
        panel_title_eic = panel_title_eic_template.format(mz=f"{target_mz:.2f}", window=mz_window)
        create_single_panel(
            axes[tic_idx + 1 + j],
            sample.ms_times, eic,
            xlabel=x_label,
            ylabel=y_label_eic,
            color=config.EIC_COLORS[j % len(config.EIC_COLORS)],
            smoothing=eic_smoothing,
            line_width=line_width,
            show_grid=show_grid,
            y_scale=y_scale
        )
        axes[tic_idx + 1 + j].set_title(panel_title_eic)

    plt.tight_layout()
    return fig


def create_eic_comparison_figure(
    sample: SampleData,
    mz_values: list[float],
    mz_window: float = config.DEFAULT_MZ_WINDOW,
    smoothing: int = config.EIC_SMOOTHING_WINDOW,
    overlay: bool = True
) -> matplotlib.figure.Figure:
    """
    Create a figure comparing multiple EICs from a single sample.

    Args:
        sample: SampleData object
        mz_values: List of m/z values to extract
        mz_window: m/z window for extraction
        smoothing: Smoothing window size
        overlay: If True, overlay all EICs; if False, separate panels

    Returns:
        Matplotlib Figure object
    """
    if overlay:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(f"EIC Comparison: {sample.name}", fontsize=10, fontweight='bold')

        eic_traces = []
        for mz in mz_values:
            eic = extract_eic(sample, mz, mz_window)
            if eic is not None and sample.ms_times is not None:
                eic_traces.append((f"m/z {mz:.2f}", sample.ms_times, eic))

        create_overlay_panel(ax, eic_traces, smoothing=smoothing, normalize=True)
        ax.set_ylabel("Normalized Intensity")
    else:
        n_panels = len(mz_values)
        fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels))
        if n_panels == 1:
            axes = [axes]

        fig.suptitle(f"EIC Comparison: {sample.name}", fontsize=10, fontweight='bold')

        for i, mz in enumerate(mz_values):
            eic = extract_eic(sample, mz, mz_window)
            create_single_panel(
                axes[i],
                sample.ms_times, eic,
                label=f"m/z {mz:.2f}",
                color=config.EIC_COLORS[i % len(config.EIC_COLORS)],
                smoothing=smoothing
            )
            axes[i].set_title(f"EIC m/z {mz:.2f} (±{mz_window})")

    plt.tight_layout()
    return fig


def export_figure(fig: matplotlib.figure.Figure, dpi: int = config.EXPORT_DPI, format: str = 'png') -> bytes:
    """
    Export figure to bytes in specified format with transparent background.

    Args:
        fig: Matplotlib Figure object
        dpi: Resolution in dots per inch
        format: Output format ('png', 'svg', 'pdf')

    Returns:
        Image/document as bytes
    """
    # Make all axes backgrounds transparent
    for ax in fig.get_axes():
        ax.set_facecolor('none')
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight', facecolor='none', edgecolor='none', transparent=True)
    buf.seek(0)
    return buf.getvalue()


def export_figure_png(fig: matplotlib.figure.Figure, dpi: int = config.EXPORT_DPI) -> bytes:
    """Export figure to PNG bytes."""
    return export_figure(fig, dpi=dpi, format='png')


def export_figure_svg(fig: matplotlib.figure.Figure) -> bytes:
    """Export figure to SVG bytes with transparent background."""
    # Make all axes backgrounds transparent
    for ax in fig.get_axes():
        ax.set_facecolor('none')
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', facecolor='none', edgecolor='none', transparent=True)
    buf.seek(0)
    return buf.getvalue()


def export_figure_pdf(fig: matplotlib.figure.Figure, dpi: int = config.EXPORT_DPI) -> bytes:
    """Export figure to PDF bytes."""
    return export_figure(fig, dpi=dpi, format='pdf')


def export_figure_to_file(fig: matplotlib.figure.Figure, filepath: str, dpi: int = config.EXPORT_DPI) -> None:
    """
    Export figure to file.

    Args:
        fig: Matplotlib Figure object
        filepath: Output file path
        dpi: Resolution in dots per inch
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')


def create_mass_spectrum_figure(mz: np.ndarray, intensity: np.ndarray,
                                 title: str = "Mass Spectrum",
                                 mz_range: tuple = None,
                                 highlight_peaks: list = None,
                                 deconv_results: list = None,
                                 style: dict = None) -> matplotlib.figure.Figure:
    """
    Create a mass spectrum figure with optional peak highlighting and deconvolution overlay.

    Args:
        mz: m/z array
        intensity: intensity array
        title: Figure title
        mz_range: Optional (min, max) to zoom
        highlight_peaks: List of m/z values to highlight
        deconv_results: Deconvolution results to overlay theoretical peaks
        style: Style settings dict

    Returns:
        Matplotlib Figure
    """
    style = style or {}
    fig_width = style.get('fig_width', 12)
    line_width = style.get('line_width', 0.8)
    show_grid = style.get('show_grid', True)

    if deconv_results:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, 8),
                                        gridspec_kw={'height_ratios': [2, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(fig_width, 5))

    # Plot mass spectrum
    ax1.plot(mz, intensity, 'b-', linewidth=line_width)
    ax1.set_xlabel("m/z")
    ax1.set_ylabel("Intensity")
    ax1.set_title(title, fontweight='bold')

    if mz_range:
        ax1.set_xlim(mz_range)

    if show_grid:
        ax1.grid(True, alpha=0.3)

    ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    # Highlight peaks if provided
    if highlight_peaks:
        for peak_mz in highlight_peaks:
            ax1.axvline(x=peak_mz, color='red', linestyle='--', alpha=0.5, linewidth=0.5)

    # Plot deconvoluted mass spectrum
    if deconv_results and len(deconv_results) > 0:
        masses = [r['mass'] for r in deconv_results]
        intensities = [r['intensity'] for r in deconv_results]

        # Normalize intensities
        max_int = max(intensities) if intensities else 1
        norm_intensities = [i / max_int * 100 for i in intensities]

        ax2.bar(masses, norm_intensities, width=50, color='green', alpha=0.7)
        ax2.set_xlabel("Deconvoluted Mass (Da)")
        ax2.set_ylabel("Relative Intensity (%)")
        ax2.set_title("Deconvoluted Mass Spectrum", fontweight='bold')

        if show_grid:
            ax2.grid(True, alpha=0.3)

        # Add mass labels with appropriate precision
        for mass, rel_int in zip(masses, norm_intensities):
            if rel_int > 10:  # Only label significant peaks
                if mass >= 10000:
                    label_text = f"{mass:.1f}"
                else:
                    label_text = f"{mass:.2f}"
                ax2.annotate(label_text,
                           xy=(mass, rel_int),
                           xytext=(0, 5),
                           textcoords='offset points',
                           ha='center', fontsize=8)

    plt.tight_layout()
    return fig


def create_deconvolution_figure(sample, start_time: float, end_time: float,
                                 deconv_results: list,
                                 style: dict = None) -> matplotlib.figure.Figure:
    """
    Create a comprehensive deconvolution figure with chromatogram, mass spectrum, and results.

    Args:
        sample: SampleData object
        start_time: Start of selected region
        end_time: End of selected region
        deconv_results: Deconvolution results
        style: Style settings dict

    Returns:
        Matplotlib Figure
    """
    from analysis import sum_spectra_in_range, get_theoretical_mz

    style = style or {}
    fig_width = style.get('fig_width', 8)  # Smaller default
    line_width = style.get('line_width', 0.8)
    show_grid = style.get('show_grid', True)

    fig = plt.figure(figsize=(fig_width, 5.5))  # Smaller, less zoomed in

    # Create grid: top row for chromatogram, bottom left for spectrum, bottom right for deconv
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], hspace=0.3, wspace=0.3)

    # Top: TIC with selected region highlighted
    ax_tic = fig.add_subplot(gs[0, :])

    if sample.tic is not None and sample.ms_times is not None:
        ax_tic.plot(sample.ms_times, sample.tic, 'b-', linewidth=line_width)
        ax_tic.axvspan(start_time, end_time, alpha=0.3, color='yellow', label='Selected region')
        ax_tic.set_xlabel("Time (min)")
        ax_tic.set_ylabel("TIC Intensity")
        ax_tic.set_title(f"TIC - Selected region: {start_time:.2f} - {end_time:.2f} min", fontweight='bold')
        ax_tic.legend()
        if show_grid:
            ax_tic.grid(True, alpha=0.3)
        ax_tic.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    # Bottom left: Summed mass spectrum
    ax_spec = fig.add_subplot(gs[1, 0])

    mz, intensity = sum_spectra_in_range(sample, start_time, end_time)

    if len(mz) > 0:
        ax_spec.plot(mz, intensity, 'b-', linewidth=line_width)
        ax_spec.set_xlabel("m/z")
        ax_spec.set_ylabel("Intensity")
        ax_spec.set_title("Summed Mass Spectrum", fontweight='bold')
        if show_grid:
            ax_spec.grid(True, alpha=0.3)
        ax_spec.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

        # Add peak labels (only significant peaks > 20% of max)
        from analysis import find_spectrum_peaks
        peaks = find_spectrum_peaks(mz, intensity, height_threshold=0.2, min_distance=5, use_centroid=True)
        for peak in peaks:
            ax_spec.annotate(
                f"{peak['mz']:.2f}",
                xy=(peak['mz'], peak['intensity']),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                fontsize=5,
                rotation=90
            )
        # Add headroom for labels
        y_max = intensity.max() if len(intensity) > 0 else 1
        ax_spec.set_ylim(0, y_max * 1.2)

        # Overlay theoretical peaks for top result
        if deconv_results and len(deconv_results) > 0:
            top_result = deconv_results[0]
            theoretical = get_theoretical_mz(top_result['mass'], top_result['charge_states'])
            for t in theoretical:
                ax_spec.axvline(x=t['mz'], color='red', linestyle='--', alpha=0.5, linewidth=0.5)

    # Bottom right: Deconvoluted masses (linear kDa scale with vertical lines)
    ax_deconv = fig.add_subplot(gs[1, 1])

    # Color palette for bars and labels
    bar_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    label_colors = ['#1a6b1a', '#0d4f8a', '#cc6600', '#a31d1d', '#6b4a91',
                    '#5c3a32', '#b8518f', '#4d4d4d', '#8a8b19', '#0f8a94']

    if deconv_results and len(deconv_results) > 0:
        # Use all results passed (already limited by caller)
        masses = [r['mass'] for r in deconv_results]
        intensities = [r['intensity'] for r in deconv_results]

        max_int = max(intensities) if intensities else 1
        norm_intensities = [i / max_int * 100 for i in intensities]

        # Convert to kDa for x-axis
        masses_kda = [m / 1000 for m in masses]

        # Draw vertical lines (stem plot style) with different colors
        for i, (m_kda, intensity) in enumerate(zip(masses_kda, norm_intensities)):
            bar_color = bar_colors[i % len(bar_colors)]
            ax_deconv.vlines(x=m_kda, ymin=0, ymax=intensity, color=bar_color, linewidth=2)

        # Set x-axis range: min-20% to max+20%
        min_mass = min(masses_kda)
        max_mass = max(masses_kda)
        mass_range = max_mass - min_mass if max_mass > min_mass else max_mass * 0.1
        x_margin = mass_range * 0.2 if mass_range > 0 else max_mass * 0.2
        ax_deconv.set_xlim(min_mass - x_margin, max_mass + x_margin)
        ax_deconv.set_ylim(0, 120)  # Extra headroom for labels

        # Add mass labels with overlap avoidance
        # Sort by x position for overlap detection
        labeled_peaks = sorted(enumerate(zip(masses_kda, norm_intensities, masses)), key=lambda x: x[1][0])
        label_positions = []  # Track (x, y) of placed labels

        for orig_idx, (m_kda, intensity, mass_da) in labeled_peaks:
            # Default label position
            label_y = intensity + 3
            label_x = m_kda

            # Check for overlap with existing labels
            for prev_x, prev_y in label_positions:
                x_dist = abs(m_kda - prev_x) / (mass_range + 0.001) * 100  # As % of range
                y_dist = abs(label_y - prev_y)
                if x_dist < 15 and y_dist < 10:  # Too close
                    label_y = prev_y + 12  # Stagger vertically

            label_positions.append((label_x, label_y))

            # Format label: show Da value with appropriate precision
            if mass_da >= 10000:
                label_text = f"{mass_da:.1f}"  # 15679.3
            elif mass_da >= 1000:
                label_text = f"{mass_da:.2f}"  # 1568.89
            else:
                label_text = f"{mass_da:.2f}"  # 500.50

            # Use matching label color for each bar
            label_color = label_colors[orig_idx % len(label_colors)]
            ax_deconv.annotate(label_text, (m_kda, intensity),
                             xytext=(label_x, label_y),
                             fontsize=6, ha='center', va='bottom',
                             color=label_color, fontweight='bold')

        ax_deconv.set_xlabel("Mass (kDa)")
        ax_deconv.set_ylabel("Relative Intensity (%)")
        ax_deconv.set_title("Deconvoluted Masses", fontweight='bold')

        if show_grid:
            ax_deconv.grid(True, alpha=0.3)
    else:
        ax_deconv.text(0.5, 0.5, "No masses detected",
                      ha='center', va='center', transform=ax_deconv.transAxes)
        ax_deconv.set_title("Deconvoluted Masses", fontweight='bold')

    plt.suptitle(f"Protein Deconvolution: {sample.name}", fontsize=10, fontweight='bold', y=1.02)

    return fig
