# LC-MS Analysis

Analyze LC-MS (Liquid Chromatography-Mass Spectrometry) data from Agilent .D folders.

## Features

- **UV-Vis Chromatograms**: Multi-wavelength detector (MWD) data at 194, 254, 280 nm
- **Mass Spectrometry**: Total Ion Chromatogram (TIC) and Extracted Ion Chromatograms (EIC)
- **Protein Deconvolution**: Calculate neutral mass from multiply-charged ion series
- **Time Progression**: Compare 2-3 samples with color-coded overlays
- **Export**: Download plots as PNG (300 DPI), SVG, or PDF

## Download Desktop App

Download the app for your platform - no installation required:

| Platform | Download |
|----------|----------|
| **macOS** | [LCMS-Analysis-macOS.zip](../../releases/latest/download/LCMS-Analysis-macOS.zip) |
| **Windows** | [LCMS-Analysis-Windows.zip](../../releases/latest/download/LCMS-Analysis-Windows.zip) |

### Installation

1. Download and extract the ZIP for your platform
2. **macOS**: Open Terminal, navigate to the folder, run `./LCMS-Analysis`
3. **Windows**: Double-click `LCMS-Analysis.exe`
4. Your browser will open with the app
5. Use **Browse Files** mode to select .D folders directly (no ZIP needed!)

## Web Version

Use the online version (requires ZIP upload): [Open Web App](https://lcms-webapp.streamlit.app)

## Run from Source

```bash
git clone https://github.com/dspelveris/lcms-webapp.git
cd lcms-webapp
pip install -r requirements.txt
streamlit run app/main.py
```

## Configuration

### Environment Variables

- `LCMS_BASE_PATH`: Root directory for LC-MS data (default: `/data/lcms`)

### Default Settings

- UV Wavelength: 194 nm
- UV Smoothing: 36
- EIC Smoothing: 5
- m/z Window: ±0.5
- Default m/z values: 680.1, 382.98, 243.99, 321.15
- Export DPI: 300

## Usage

1. **Select Files**: Use the sidebar file browser to navigate to your LC-MS data
2. **Choose Analysis Mode**:
   - Single sample: View UV, TIC, and EICs for one file
   - Time progression: Compare 2-3 samples across timepoints
   - EIC batch: Extract multiple EICs with peak areas
3. **Adjust Settings**: Modify UV wavelength, smoothing, and m/z window in sidebar
4. **Configure m/z Targets**: Add/remove target m/z values for EIC extraction
5. **Export**: Download plots as PNG files

## Time Progression Colors

- Grey: Initial (t=0)
- Blue: Mid timepoint
- Red: Final/Overnight

## Troubleshooting

### Network drive not accessible

Make sure the network drive is mounted before starting Docker:
```bash
# macOS
mount -t smbfs //server/share /Volumes/chab_loc_lang_s1
```

Then update `docker-compose.yml` with the correct path.

### rainbow-api errors

The app uses [rainbow-api](https://github.com/evanyeyeye/rainbow) for reading Agilent data. If you encounter issues:

```bash
pip install --upgrade rainbow-api
```
