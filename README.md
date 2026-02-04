# LC-MS Analysis Web App

A Dockerized Streamlit web application for LC-MS data analysis with multi-file selection, time progression analysis, and EIC extraction capabilities.

## Features

- **File Browser**: Browse network drive for Agilent `.D` folders
- **Single Sample Analysis**: View UV chromatogram, MS TIC, and interactive EIC extraction
- **Time Progression Comparison**: Compare 2-3 samples with color-coded overlays
- **EIC Batch Extraction**: Extract multiple EICs with peak area calculation
- **PNG Export**: Download plots at 200 DPI

## Quick Start

### Prerequisites

- Docker and docker-compose installed
- Network drive mounted at `/Volumes/chab_loc_lang_s1` (or adjust in docker-compose.yml)

### Run the Application

```bash
cd lcms-webapp
docker-compose up --build
```

Open http://localhost:8501 in your browser.

### Development Mode (without Docker)

```bash
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
- Export DPI: 200

## Project Structure

```
lcms-webapp/
├── docker-compose.yml      # Docker compose configuration
├── Dockerfile              # Docker image definition
├── requirements.txt        # Python dependencies
├── app/
│   ├── main.py            # Streamlit entry point
│   ├── data_reader.py     # Rainbow .D folder reader
│   ├── analysis.py        # EIC extraction, smoothing
│   ├── plotting.py        # Plot generation functions
│   └── config.py          # Default values, paths
└── README.md
```

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
