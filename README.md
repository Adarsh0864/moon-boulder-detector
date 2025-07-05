# Lunar GeoDetect 

An AI-powered web application for detecting boulders and landslides on the lunar surface using Chandrayaan mission imagery.

## Features

- **3D Lunar Terrain Visualization**: Interactive 3D rendering of lunar surface using Three.js
- **Boulder Detection**: AI-powered detection of rocky formations and debris
- **Landslide Detection**: Identification of surface displacement features
- **Real-time Analysis**: Adjustable detection sensitivity with instant results
- **Export Capabilities**: Export detection results in GeoJSON, CSV, and PNG formats
- **Statistics Dashboard**: Size distribution charts and detection confidence metrics

## Tech Stack

### Frontend
- React + Vite
- Tailwind CSS for styling
- Three.js (react-three-fiber) for 3D visualization
- Recharts for data visualization
- Axios for API communication

### Backend
- FastAPI (Python)
- OpenCV for computer vision
- NumPy for numerical operations
- Scikit-image for image processing
- PIL for image handling

## Installation & Setup

### Prerequisites
- Node.js (v16 or higher)
- Python 3.8 or higher
- npm or yarn

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd lunar-geodetect/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Backend Setup

1. Navigate to the backend directory:
```bash
cd lunar-geodetect/backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- On macOS/Linux: `source venv/bin/activate`
- On Windows: `venv\Scripts\activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Start the FastAPI server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## Dataset Setup

**Important**: The large dataset files (*.zip) are not included in this repository due to GitHub's file size limits. To run the application with the full dataset:

1. **Download the required dataset files**:
   - `ch2_tmc_ndn_20250217T0120183473_d_oth_d18.zip` (586 MB)
   - `ch2_ohr_ncp_20250304T0456267027_d_img_d18.zip` (240 MB)  
   - `ch2_ohr_ncp_20250304T0655103610_d_img_d18.zip` (376 MB)
   - `ch2_tmc_ndn_20250216T2322125310_d_dtm_d18.zip` (119 MB)
   - `ch2_tmc_ndn_20250217T0120183473_d_dtm_d18.zip` (119 MB)
   - `ch2_tmc_ndn_20250216T2322125310_d_oth_d18.zip` (588 MB)

2. **Place the dataset files** in the `data/` directory:
   ```
   lunar-geodetect/data/
   ├── ch2_tmc_ndn_20250217T0120183473_d_oth_d18.zip
   ├── ch2_ohr_ncp_20250304T0456267027_d_img_d18.zip
   ├── ch2_ohr_ncp_20250304T0655103610_d_img_d18.zip
   ├── ch2_tmc_ndn_20250216T2322125310_d_dtm_d18.zip
   ├── ch2_tmc_ndn_20250217T0120183473_d_dtm_d18.zip
   └── ch2_tmc_ndn_20250216T2322125310_d_oth_d18.zip
   ```

3. **Extract the files** (if needed by the application):
   ```bash
   cd lunar-geodetect/data
   unzip "*.zip"
   ```

**Note**: The application will work with sample data without these files, but full functionality requires the complete dataset.

## Usage

1. **Upload Image**: Click on the upload area in the left sidebar to select a lunar image (supports JPG, PNG, TIF formats)

2. **Select Detection Mode**: 
   - **Landslide Mode**: Detects surface displacement features
   - **Boulder Mode**: Detects rocky formations and debris

3. **Adjust Sensitivity**:
   - **Brightness Threshold**: Controls detection sensitivity to bright features
   - **Shape Size Filter**: Filters detections by size
   - **Shadow Detection**: Enhances detection using shadow analysis

4. **Run Detection**: Click "Run Detection" to process the image

5. **View Results**: 
   - 3D visualization shows detected features on the lunar terrain
   - Right sidebar displays detailed results with coordinates
   - Statistics tab shows size distribution and summary

6. **Export Results**: Click "Export Results" to download detection data

## API Endpoints

- `GET /` - Health check
- `POST /detect/boulders` - Boulder detection endpoint
- `POST /detect/landslides` - Landslide detection endpoint
- `POST /detect/all` - Combined detection (both boulders and landslides)
- `GET /formats` - Get supported image formats

## Project Structure

```
lunar-geodetect/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── LeftSidebar.jsx
│   │   │   ├── MainView.jsx
│   │   │   └── RightSidebar.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
├── backend/
│   ├── models/
│   │   ├── boulder_detector.py
│   │   └── landslide_detector.py
│   ├── main.py
│   └── requirements.txt
└── data/
    ├── boulders/
    ├── landslides/
    └── heightmaps/
```

## Detection Algorithms

### Boulder Detection
- Uses morphological operations and contour detection
- Analyzes circular features and shadows
- Confidence scoring based on circularity, size, and brightness

### Landslide Detection
- Edge detection using Canny algorithm
- Gradient analysis for slope identification
- Morphological operations to identify displaced regions
- Confidence scoring based on elongation and gradient magnitude

