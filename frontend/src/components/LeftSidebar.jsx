import React, { useState, useEffect } from 'react'
import { Play, Download, Mountain, Circle, Image as ImageIcon, Satellite, MapPin, ChevronLeft, ChevronRight } from 'lucide-react'

const LeftSidebar = ({ 
  detectionMode, 
  setDetectionMode, 
  detectionSettings, 
  setDetectionSettings,
  onImageSelect,
  onRunDetection,
  isProcessing,
  selectedImage,
  detectionResults,
  isExpanded,
  onToggleExpand,
  width = 288
}) => {
  const [tmc2Datasets, setTmc2Datasets] = useState([])
  const [loadingTmc2, setLoadingTmc2] = useState(false)

  // Fetch TMC-2 datasets on component mount
  useEffect(() => {
    const fetchTmc2Datasets = async () => {
      setLoadingTmc2(true)
      try {
        const response = await fetch('http://localhost:8000/datasets/tmc2')
        if (response.ok) {
          const data = await response.json()
          const formattedDatasets = Object.values(data.datasets).map(dataset => ({
            id: dataset.id,
            name: dataset.name,
            description: dataset.description,
            mission: dataset.mission,
            coordinates: 'TMC-2 Real Data',
            resolution: dataset.resolution,
            thumbnail: '/api/placeholder/120/80',
            features: dataset.features,
            date: dataset.date,
            isRealData: true,
            hasOrtho: dataset.has_ortho,
            hasDtm: dataset.has_dtm
          }))
          setTmc2Datasets(formattedDatasets)
        } else {
          console.error('TMC-2 fetch failed:', response.status, response.statusText)
        }
      } catch (error) {
        console.error('Failed to fetch TMC-2 datasets:', error)
      } finally {
        setLoadingTmc2(false)
      }
    }

    fetchTmc2Datasets()
  }, [])

  // Pre-loaded lunar images from Chandrayaan mission (mock data for demo)
  const lunarImages = [
    {
      id: 'tmc_crater_field',
      name: 'TMC Crater Field',
      description: 'Multiple impact craters with potential boulder fields',
      mission: 'Chandrayaan-1 TMC',
      coordinates: '23.45°S, 45.67°W',
      resolution: '5m/pixel',
      thumbnail: '/api/placeholder/120/80',
      features: ['Craters', 'Boulders', 'Ejecta']
    },
    {
      id: 'tmc_slope_region',
      name: 'TMC Slope Region',
      description: 'Steep lunar terrain with landslide activity',
      mission: 'Chandrayaan-1 TMC', 
      coordinates: '12.34°N, 67.89°E',
      resolution: '5m/pixel',
      thumbnail: '/api/placeholder/120/80',
      features: ['Slopes', 'Landslides', 'Scarps']
    },
    {
      id: 'ohrc_boulder_field',
      name: 'OHRC Boulder Field',
      description: 'High-resolution view of lunar boulder distribution',
      mission: 'Chandrayaan-2 OHRC',
      coordinates: '45.12°S, 123.45°E', 
      resolution: '0.3m/pixel',
      thumbnail: '/api/placeholder/120/80',
      features: ['Boulders', 'Shadows', 'Regolith']
    },
    {
      id: 'tmc_south_pole',
      name: 'TMC South Pole',
      description: 'Polar region with complex terrain features',
      mission: 'Chandrayaan-1 TMC',
      coordinates: '89.5°S, 0°E',
      resolution: '5m/pixel', 
      thumbnail: '/api/placeholder/120/80',
      features: ['Craters', 'Permanent Shadow', 'Ice Deposits']
    }
  ]

  const handleImageSelect = (imageData) => {
    onImageSelect(imageData)
  }

  const handleSliderChange = (setting, value) => {
    setDetectionSettings(prev => ({
      ...prev,
      [setting]: value
    }))
  }

  const handleExportResults = () => {
    if (!detectionResults || (!detectionResults.boulders?.length && !detectionResults.landslides?.length)) {
      alert('No detection results to export')
      return
    }

    // Create GeoJSON format
    const geoJson = {
      type: 'FeatureCollection',
      features: []
    }

    // Add boulders to GeoJSON
    detectionResults.boulders?.forEach(boulder => {
      geoJson.features.push({
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: [boulder.lon, boulder.lat]
        },
        properties: {
          id: boulder.id,
          type: 'boulder',
          diameter: boulder.diameter,
          confidence: boulder.confidence
        }
      })
    })

    // Add landslides to GeoJSON
    detectionResults.landslides?.forEach(landslide => {
      geoJson.features.push({
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: [landslide.center[1], landslide.center[0]]
        },
        properties: {
          id: landslide.id,
          type: 'landslide',
          area_km2: landslide.area_km2,
          confidence: landslide.confidence
        }
      })
    })

    // Download GeoJSON file
    const dataStr = JSON.stringify(geoJson, null, 2)
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr)
    
    const exportFileDefaultName = `lunar_detection_${new Date().toISOString().slice(0,10)}.geojson`
    
    const linkElement = document.createElement('a')
    linkElement.setAttribute('href', dataUri)
    linkElement.setAttribute('download', exportFileDefaultName)
    linkElement.click()
  }

  return (
    <aside 
      className="bg-gray-800 border-r border-gray-700 transition-all duration-300 relative overflow-hidden"
      style={{ width: `${width}px` }}
    >
      {/* Toggle Button */}
      <button
        onClick={onToggleExpand}
        className="absolute top-4 right-2 z-10 p-1 rounded-md bg-gray-700 hover:bg-gray-600 transition-colors"
        title={isExpanded ? 'Collapse Panel' : 'Expand Panel'}
      >
        {isExpanded ? (
          <ChevronLeft className="w-4 h-4 text-gray-300" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-300" />
        )}
      </button>

      {/* Collapsed State */}
      {!isExpanded && (
        <div className="p-2 pt-12 flex flex-col items-center space-y-4">
          <div className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 cursor-pointer transition-colors" 
               onClick={onToggleExpand}
               title="Chandrayaan Data">
            <Satellite className="w-6 h-6 text-blue-400" />
          </div>
          <div className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 cursor-pointer transition-colors"
               onClick={onToggleExpand}
               title="Analysis Mode">
            <Circle className="w-6 h-6 text-orange-400" />
          </div>
          <div className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 cursor-pointer transition-colors"
               onClick={onToggleExpand}
               title="Detection Settings">
            <Mountain className="w-6 h-6 text-yellow-400" />
          </div>
          {selectedImage && (
            <button
              onClick={onRunDetection}
              disabled={!selectedImage || isProcessing}
              className="p-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              title={isProcessing ? 'Analyzing...' : 'Start Analysis'}
            >
              <Play className="w-6 h-6 text-white" />
            </button>
          )}
        </div>
      )}

      {/* Expanded State */}
      {isExpanded && (
        <div className="p-6 pt-12 overflow-y-auto scrollbar-thin h-full">
      {/* Chandrayaan Image Selection */}
      <div className="mb-8">
        <h3 className="text-sm font-medium text-gray-400 mb-4 flex items-center gap-2">
          <Satellite className="w-4 h-4" />
          CHANDRAYAAN DATA
        </h3>
        
        <div className="space-y-3 max-h-80 overflow-y-auto scrollbar-thin">
          {/* Debug info */}
          {process.env.NODE_ENV === 'development' && (
            <div className="text-xs text-yellow-400 p-2 bg-yellow-900/20 rounded">
              TMC-2 Datasets: {tmc2Datasets.length} | Loading: {loadingTmc2.toString()}
            </div>
          )}
          
          {/* TMC-2 Real Datasets */}
          {tmc2Datasets.map((dataset) => (
            <div
              key={dataset.id}
              onClick={() => handleImageSelect(dataset)}
              className={`glass-effect rounded-lg p-4 cursor-pointer transition-all duration-200 border-2 ${
                selectedImage?.id === dataset.id 
                  ? 'border-green-500 bg-green-500/10' 
                  : 'border-transparent hover:border-gray-600'
              }`}
            >
              <div className="flex gap-3">
                <div className="w-16 h-16 bg-green-700 rounded-lg flex items-center justify-center flex-shrink-0">
                  <Satellite className="w-8 h-8 text-green-400" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <h4 className="text-sm font-medium text-white truncate">{dataset.name}</h4>
                    <span className="px-1 py-0.5 bg-green-600/20 text-green-400 text-xs rounded">REAL</span>
                  </div>
                  <p className="text-xs text-gray-400 mt-1 leading-tight">{dataset.description}</p>
                  <div className="flex items-center gap-2 mt-2">
                    <span className="text-xs px-2 py-0.5 bg-green-600/20 text-green-400 rounded">
                      {dataset.mission}
                    </span>
                  </div>
                  <div className="flex items-center gap-1 mt-1">
                    <MapPin className="w-3 h-3 text-gray-500" />
                    <span className="text-xs text-gray-500">{dataset.coordinates}</span>
                  </div>
                  <div className="flex flex-wrap gap-1 mt-2">
                    {dataset.features.map((feature, idx) => (
                      <span key={idx} className="text-xs px-1.5 py-0.5 bg-green-700 text-green-300 rounded">
                        {feature}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              {selectedImage?.id === dataset.id && (
                <div className="mt-3 flex items-center gap-2 text-green-400">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  <span className="text-xs">Selected for Real Analysis</span>
                </div>
              )}
            </div>
          ))}
          
          {loadingTmc2 && (
            <div className="glass-effect rounded-lg p-4 text-center text-gray-400">
              <div className="animate-spin w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
              <span className="text-xs">Loading TMC-2 datasets...</span>
            </div>
          )}
          
          {/* Demo/Mock Datasets */}
          <div className="border-t border-gray-700 pt-3 mt-3">
            <h4 className="text-xs font-medium text-gray-500 mb-3 flex items-center gap-2">
              <span>DEMO DATASETS (Simulated Data for Testing)</span>
              <div className="px-2 py-1 bg-blue-600/20 text-blue-400 text-xs rounded">MOCK</div>
            </h4>
          </div>
          
          {lunarImages.map((image) => (
            <div
              key={image.id}
              onClick={() => handleImageSelect(image)}
              className={`glass-effect rounded-lg p-4 cursor-pointer transition-all duration-200 border-2 ${
                selectedImage?.id === image.id 
                  ? 'border-blue-500 bg-blue-500/10' 
                  : 'border-transparent hover:border-gray-600'
              }`}
            >
              <div className="flex gap-3">
                <div className="w-16 h-16 bg-gray-700 rounded-lg flex items-center justify-center flex-shrink-0">
                  <ImageIcon className="w-8 h-8 text-gray-400" />
                </div>
                <div className="flex-1 min-w-0">
                  <h4 className="text-sm font-medium text-white truncate">{image.name}</h4>
                  <p className="text-xs text-gray-400 mt-1 leading-tight">{image.description}</p>
                  <div className="flex items-center gap-2 mt-2">
                    <span className="text-xs px-2 py-0.5 bg-blue-600/20 text-blue-400 rounded">
                      {image.mission}
                    </span>
                  </div>
                  <div className="flex items-center gap-1 mt-1">
                    <MapPin className="w-3 h-3 text-gray-500" />
                    <span className="text-xs text-gray-500">{image.coordinates}</span>
                  </div>
                  <div className="flex flex-wrap gap-1 mt-2">
                    {image.features.map((feature, idx) => (
                      <span key={idx} className="text-xs px-1.5 py-0.5 bg-gray-700 text-gray-300 rounded">
                        {feature}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              {selectedImage?.id === image.id && (
                <div className="mt-3 flex items-center gap-2 text-green-400">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  <span className="text-xs">Selected for Analysis</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Algorithm Selection */}
      <div className="mb-8">
        <h3 className="text-sm font-medium text-gray-400 mb-4 flex items-center gap-2">
          <Circle className="w-4 h-4" />
          ANALYSIS MODE
        </h3>
        
        <div className="space-y-3">
          <label className="flex items-center gap-3 p-3 rounded-lg cursor-pointer hover:bg-gray-800/50 transition-colors">
            <input
              type="radio"
              name="detectionMode"
              value="comprehensive"
              checked={detectionMode === 'comprehensive'}
              onChange={(e) => setDetectionMode(e.target.value)}
              className="w-4 h-4 text-blue-500"
            />
            <div className="flex flex-col">
              <div className="flex items-center gap-2">
                <div className="flex -space-x-1">
                  <Mountain className="w-4 h-4 text-yellow-500" />
                  <Circle className="w-4 h-4 text-orange-500" />
                </div>
                <span className="text-sm font-medium">Comprehensive Analysis</span>
              </div>
              <span className="text-xs text-gray-400 ml-6">Novel algorithms + Statistical analysis</span>
            </div>
          </label>
          
          <label className="flex items-center gap-3 p-3 rounded-lg cursor-pointer hover:bg-gray-800/50 transition-colors">
            <input
              type="radio"
              name="detectionMode"
              value="landslide"
              checked={detectionMode === 'landslide'}
              onChange={(e) => setDetectionMode(e.target.value)}
              className="w-4 h-4 text-blue-500"
            />
            <div className="flex flex-col">
              <div className="flex items-center gap-2">
                <Mountain className="w-5 h-5 text-yellow-500" />
                <span className="text-sm">Landslide Detection</span>
              </div>
              <span className="text-xs text-gray-400 ml-7">Adaptive terrain analysis</span>
            </div>
          </label>
          
          <label className="flex items-center gap-3 p-3 rounded-lg cursor-pointer hover:bg-gray-800/50 transition-colors">
            <input
              type="radio"
              name="detectionMode"
              value="boulder"
              checked={detectionMode === 'boulder'}
              onChange={(e) => setDetectionMode(e.target.value)}
              className="w-4 h-4 text-blue-500"
            />
            <div className="flex flex-col">
              <div className="flex items-center gap-2">
                <Circle className="w-5 h-5 text-orange-500" />
                <span className="text-sm">Boulder Detection</span>
              </div>
              <span className="text-xs text-gray-400 ml-7">Multi-scale wavelet analysis</span>
            </div>
          </label>
        </div>
      </div>

      {/* Detection Sensitivity */}
      <div className="mb-8">
        <h3 className="text-sm font-medium text-gray-400 mb-4">
          DETECTION SENSITIVITY
        </h3>
        
        <div className="space-y-6">
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span>Brightness Threshold</span>
              <span className="text-gray-400">{detectionSettings.brightnessThreshold}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={detectionSettings.brightnessThreshold}
              onChange={(e) => handleSliderChange('brightnessThreshold', e.target.value)}
              className="slider-track"
            />
          </div>
          
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span>Shape Size Filter</span>
              <span className="text-gray-400">{detectionSettings.shapeSizeFilter}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={detectionSettings.shapeSizeFilter}
              onChange={(e) => handleSliderChange('shapeSizeFilter', e.target.value)}
              className="slider-track"
            />
          </div>
          
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span>Shadow Detection</span>
              <span className="text-gray-400">{detectionSettings.shadowDetection}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={detectionSettings.shadowDetection}
              onChange={(e) => handleSliderChange('shadowDetection', e.target.value)}
              className="slider-track"
            />
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="space-y-3">
        <button
          onClick={onRunDetection}
          disabled={!selectedImage || isProcessing}
          className={`w-full btn-primary ${(!selectedImage || isProcessing) ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <Play className="w-4 h-4" />
          {isProcessing ? 'Analyzing...' : 'Start Analysis'}
        </button>
        
        {selectedImage && (
          <div className="glass-effect rounded-lg p-3 text-center">
            <div className="text-xs text-gray-400 mb-1">Selected Dataset</div>
            <div className="text-sm font-medium text-white">{selectedImage.name}</div>
            <div className="text-xs text-gray-500">{selectedImage.mission}</div>
          </div>
        )}
        
        <button 
          onClick={handleExportResults}
          disabled={!detectionResults || (!detectionResults.boulders?.length && !detectionResults.landslides?.length)}
          className={`w-full btn-secondary ${(!detectionResults || (!detectionResults.boulders?.length && !detectionResults.landslides?.length)) ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <Download className="w-4 h-4" />
          Export Results
        </button>
      </div>
        </div>
      )}
    </aside>
  )
}

export default LeftSidebar 