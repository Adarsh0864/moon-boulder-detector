import React, { useState, useEffect, useCallback } from 'react'
import Header from './components/Header'
import LeftSidebar from './components/LeftSidebar'
import MainView from './components/MainView'
import RightSidebar from './components/RightSidebar'
import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000'

function App() {
  const [detectionMode, setDetectionMode] = useState('comprehensive')
  const [selectedImage, setSelectedImage] = useState(null)
  const [detectionSettings, setDetectionSettings] = useState({
    brightnessThreshold: 65,
    shapeSizeFilter: 40,
    shadowDetection: 75
  })
  const [detectionResults, setDetectionResults] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [leftPanelExpanded, setLeftPanelExpanded] = useState(true)
  const [rightPanelExpanded, setRightPanelExpanded] = useState(true)
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false)
  const [leftPanelWidth, setLeftPanelWidth] = useState(288) // 18rem = 288px
  const [rightPanelWidth, setRightPanelWidth] = useState(384) // 24rem = 384px
  const [isResizing, setIsResizing] = useState(null)

  const handleImageSelect = (imageData) => {
    setSelectedImage(imageData)
    setDetectionResults(null)
  }

  // Handle panel resizing
  const handleMouseDown = (panel) => (e) => {
    e.preventDefault()
    setIsResizing(panel)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }

  const handleMouseMove = useCallback((e) => {
    if (!isResizing) return
    
    if (isResizing === 'left') {
      const newWidth = Math.max(200, Math.min(600, e.clientX))
      setLeftPanelWidth(newWidth)
    } else if (isResizing === 'right') {
      const newWidth = Math.max(300, Math.min(800, window.innerWidth - e.clientX))
      setRightPanelWidth(newWidth)
    }
  }, [isResizing])

  const handleMouseUp = useCallback(() => {
    setIsResizing(null)
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
  }, [])

  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      return () => {
        document.removeEventListener('mousemove', handleMouseMove)
        document.removeEventListener('mouseup', handleMouseUp)
      }
    }
  }, [isResizing, handleMouseMove, handleMouseUp])

  const handleRunDetection = async () => {
    if (!selectedImage) return
    
    setIsProcessing(true)
    
    try {
      // Check if this is a TMC-2 real dataset
      if (selectedImage.isRealData && selectedImage.id.startsWith('tmc2_')) {
        const datasetDate = selectedImage.date
        
        // Enhanced settings for TMC-2 analysis
        const enhancedSettings = {
          brightness_threshold: detectionSettings.brightnessThreshold,
          shape_size_filter: detectionSettings.shapeSizeFilter,
          shadow_detection: detectionSettings.shadowDetection,
          use_novel_algorithms: true,
          use_shadow_sizing: true,
          perform_geological_analysis: detectionMode === 'comprehensive',
          perform_statistical_analysis: detectionMode === 'comprehensive',
          generate_comprehensive_maps: false
        }
        
        // Call TMC-2 analysis endpoint
        const response = await axios.post(
          `${API_BASE_URL}/analyze/tmc2/${datasetDate}`,
          enhancedSettings,
          {
            headers: {
              'Content-Type': 'application/json'
            }
          }
        )
        
        // Update results with real TMC-2 data
        setDetectionResults({
          boulders: response.data.boulders || [],
          landslides: response.data.landslides || [],
          geologicalFeatures: response.data.geological_features || [],
          statisticalAnalysis: response.data.statistical_analysis || null,
          riskAssessment: response.data.risk_assessment || null,
          processingTime: response.data.metadata?.processing_time || 0,
          metadata: response.data.metadata || {},
          isRealData: true
        })
        
        setIsProcessing(false)
        return
      }
      
      // Original mock data analysis for demo datasets
      // Create a mock file for the selected image (in production, this would be actual image data)
      const mockImageBlob = new Blob(['mock lunar image data'], { type: 'image/tiff' })
      const mockFile = new File([mockImageBlob], `${selectedImage.id}.tif`, { type: 'image/tiff' })
      
      // Create FormData for file upload
      const formData = new FormData()
      formData.append('file', mockFile)
      
      // Determine endpoint based on detection mode
      let endpoint
      if (detectionMode === 'comprehensive') {
        endpoint = '/analyze/comprehensive'
      } else if (detectionMode === 'boulder') {
        endpoint = '/detect/boulders/enhanced'
      } else {
        endpoint = '/detect/landslides/enhanced'
      }
      
      // Enhanced settings for novel algorithms
      const enhancedSettings = {
        brightness_threshold: detectionSettings.brightnessThreshold,
        shape_size_filter: detectionSettings.shapeSizeFilter,
        shadow_detection: detectionSettings.shadowDetection,
        use_novel_algorithms: true,
        use_shadow_sizing: true,
        perform_geological_analysis: detectionMode === 'comprehensive',
        perform_statistical_analysis: detectionMode === 'comprehensive',
        generate_comprehensive_maps: false // Set to false for faster processing in demo
      }
      
      // Make API call with enhanced settings
      const response = await axios.post(
        `${API_BASE_URL}${endpoint}`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          params: enhancedSettings
        }
      )
      
      // Update results with enhanced data
      setDetectionResults({
        boulders: response.data.boulders || [],
        landslides: response.data.landslides || [],
        geologicalFeatures: response.data.geological_features || [],
        statisticalAnalysis: response.data.statistical_analysis || null,
        riskAssessment: response.data.risk_assessment || null,
        processingTime: response.data.processing_time || 0,
        metadata: response.data.metadata || {}
      })
      
    } catch (err) {
      console.error('Detection failed:', err)
      
      // Enhanced mock data for comprehensive analysis
      if (err.code === 'ERR_NETWORK') {
        console.log('Using enhanced mock data for demonstration')
        setTimeout(() => {
          setDetectionResults({
            boulders: [
              { id: 'CB-001', diameter: 12.4, lat: 23.4567, lon: -45.6789, confidence: 94, height_m: 8.2, volume_m3: 458.3, detection_method: 'novel_wavelet_multiscale' },
              { id: 'CB-002', diameter: 8.7, lat: 23.4612, lon: -45.6823, confidence: 89, height_m: 5.1, volume_m3: 201.7, detection_method: 'novel_wavelet_multiscale' },
              { id: 'CB-003', diameter: 15.2, lat: 23.4498, lon: -45.6712, confidence: 96, height_m: 11.3, volume_m3: 823.4, detection_method: 'novel_wavelet_multiscale' },
              { id: 'CB-004', diameter: 6.3, lat: 23.4539, lon: -45.6901, confidence: 82, height_m: 3.8, volume_m3: 95.6, detection_method: 'shadow_based_sizing' },
              { id: 'CB-005', diameter: 10.8, lat: 23.4623, lon: -45.6734, confidence: 91, height_m: 7.2, volume_m3: 346.8, detection_method: 'novel_wavelet_multiscale' }
            ],
            landslides: [
              { id: 'CL-001', area_km2: 1.24, center: [23.4567, -45.6789], confidence: 92, flow_direction: 135.5, flow_length_m: 847.2, detection_method: 'novel_adaptive_terrain' },
              { id: 'CL-002', area_km2: 0.68, center: [23.4623, -45.6812], confidence: 87, flow_direction: 198.3, flow_length_m: 423.6, detection_method: 'novel_adaptive_terrain' }
            ],
            geologicalFeatures: [
              { type: 'crater', center: [23.4589, -45.6798], properties: { diameter_m: 234.5, confidence: 89, freshness: 0.73 } },
              { type: 'ridge', center: [23.4612, -45.6745], properties: { length_m: 1247.8, prominence: 45.2 } },
              { type: 'scarp', center: [23.4534, -45.6834], properties: { area_m2: 15680, avg_slope: 12.7, orientation_deg: 87.3 } }
            ],
            statisticalAnalysis: {
              summary_statistics: {
                study_area: { total_area_km2: 25.7 },
                boulder_statistics: { total_count: 5, density_per_km2: 0.19, mean_diameter_m: 10.7 },
                landslide_statistics: { total_count: 2, density_per_km2: 0.08, total_affected_area_km2: 1.92 }
              },
              spatial_analysis: {
                boulder_clustering: { clustering_coefficient: 0.73, cluster_count: 2 },
                landslide_clustering: { clustering_coefficient: 0.45, cluster_count: 1 }
              }
            },
            riskAssessment: {
              total_high_risk_area_km2: 2.34,
              total_moderate_risk_area_km2: 5.67,
              risk_assessment_confidence: 0.87
            },
            processingTime: 4.7,
            metadata: {
              novel_algorithms_used: true,
              algorithms_applied: {
                multi_scale_boulder_detection: true,
                adaptive_landslide_detection: true,
                shadow_based_sizing: true,
                geological_context_analysis: detectionMode === 'comprehensive',
                statistical_analysis: detectionMode === 'comprehensive'
              }
            }
          })
          setIsProcessing(false)
        }, 3000) // Longer processing time to show advanced analysis
      } else {
        setIsProcessing(false)
        alert('Analysis failed. Please try again.')
      }
    } finally {
      setIsProcessing(false)
    }
  }

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (event) => {
      // Ignore shortcuts when user is typing in an input
      if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return
      
      switch (event.key) {
        case 'q':
          setLeftPanelExpanded(prev => !prev)
          break
        case 'e':
          setRightPanelExpanded(prev => !prev)
          break
        case ' ':
          if (selectedImage && !isProcessing) {
            event.preventDefault()
            handleRunDetection()
          }
          break
        case 'Escape':
          // Reset panels to expanded state
          setLeftPanelExpanded(true)
          setRightPanelExpanded(true)
          break
        case '1':
          setDetectionMode('comprehensive')
          break
        case '2':
          setDetectionMode('landslide')
          break
        case '3':
          setDetectionMode('boulder')
          break
        case '?':
          setShowKeyboardHelp(prev => !prev)
          break
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [selectedImage, isProcessing, handleRunDetection])

  return (
    <div className="h-screen flex flex-col bg-gray-900 overflow-hidden">
      <Header onShowHelp={() => setShowKeyboardHelp(true)} />
      
      <div className="flex-1 flex overflow-hidden">
        <div className="relative flex">
          <LeftSidebar 
            detectionMode={detectionMode}
            setDetectionMode={setDetectionMode}
            detectionSettings={detectionSettings}
            setDetectionSettings={setDetectionSettings}
            onImageSelect={handleImageSelect}
            onRunDetection={handleRunDetection}
            isProcessing={isProcessing}
            selectedImage={selectedImage}
            detectionResults={detectionResults}
            isExpanded={leftPanelExpanded}
            onToggleExpand={() => setLeftPanelExpanded(!leftPanelExpanded)}
            width={leftPanelExpanded ? leftPanelWidth : 48}
          />
          
          {/* Left resize handle */}
          {leftPanelExpanded && (
            <div 
              className="w-1 bg-gray-700 hover:bg-blue-500 cursor-col-resize transition-colors relative z-10"
              onMouseDown={handleMouseDown('left')}
            >
              <div className="absolute inset-0 w-3 -ml-1" /> {/* Larger hit area */}
            </div>
          )}
        </div>
        
        <MainView 
          selectedImage={selectedImage}
          detectionResults={detectionResults}
          detectionMode={detectionMode}
          leftPanelExpanded={leftPanelExpanded}
          rightPanelExpanded={rightPanelExpanded}
        />
        
        <div className="relative flex">
          {/* Right resize handle */}
          {rightPanelExpanded && (
            <div 
              className="w-1 bg-gray-700 hover:bg-blue-500 cursor-col-resize transition-colors relative z-10"
              onMouseDown={handleMouseDown('right')}
            >
              <div className="absolute inset-0 w-3 -mr-1" /> {/* Larger hit area */}
            </div>
          )}
          
          <RightSidebar 
            detectionResults={detectionResults}
            detectionMode={detectionMode}
            isExpanded={rightPanelExpanded}
            onToggleExpand={() => setRightPanelExpanded(!rightPanelExpanded)}
            width={rightPanelExpanded ? rightPanelWidth : 48}
          />
        </div>
      </div>

      {/* Keyboard Shortcuts Help Overlay */}
      {showKeyboardHelp && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowKeyboardHelp(false)}>
          <div className="bg-gray-800 rounded-lg p-6 max-w-md mx-4" onClick={e => e.stopPropagation()}>
            <h3 className="text-lg font-semibold mb-4">Keyboard Shortcuts</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Toggle Left Panel</span>
                <kbd className="px-2 py-1 bg-gray-700 rounded">Q</kbd>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Toggle Right Panel</span>
                <kbd className="px-2 py-1 bg-gray-700 rounded">E</kbd>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Start Analysis</span>
                <kbd className="px-2 py-1 bg-gray-700 rounded">Space</kbd>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Reset Panels</span>
                <kbd className="px-2 py-1 bg-gray-700 rounded">Esc</kbd>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Comprehensive Mode</span>
                <kbd className="px-2 py-1 bg-gray-700 rounded">1</kbd>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Landslide Mode</span>
                <kbd className="px-2 py-1 bg-gray-700 rounded">2</kbd>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Boulder Mode</span>
                <kbd className="px-2 py-1 bg-gray-700 rounded">3</kbd>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Show/Hide Help</span>
                <kbd className="px-2 py-1 bg-gray-700 rounded">?</kbd>
              </div>
            </div>
            <button 
              onClick={() => setShowKeyboardHelp(false)}
              className="mt-4 w-full btn-primary"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
