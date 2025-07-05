import React, { Suspense, useRef, useMemo } from 'react'
import { Canvas, useLoader } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, Stats } from '@react-three/drei'
import * as THREE from 'three'
import { ZoomIn, ZoomOut, RotateCw, Maximize2 } from 'lucide-react'

// Lunar Terrain Component
function LunarTerrain({ selectedImage, detectionResults }) {
  const meshRef = useRef()
  
  // Generate terrain geometry based on selected image type
  const geometry = useMemo(() => {
    if (!selectedImage) return new THREE.PlaneGeometry(20, 20, 64, 64)
    
    const geo = new THREE.PlaneGeometry(20, 20, 128, 128)
    const positions = geo.attributes.position.array
    
    // Different terrain patterns for different datasets
    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i]
      const y = positions[i + 1]
      let z = 0
      
      switch (selectedImage.id) {
        case 'tmc_crater_field':
          // Multiple impact craters
          z = Math.sin(x * 0.2) * Math.cos(y * 0.2) * 0.3
          // Add multiple craters
          for (let j = 0; j < 5; j++) {
            const craterX = (j - 2) * 4
            const craterY = (j % 2 - 0.5) * 4
            const distance = Math.sqrt((x - craterX) ** 2 + (y - craterY) ** 2)
            if (distance < 3) {
              z -= (3 - distance) * 0.4 * (1 - j * 0.1)
            }
          }
          break
          
        case 'tmc_slope_region':
          // Steep slopes and landslide features
          z = x * 0.3 + Math.sin(y * 0.5) * 2
          // Add landslide scars
          if (x > 2 && y > -2 && y < 2) {
            z -= (x - 2) * 0.2
          }
          break
          
        case 'ohrc_boulder_field':
          // Boulder field with scattered rocks
          z = Math.sin(x * 0.1) * Math.cos(y * 0.1) * 0.2
          // Add boulder-like features
          const boulderNoise = Math.sin(x * 3) * Math.cos(y * 3) * 0.15
          z += boulderNoise
          break
          
        case 'tmc_south_pole':
          // Polar terrain with permanent shadows
          z = Math.sin(x * 0.4) * Math.cos(y * 0.4) * 0.6
          // Add complex crater rim features
          const rimDistance = Math.sqrt(x ** 2 + y ** 2)
          if (rimDistance > 8 && rimDistance < 10) {
            z += (rimDistance - 8) * 0.5
          }
          break
          
        default:
          z = Math.sin(x * 0.3) * Math.cos(y * 0.3) * 0.5
      }
      
      // Add fine-scale surface roughness
      z += (Math.random() - 0.5) * 0.08
      
      positions[i + 2] = z
    }
    
    geo.computeVertexNormals()
    return geo
  }, [selectedImage])

  // Create procedural lunar texture and material based on selected image
  const material = useMemo(() => {
    if (!selectedImage) {
      return new THREE.MeshStandardMaterial({
        color: "#444444",
        roughness: 0.95,
        metalness: 0.0
      })
    }

    // Create procedural textures for lunar surface
    const canvas = document.createElement('canvas')
    canvas.width = 512
    canvas.height = 512
    const ctx = canvas.getContext('2d')
    
    // Base lunar surface color
    let baseColor, detailColor, craterColor
    
    switch (selectedImage.id) {
      case 'tmc_crater_field':
        baseColor = '#6B6B6B'
        detailColor = '#555555'
        craterColor = '#3A3A3A'
        break
      case 'tmc_slope_region':
        baseColor = '#7A7A7A'
        detailColor = '#666666'
        craterColor = '#4A4A4A'
        break
      case 'ohrc_boulder_field':
        baseColor = '#888888'
        detailColor = '#777777'
        craterColor = '#555555'
        break
      case 'tmc_south_pole':
        baseColor = '#555555'
        detailColor = '#444444'
        craterColor = '#2A2A2A'
        break
      default:
        baseColor = '#777777'
        detailColor = '#666666'
        craterColor = '#444444'
    }

    // Fill base color
    ctx.fillStyle = baseColor
    ctx.fillRect(0, 0, 512, 512)
    
    // Add surface texture details
    for (let i = 0; i < 200; i++) {
      const x = Math.random() * 512
      const y = Math.random() * 512
      const size = Math.random() * 4 + 1
      
      ctx.fillStyle = Math.random() > 0.5 ? detailColor : craterColor
      ctx.beginPath()
      ctx.arc(x, y, size, 0, Math.PI * 2)
      ctx.fill()
    }
    
    // Add crater features specific to dataset
    if (selectedImage.id === 'tmc_crater_field') {
      for (let i = 0; i < 8; i++) {
        const x = Math.random() * 512
        const y = Math.random() * 512
        const size = Math.random() * 30 + 10
        
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, size)
        gradient.addColorStop(0, craterColor)
        gradient.addColorStop(0.7, detailColor)
        gradient.addColorStop(1, baseColor)
        
        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.arc(x, y, size, 0, Math.PI * 2)
        ctx.fill()
      }
    }
    
    // Add boulder texture for boulder field
    if (selectedImage.id === 'ohrc_boulder_field') {
      for (let i = 0; i < 50; i++) {
        const x = Math.random() * 512
        const y = Math.random() * 512
        const size = Math.random() * 3 + 1
        
        ctx.fillStyle = '#999999'
        ctx.beginPath()
        ctx.arc(x, y, size, 0, Math.PI * 2)
        ctx.fill()
        
        // Add shadow
        ctx.fillStyle = '#333333'
        ctx.beginPath()
        ctx.arc(x + 1, y + 1, size * 0.8, 0, Math.PI * 2)
        ctx.fill()
      }
    }

    // Create Three.js texture from canvas
    const texture = new THREE.CanvasTexture(canvas)
    texture.wrapS = THREE.RepeatWrapping
    texture.wrapT = THREE.RepeatWrapping
    texture.repeat.set(2, 2)

    // Create normal map for surface detail
    const normalCanvas = document.createElement('canvas')
    normalCanvas.width = 256
    normalCanvas.height = 256
    const normalCtx = normalCanvas.getContext('2d')
    
    // Generate noise for normal map
    const imageData = normalCtx.createImageData(256, 256)
    for (let i = 0; i < imageData.data.length; i += 4) {
      const noise = Math.random() * 50 + 100
      imageData.data[i] = noise     // R
      imageData.data[i + 1] = noise // G  
      imageData.data[i + 2] = 255   // B (normal Z)
      imageData.data[i + 3] = 255   // A
    }
    normalCtx.putImageData(imageData, 0, 0)
    
    const normalMap = new THREE.CanvasTexture(normalCanvas)
    normalMap.wrapS = THREE.RepeatWrapping
    normalMap.wrapT = THREE.RepeatWrapping
    normalMap.repeat.set(4, 4)

    return new THREE.MeshStandardMaterial({
      map: texture,
      normalMap: normalMap,
      normalScale: new THREE.Vector2(0.3, 0.3),
      roughness: 0.95,
      metalness: 0.02,
      color: '#FFFFFF'
    })
  }, [selectedImage])

  return (
    <mesh ref={meshRef} geometry={geometry} material={material} rotation={[-Math.PI / 2, 0, 0]}>
    </mesh>
  )
}

// Detection markers
function DetectionMarkers({ detectionResults, detectionMode }) {
  if (!detectionResults) return null

  const markers = detectionMode === 'boulder' ? detectionResults.boulders : detectionResults.landslides

  return (
    <>
      {markers?.map((marker, index) => (
        <mesh
          key={marker.id}
          position={[
            (Math.random() - 0.5) * 15,
            0.5,
            (Math.random() - 0.5) * 15
          ]}
        >
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshStandardMaterial 
            color={detectionMode === 'boulder' ? '#ff6b35' : '#ffd93d'}
            emissive={detectionMode === 'boulder' ? '#ff6b35' : '#ffd93d'}
            emissiveIntensity={0.5}
          />
        </mesh>
      ))}
    </>
  )
}

const MainView = ({ selectedImage, detectionResults, detectionMode, leftPanelExpanded, rightPanelExpanded }) => {
  const canvasRef = useRef()

  return (
    <main className="flex-1 relative bg-black transition-all duration-300">
      {!selectedImage ? (
        /* No Image Selected */
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className="w-24 h-24 mx-auto mb-6 bg-gray-800 rounded-full flex items-center justify-center">
              <img src="/api/placeholder/48/48" alt="Lunar Surface" className="w-12 h-12 opacity-50" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Select a Lunar Dataset</h3>
            <p className="text-gray-400 mb-6 max-w-md">
              Choose from pre-loaded Chandrayaan mission data to begin analysis.<br/>
              Each dataset contains high-resolution lunar surface imagery.
            </p>
            <div className="glass-effect rounded-lg p-4 max-w-md mx-auto">
              <div className="text-sm text-gray-300 mb-3">How to Use:</div>
              <div className="space-y-2 text-xs text-gray-400">
                <div className="flex items-start gap-2">
                  <span className="text-blue-400 font-bold">1.</span>
                  <span>Select a Chandrayaan dataset from the left panel</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-blue-400 font-bold">2.</span>
                  <span>Choose analysis mode (Comprehensive/Boulder/Landslide)</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-blue-400 font-bold">3.</span>
                  <span>Adjust detection sensitivity settings</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-blue-400 font-bold">4.</span>
                  <span>Click "Start Analysis" to run novel algorithms</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-blue-400 font-bold">5.</span>
                  <span>View results in 3D visualization and analysis panel</span>
                </div>
              </div>
              <div className="mt-4 pt-3 border-t border-gray-700">
                <div className="text-xs text-gray-500 mb-2">Available Datasets:</div>
                <div className="space-y-1 text-xs text-gray-500">
                  <div>• TMC Crater Field (5m/pixel)</div>
                  <div>• TMC Slope Region (5m/pixel)</div>
                  <div>• OHRC Boulder Field (0.3m/pixel)</div>
                  <div>• TMC South Pole (5m/pixel)</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* 3D Viewer with Selected Image */
        <div className="absolute inset-0">
          <Canvas ref={canvasRef} shadows>
            <PerspectiveCamera makeDefault position={[10, 10, 10]} />
            <OrbitControls enableDamping dampingFactor={0.05} />
            
            {/* Lunar lighting setup */}
            <ambientLight intensity={0.1} color="#1a1a2e" />
            <directionalLight 
              position={[20, 15, 10]} 
              intensity={2.5}
              color="#ffffff"
              castShadow
              shadow-mapSize={[4096, 4096]}
              shadow-camera-near={0.1}
              shadow-camera-far={100}
              shadow-camera-left={-20}
              shadow-camera-right={20}
              shadow-camera-top={20}
              shadow-camera-bottom={-20}
            />
            {/* Subtle fill light */}
            <pointLight position={[-15, 5, -10]} intensity={0.3} color="#404080" />
            
            {/* Space background */}
            <color attach="background" args={['#000000']} />
            
            <Suspense fallback={null}>
              <LunarTerrain detectionResults={detectionResults} selectedImage={selectedImage} />
              <DetectionMarkers 
                detectionResults={detectionResults} 
                detectionMode={detectionMode}
              />
            </Suspense>
            
            {/* Grid Helper */}
            <gridHelper args={[30, 30, '#333333', '#222222']} />
            
            {/* Performance Stats */}
            <Stats className="!left-auto !right-0" />
          </Canvas>
        </div>
      )}

      {/* Control Buttons */}
      <div className="absolute top-4 right-4 flex flex-col gap-2">
        <button className="p-2 bg-gray-800/80 hover:bg-gray-700/80 rounded-lg backdrop-blur">
          <ZoomIn className="w-5 h-5" />
        </button>
        <button className="p-2 bg-gray-800/80 hover:bg-gray-700/80 rounded-lg backdrop-blur">
          <ZoomOut className="w-5 h-5" />
        </button>
        <button className="p-2 bg-gray-800/80 hover:bg-gray-700/80 rounded-lg backdrop-blur">
          <RotateCw className="w-5 h-5" />
        </button>
        <button className="p-2 bg-gray-800/80 hover:bg-gray-700/80 rounded-lg backdrop-blur">
          <Maximize2 className="w-5 h-5" />
        </button>
      </div>

      {/* Info Overlay */}
      {selectedImage && (
        <div className="absolute bottom-4 left-4 text-xs text-gray-400 font-mono">
          <div className="bg-gray-900/80 backdrop-blur px-3 py-2 rounded-lg">
            {detectionMode === 'comprehensive' ? (
              <div className="flex items-center gap-2">
                <div className="flex -space-x-1">
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                </div>
                <span>Comprehensive Analysis Active</span>
              </div>
            ) : detectionMode === 'landslide' ? (
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <span>Landslide Detection Active</span>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                <span>Boulder Detection Active</span>
              </div>
            )}
            {detectionResults && (
              <div className="mt-1 space-y-0.5">
                {detectionResults.boulders?.length > 0 && (
                  <div>{detectionResults.boulders.length} boulders detected</div>
                )}
                {detectionResults.landslides?.length > 0 && (
                  <div>{detectionResults.landslides.length} landslides detected</div>
                )}
                {!detectionResults.boulders?.length && !detectionResults.landslides?.length && (
                  <div>Ready for analysis</div>
                )}
              </div>
            )}
            {selectedImage && !detectionResults && (
              <div className="mt-1 text-gray-500">
                {selectedImage.mission} • {selectedImage.resolution}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Coordinates Display */}
      {selectedImage && (
        <div className="absolute bottom-4 right-4 text-xs text-gray-400 font-mono">
          <div className="bg-gray-900/80 backdrop-blur px-3 py-2 rounded-lg">
            <div>{selectedImage.coordinates}</div>
            <div className="text-gray-500 text-xs mt-0.5">{selectedImage.name}</div>
          </div>
        </div>
      )}
    </main>
  )
}

export default MainView 