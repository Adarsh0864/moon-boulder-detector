import React from 'react'

const MainView = ({ selectedImage, detectionResults, detectionMode, leftPanelExpanded, rightPanelExpanded }) => {
  return (
    <main className="flex-1 relative bg-black transition-all duration-300">
      {!selectedImage ? (
        /* No Image Selected */
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className="w-24 h-24 mx-auto mb-6 bg-gray-800 rounded-full flex items-center justify-center">
              <div className="w-12 h-12 bg-gray-600 rounded"></div>
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Select a Lunar Dataset</h3>
            <p className="text-gray-400 mb-6 max-w-md">
              Choose from pre-loaded Chandrayaan mission data to begin analysis.<br/>
              Each dataset contains high-resolution lunar surface imagery.
            </p>
          </div>
        </div>
      ) : (
        /* Dataset Selected - Simplified View */
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <h3 className="text-xl font-semibold text-white mb-2">3D Visualization</h3>
            <p className="text-gray-400 mb-4">Dataset: {selectedImage.name}</p>
            <div className="w-96 h-96 bg-gray-800 rounded-lg flex items-center justify-center">
              <div className="text-gray-500">3D Viewer will render here</div>
            </div>
            {detectionResults && (
              <div className="mt-4 text-green-400">
                Analysis Complete: {detectionResults.boulders?.length || 0} boulders, {detectionResults.landslides?.length || 0} landslides
              </div>
            )}
          </div>
        </div>
      )}
    </main>
  )
}

export default MainView