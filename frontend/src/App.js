import React, { useState, useEffect } from 'react';
// Add the ModelViewer import
import ModelViewer from './components/ModelViewer';

function App() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [serverStatus, setServerStatus] = useState(null);
  const [output, setOutput] = useState([]);
  const [parameters, setParameters] = useState(null);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [numIterations, setNumIterations] = useState(100);
  // Add new state variables for image upload
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [targetImages, setTargetImages] = useState([]);
  const [uploadStatus, setUploadStatus] = useState('');
  // Add new state variables for model upload
  const [selectedModelFile, setSelectedModelFile] = useState(null);
  const [currentModel, setCurrentModel] = useState(null);
  const [modelUploadStatus, setModelUploadStatus] = useState('');

  // Function to start optimization using Node.js backend
  const startOptimization = async () => {
    try {
      setError(null);
      setOutput([]);
      setParameters(null);
      setCurrentIteration(0);

      const response = await fetch('http://localhost:3001/api/run-optimization', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          method: 'cuda_ad_rgb',
          // Use the current model path if available, otherwise use default
          geometryPath: currentModel ? currentModel.path : './geometry/bunny/bunny.obj',
          iterations: numIterations
        })
      });

      const data = await response.json();

      if (data.status === 'success') {
        setStatus('running');
        console.log('Optimization started successfully');
        // Start polling for status updates
        checkStatus();
        // Start polling for parameter updates
        checkParameters();
      } else {
        setError(data.message);
      }
    } catch (error) {
      console.error('Error starting optimization:', error);
      setError('Failed to connect to the server');
    }
  };

  const checkServerStatus = async () => {
    setServerStatus('checking');
    try {
      const response = await fetch('http://localhost:3001/api');
      const data = await response.json();

      setServerStatus(data.status);
    } catch (error) {
      setServerStatus('error');
      console.error('Error checking server status:', error);
    }
  };

  // Function to check optimization status
  const checkStatus = async () => {
    try {
      const response = await fetch('http://localhost:3001/api/optimization-status');
      const data = await response.json();

      setStatus(data.running ? 'running' : 'completed');
      setOutput(data.output);

      if (data.error) {
        setError(data.error);
      }

      // Continue polling if still running
      if (data.running) {
        setTimeout(checkStatus, 2000);
      }
    } catch (error) {
      console.error('Error checking status:', error);
    }
  };

  // Update the checkParameters function to track current iteration
  const checkParameters = async () => {
    try {
      const response = await fetch('http://localhost:3001/api/current-parameters');
      const data = await response.json();
      console.log('Current parameters:', data);

      if (data.exists) {
        setParameters(data.parameters);

        // Extract current iteration from parameters
        if (Object.keys(data.parameters).length > 0) {
          const iteration = parseInt(Object.keys(data.parameters)[0]);
          setCurrentIteration(iteration);
        }
      }
    } catch (error) {
      console.error('Error checking parameters:', error);
    }
  };

  // Modify function to handle file selection and immediate upload
  const handleFileSelect = async (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files);
      // Only accept image files
      const imageFiles = newFiles.filter(file =>
        file.type.startsWith('image/')
      );

      if (imageFiles.length === 0) {
        setUploadStatus('No valid image files selected');
        return;
      }

      setSelectedFiles(imageFiles);
      setUploadStatus('Uploading...');

      // Create form data and upload immediately
      const formData = new FormData();
      imageFiles.forEach((file) => {
        formData.append('targetImages', file);
      });

      try {
        const response = await fetch('http://localhost:3001/api/upload-targets', {
          method: 'POST',
          body: formData
        });

        const data = await response.json();

        if (data.success) {
          setUploadStatus(`Upload successful! ${imageFiles.length} images uploaded.`);
          setSelectedFiles([]);
          // Refresh the list of target images
          fetchTargetImages();
        } else {
          setUploadStatus(`Upload failed: ${data.message}`);
        }
      } catch (error) {
        console.error('Error uploading files:', error);
        setUploadStatus('Upload failed: Network error');
      }
    }
  };

  // Function to handle model file selection and upload
  const handleModelSelect = async (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      
      // Check if file is an OBJ file
      if (!file.name.toLowerCase().endsWith('.obj')) {
        setModelUploadStatus('Only OBJ files are supported');
        return;
      }
      
      setSelectedModelFile(file);
      setModelUploadStatus('Model selected. Click upload to proceed.');
    }
  };

  // Function to upload the selected model file
  const uploadModel = async () => {
    if (!selectedModelFile) {
      setModelUploadStatus('No model file selected');
      return;
    }

    setModelUploadStatus('Uploading model...');

    const formData = new FormData();
    formData.append('modelFile', selectedModelFile);

    try {
      const response = await fetch('http://localhost:3001/api/upload-model', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (data.success) {
        setModelUploadStatus(`Model uploaded successfully!`);
        setSelectedModelFile(null);
        // Refresh the current model information
        fetchCurrentModel();
      } else {
        setModelUploadStatus(`Upload failed: ${data.message}`);
      }
    } catch (error) {
      console.error('Error uploading model:', error);
      setModelUploadStatus('Upload failed: Network error');
    }
  };

  // Function to fetch the current model information
  const fetchCurrentModel = async () => {
    try {
      const response = await fetch('http://localhost:3001/api/current-model');
      const data = await response.json();

      if (data.success && data.model) {
        setCurrentModel(data.model);
      } else {
        setCurrentModel(null);
      }
    } catch (error) {
      console.error('Error fetching current model:', error);
      setCurrentModel(null);
    }
  };

  // Function to fetch the current target images from the server
  const fetchTargetImages = async () => {
    try {
      const response = await fetch('http://localhost:3001/api/target-images');
      const data = await response.json();

      if (data.images) {
        setTargetImages(data.images);
      }
    } catch (error) {
      console.error('Error fetching target images:', error);
    }
  };

  // Auto-check for parameters on component mount
  useEffect(() => {
    checkServerStatus();
    checkParameters();
    // Add call to fetch target images
    fetchTargetImages();
    // Add call to fetch current model
    fetchCurrentModel();
  }, []);

  // Add new useEffect to handle parameter polling based on status
  useEffect(() => {
    let timerId = null;

    const pollParameters = async () => {
      await checkParameters();
      if (status === 'running') {
        timerId = setTimeout(pollParameters, 100);
      }
    };

    if (status === 'running') {
      pollParameters();
    }

    // Cleanup function to clear timeout when status changes or component unmounts
    return () => {
      if (timerId) clearTimeout(timerId);
    };
  }, [status]); // This makes the effect run whenever status changes

  // Helper function to format parameter display
  const renderParameterValue = (name, value) => {
    // For base_color, display a color swatch
    if (name === 'base_color') {
      const rgbColor = `rgb(${Math.round(value[0] * 255)}, ${Math.round(value[1] * 255)}, ${Math.round(value[2] * 255)})`;
      return (
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div
            style={{
              width: '20px',
              height: '20px',
              backgroundColor: rgbColor,
              marginRight: '10px',
              border: '1px solid #ccc'
            }}
          />
          <span>[{value.map(v => v.toFixed(3)).join(', ')}]</span>
        </div>
      );
    }

    // For numeric values, show a progress bar
    if (typeof value === 'number') {
      return (
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ flex: 1, marginRight: '10px' }}>
            <div style={{
              width: `${value * 100}%`,
              height: '8px',
              backgroundColor: '#4CAF50',
              borderRadius: '4px'
            }} />
          </div>
          <span style={{ minWidth: '60px', textAlign: 'right' }}>{value.toFixed(3)}</span>
        </div>
      );
    }

    // Default rendering
    return JSON.stringify(value);
  };

  return (
    <> {serverStatus === 'success' && (
      <div className="App" style={{ padding: '20px', maxWidth: '900px', margin: '0 auto' }}>
        <h1>Parameter Optimization Demo</h1>

        {/* Add Model Upload Section */}
        <div style={{
          border: '1px solid #ddd',
          borderRadius: '4px',
          padding: '15px',
          backgroundColor: '#f9f9f9',
          marginBottom: '20px',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
        }}>
          <h3>3D Model</h3>
          <p>Upload an OBJ file to use for the optimization process</p>

          <div style={{ marginBottom: '15px', display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
            <input
              type="file"
              accept=".obj"
              onChange={handleModelSelect}
              style={{ flexGrow: 1 }}
            />
            <button 
              onClick={uploadModel}
              disabled={!selectedModelFile || status === 'running'}
              style={{
                padding: '8px 15px',
                backgroundColor: !selectedModelFile || status === 'running' ? '#ccc' : '#2196F3',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: !selectedModelFile || status === 'running' ? 'not-allowed' : 'pointer'
              }}
            >
              Upload Model
            </button>
          </div>

          {modelUploadStatus && (
            <div style={{ marginTop: '5px', color: modelUploadStatus.includes('failed') ? 'red' : 'green' }}>
              {modelUploadStatus}
            </div>
          )}

          {/* Add 3D model preview */}
          {currentModel && (
            <div style={{ marginTop: '15px' }}>
              <ModelViewer 
                modelPath={currentModel.path} 
                bsdfParams={parameters && Object.keys(parameters).length > 0 ? 
                  parameters[Object.keys(parameters)[0]].bsdf_params : null} 
              />
            </div>
          )}
        </div>

        {/* Add Target Image Upload Section */}
        <div style={{
          border: '1px solid #ddd',
          borderRadius: '4px',
          padding: '15px',
          backgroundColor: '#f9f9f9',
          marginBottom: '20px',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
        }}>
          <h3>Target Images</h3>
          <p>Upload images to use as optimization targets</p>

          <div style={{ marginBottom: '15px' }}>
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={handleFileSelect}
              style={{ marginRight: '10px' }}
            />
            {uploadStatus && (
              <div style={{ marginTop: '5px', color: uploadStatus.includes('failed') ? 'red' : 'green' }}>
                {uploadStatus}
              </div>
            )}
          </div>

          {/* Current target images */}
          {targetImages.length > 0 ? (
            <div>
              <h4>Current Target Images:</h4>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                {targetImages.map((image, index) => (
                  <img
                    src={`http://localhost:3001/targets/${image.filename}`}
                    alt={image.filename}
                    style={{ width: '100px', height: '100px', objectFit: 'cover', borderRadius: '7px' }}
                  />
                ))}
              </div>
            </div>
          ) : (
            <p>No target images uploaded yet.</p>
          )}
        </div>

        <div style={{ marginBottom: '20px' }}>
          <div style={{ marginBottom: '15px' }}>
            <label htmlFor="iterations" style={{ marginRight: '10px', fontWeight: 'bold' }}>
              Number of Iterations:
            </label>
            <input
              id="iterations"
              type="number"
              min="1"
              max="10000"
              value={numIterations}
              onChange={(e) => setNumIterations(Math.max(1, parseInt(e.target.value) || 1))}
              disabled={status === 'running'}
              style={{
                padding: '8px',
                borderRadius: '4px',
                border: '1px solid #ccc',
                width: '100px'
              }}
            />
          </div>

          <button
            onClick={startOptimization}
            disabled={status === 'running'}
            style={{
              padding: '10px 20px',
              fontSize: '16px',
              backgroundColor: status === 'running' ? '#ccc' : '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: status === 'running' ? 'not-allowed' : 'pointer'
            }}
          >
            {status === 'running' ? 'Optimization Running...' : 'Start Optimization'}
          </button>

          {status === 'running' && (
            <div style={{ marginTop: '15px' }}>
              {/* Progress bar */}
              <div style={{ marginBottom: '5px' }}>
                <span>Progress: {currentIteration} / {numIterations} iterations ({Math.round((currentIteration / numIterations) * 100)}%)</span>
              </div>
              <div style={{
                width: '100%',
                height: '12px',
                backgroundColor: '#e0e0e0',
                borderRadius: '6px',
                overflow: 'hidden'
              }}>
                <div style={{
                  width: `${(currentIteration / numIterations) * 100}%`,
                  height: '100%',
                  backgroundColor: '#4CAF50',
                  transition: 'width 0.3s ease'
                }} />
              </div>
            </div>
          )}
        </div>

        {/* Enhanced Parameter display section */}
        {parameters && (
          <div style={{
            border: '1px solid #ddd',
            borderRadius: '4px',
            padding: '15px',
            backgroundColor: '#f9f9f9',
            marginBottom: '20px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
              <h3 style={{ margin: '0' }}>Current Parameters</h3>
              {Object.keys(parameters)[0] && (
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span style={{ fontWeight: 'bold', marginRight: '10px' }}>
                    Iteration: {Object.keys(parameters)[0]}
                  </span>
                  {parameters[Object.keys(parameters)[0]].loss !== undefined && (
                    <span style={{
                      backgroundColor: '#f0f0f0',
                      padding: '4px 8px',
                      borderRadius: '4px',
                      fontSize: '14px'
                    }}>
                      Loss: {parameters[Object.keys(parameters)[0]].loss.toFixed(6)}
                    </span>
                  )}
                </div>
              )}
            </div>

            <div style={{
              border: '1px solid #e0e0e0',
              borderRadius: '4px',
              padding: '10px',
              backgroundColor: '#ffffff'
            }}>
              {Object.keys(parameters).map(iterKey => (
                parameters[iterKey].bsdf_params && (
                  <div key={iterKey}>
                    <h4 style={{ borderBottom: '1px solid #eee', paddingBottom: '5px', marginBottom: '10px' }}>
                      Material Properties
                    </h4>
                    {Object.entries(parameters[iterKey].bsdf_params).map(([name, value]) => (
                      <div key={name} style={{ marginBottom: '10px' }}>
                        <div style={{ fontWeight: 'bold', marginBottom: '3px', textTransform: 'capitalize' }}>
                          {name.replace(/object.bsdf./g, '').replace(/.value/g, '').replace(/_/g, ' ')}:
                        </div>
                        {renderParameterValue(name, value)}
                      </div>
                    ))}
                  </div>
                )
              ))}
            </div>
          </div>
        )}

        {output.length > 0 && (
          <div style={{
            marginTop: '20px',
            border: '1px solid #ddd',
            borderRadius: '4px',
            padding: '10px',
            backgroundColor: '#f5f5f5',
            height: '300px',
            overflowY: 'auto'
          }}>
            <h3>Output:</h3>
            <pre style={{ whiteSpace: 'pre-wrap' }}>
              {output.join('')}
            </pre>
          </div>
        )}
      </div>
    )}
    </>
  );
}
export default App;
