const express = require('express');
const { spawn } = require('child_process');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const formidable = require('formidable');

const app = express();
const port = 3001;

// Enable CORS
app.use(cors());
app.use(express.json());

// Set up target directory for file uploads
const targetDir = path.join(__dirname, 'testing/parameter_extraction/targets');
// Set up model directory for OBJ file uploads
const modelDir = path.join(__dirname, 'geometry/uploaded_models');

// Create model directory if it doesn't exist
if (!fs.existsSync(modelDir)) {
  fs.mkdirSync(modelDir, { recursive: true });
}

// remove existing files
clearTargetDirectory(targetDir);
clearTargetDirectory(modelDir);
// Store current model information
let currentModel = null;

// In-memory storage for optimization data
const optimizationData = {
  status: {
    running: false,
    output: [],
    error: null
  },
  parameters: null
};

// API endpoint to check if the server is running
app.get('/api', (req, res) => {
  res.json({ status: 'success' });
});

// API endpoint to start optimization
app.post('/api/run-optimization', (req, res) => {
  if (optimizationData.status.running) {
    return res.json({
      status: 'error',
      message: 'Optimization already running'
    });
  }

  // Get parameters from request body
  const { method = 'cuda_ad_rgb', geometryPath = "./geometry/bunny/bunny.obj", target_images = null, iterations = 50 } = req.body;

  // Reset status and parameters
  optimizationData.status = {
    running: true,
    output: [],
    error: null
  };

  optimizationData.parameters = null;
  
  console.log('Starting Python optimization process with parameters:', { method, geometryPath, iterations });
  
  // Spawn Python process with arguments
  const pythonProcess = spawn('python', [
    '-u', 
    'run_optimization.py',
    '--method', method,
    '--geometry', geometryPath,
    '--iterations', iterations.toString(),
  ], {
    cwd: __dirname // Run in the project directory
  });

  // Collect output
  i = 0;
  pythonProcess.stdout.on('data', (data) => {
    const output = data.toString();
    //console.log(`${i}: ${output}`);
    i++;
    
    // Split the output by lines and process each line individually
    const lines = output.split('\n');
    for (const line of lines) {
      if (line.trim().startsWith('PARAMS=')) {
        try {
          // Parse JSON, extracting just the JSON part
          const jsonString = line.substring(7).trim();
          const params = JSON.parse(jsonString);
          optimizationData.parameters = params;
        } catch (error) {
          console.error('Error parsing parameters:', error);
          console.error('Raw parameter string:', line.substring(7).trim());
        }
      }
    }
  });

  // Handle errors
  pythonProcess.stderr.on('data', (data) => {
    const error = data.toString();
    console.error(`Python error: ${error}`);
    optimizationData.status.error = error;
  });

  // Process completion
  pythonProcess.on('close', (code) => {
    optimizationData.status.running = false;
    console.log(`Python process exited with code ${code}`);
  });

  // Respond immediately
  res.json({
    status: 'success',
    message: 'Optimization started'
  });
});

// API endpoint to upload model file
app.post('/api/upload-model', async (req, res) => {
  try {
    clearTargetDirectory(modelDir);

    // Create a new formidable form instance
    const form = new formidable.IncomingForm({
      keepExtensions: true,
      uploadDir: modelDir
    });
    
    // Parse the form
    form.parse(req, (err, fields, files) => {
      if (err) {
        console.error('Upload error:', err);
        return res.status(400).json({
          success: false,
          message: err.message
        });
      }
      
      const uploadedFile = files.modelFile;
      
      if (!uploadedFile) {
        return res.status(400).json({
          success: false,
          message: 'No model file was uploaded.'
        });
      }
      
      // Check if it's an OBJ file
      if (!uploadedFile[0].originalFilename.toLowerCase().endsWith('.obj')) {
        // Delete the uploaded file
        fs.unlinkSync(uploadedFile[0].filepath);
        return res.status(400).json({
          success: false,
          message: 'Only OBJ files are supported.'
        });
      }
      
      // Rename the file to include timestamp to avoid conflicts
      const filename = `model_${Date.now()}.obj`;
      const finalPath = path.join(modelDir, filename);
      
      fs.rename(uploadedFile[0].filepath, finalPath, (err) => {
        if (err) {
          console.error('Error renaming file:', err);
          return res.status(500).json({
            success: false,
            message: 'Error processing the uploaded file.'
          });
        }
        
        // Update current model information
        currentModel = {
          filename: filename,
          originalName: uploadedFile[0].originalFilename,
          path: `./geometry/uploaded_models/${filename}`,
          uploadedAt: new Date().toISOString()
        };
        
        console.log(`Model uploaded successfully: ${filename}`);
        res.json({
          success: true,
          message: 'Model uploaded successfully.',
          model: currentModel
        });
      });
    });
  } catch (error) {
    console.error('Error uploading model:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// API endpoint to get current model information
app.get('/api/current-model', (req, res) => {
  res.json({
    success: true,
    model: currentModel
  });
});

// API endpoint to check status
app.get('/api/optimization-status', (req, res) => {
  res.json(optimizationData.status);
});

// API endpoint to get current parameters
app.get('/api/current-parameters', (req, res) => {
  if (optimizationData.parameters === null) {
    return res.json({
      exists: false,
      message: 'No parameter data available yet'
    });
  }
  
  return res.json({
    exists: true,
    parameters: optimizationData.parameters,
  });
});

// API endpoint to upload target images
app.post('/api/upload-targets', async (req, res) => {
  try {
    // Clear existing target images before uploading new ones
    console.log('Clearing target directory...');
    clearTargetDirectory(targetDir);
    console.log('Target directory cleared successfully');
    
    // Create a new formidable form instance
    const form = new formidable.IncomingForm({
      multiples: true,
      keepExtensions: true,
      uploadDir: targetDir,
      filter: function ({ name, originalFilename, mimetype }) {
        // Only accept image files
        return mimetype && mimetype.startsWith('image/');
      }
    });
    
    // Parse the form
    form.parse(req, (err, fields, files) => {
      if (err) {
        console.error('Upload error:', err);
        return res.status(400).json({
          success: false,
          message: err.message
        });
      }
      
      // Extract uploaded files (formidable uses array for multiple files with same field name)
      const uploadedFiles = files.targetImages || [];
      // Convert to array if it's not already
      const fileArray = Array.isArray(uploadedFiles) ? uploadedFiles : [uploadedFiles];
      
      if (fileArray.length === 0) {
        return res.status(400).json({
          success: false,
          message: 'No files were uploaded.'
        });
      }
      
      console.log(`Successfully uploaded ${fileArray.length} files`);
      res.json({
        success: true,
        message: `${fileArray.length} files uploaded successfully.`,
      });
    });
  } catch (error) {
    console.error('Error uploading files:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// API endpoint to get all target images
app.get('/api/target-images', async (req, res) => {
  try {
    const files = await getOrderedTargetFiles();
    
    res.json({
      success: true,
      images: files.map((filename, index) => ({
        filename,
        position: index
      }))
    });
  } catch (error) {
    console.error('Error getting target images:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Helper function to get ordered list of target files
async function getOrderedTargetFiles() {
  return new Promise((resolve, reject) => {
    fs.readdir(targetDir, (err, files) => {
      if (err) {
        return reject(err);
      }
      
      // Filter for image files and sort them
      const imageFiles = files.filter(file => {
        const ext = path.extname(file).toLowerCase();
        return ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.hdr', '.exr'].includes(ext);
      }).sort();
      
      resolve(imageFiles);
    });
  });
}

// Helper function to clear all image files from target directory
function clearTargetDirectory(targetDir) {
  fs.readdir(targetDir, (err, files) => {
    if (err) {
      console.error('Error reading target directory:', err);
    } else {
      files.forEach(file => {
        fs.unlink(path.join(targetDir, file), (err) => {
          if (err) {
            console.error('Error deleting file:', err);
          }
        });
      });
    }
  });
}

// Serve any static files from optimization_results (for images, etc)
app.use('/results', express.static(path.join(__dirname, 'optimization_results')));

// Serve target images
app.use('/targets', express.static(targetDir));

// Serve model files
app.use('/models', express.static(modelDir));

// Start the server
app.listen(port, () => {
  console.log(`Node.js server running at http://localhost:${port}`);
});
