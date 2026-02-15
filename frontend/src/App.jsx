
import { useState } from 'react';
import { Shield } from 'lucide-react';
import axios from 'axios';
import UploadZone from './components/UploadZone';
import ResultsDisplay from './components/ResultsDisplay';

const API_URL = 'http://localhost:5000/api';


function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [error, setError] = useState(null);
  const [c2paData, setC2PAData] = useState(null);
  const [currentJobId, setCurrentJobId] = useState(null);

  const handleFileSelected = (file) => {
    setSelectedFile(file);
    setError(null);
  };

  const handleUpload = async (file) => {
    if (!file) return;

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const uploadResponse = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      const jobId = uploadResponse.data.job_id;
      setCurrentJobId(jobId);

      setUploading(false);
      setAnalyzing(true);

      const detectResponse = await axios.post(`${API_URL}/detect/compare/${jobId}`);

      try {
        const metadataResponse = await axios.get(`${API_URL}/metadata/${jobId}`);
        setMetadata(metadataResponse.data.metadata);
      } catch (metaErr) {
        console.log('Metadata extraction failed:', metaErr);
      }

      setResult(detectResponse.data.result);
      setAnalyzing(false);

      try {
        const c2paResponse = await axios.get(`${API_URL}/c2pa/${jobId}`);
        setC2PAData(c2paResponse.data);
      } catch (c2paErr) {
        console.log('C2PA verification unavailable:', c2paErr);
      }

    } catch (err) {
      console.error('Error:', err);
      setError(err.response?.data?.error || err.message || 'An error occurred');
      setUploading(false);
      setAnalyzing(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setResult(null);
    setMetadata(null);
    setError(null);
    setC2PAData(null);
    setCurrentJobId(null);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white flex flex-col">

      {/* Header */}
      <header className="bg-slate-900 border-b-2 border-slate-700 shadow-xl">
        <div className="max-w-7xl mx-auto px-6 py-8 text-center">
    
          <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight">
            
              AuthentiScan
            
          </h1>

          <p className="text-slate-400 text-lg mt-3">
            Deepfake Detection & Media Authenticity Platform
          </p>

        </div>
      </header>

      {/* Dashboard Layout */}
      <main className="flex flex-grow">

        {/* Sidebar */}
        <aside className="w-64 bg-slate-900 border-r-2 border-slate-700 p-6 hidden md:block">
          <h2 className="text-lg font-semibold mb-6 text-indigo-400">
            Dashboard
          </h2>
          <div className="border-b-2 border-slate-700 mb-6"></div>
          <ul className="space-y-4 text-slate-400 text-sm">
            <li className="hover:text-white transition cursor-pointer">Upload Media</li>
            <li className="hover:text-white transition cursor-pointer">Detection Results</li>
            <li className="hover:text-white transition cursor-pointer">Metadata Analysis</li>
            <li className="hover:text-white transition cursor-pointer">C2PA Verification</li>
          </ul>
        </aside>

        {/* Main Content Area */}
        <div className="flex-1 p-10 max-w-6xl mx-auto w-full">

          {/* Error */}
          {error && (
            <div className="mb-8 bg-red-900/40 border border-red-600 p-5 rounded-xl shadow-md">
              <p className="text-red-400 font-semibold">
                Error: {error}
              </p>
            </div>
          )}

          {/* Loading */}
          {(uploading || analyzing) && (
            <div className="mb-10 bg-indigo-900/30 border border-indigo-600 p-6 rounded-2xl shadow-xl">
              <div className="flex items-center">
                <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-indigo-400 border-r-2 mr-5"></div>
                <div>
                  <p className="text-indigo-300 font-semibold text-lg">
                    {uploading ? 'Uploading file...' : 'Analyzing media with ensemble models...'}
                  </p>
                  <p className="text-slate-400 text-sm mt-1">
                    {analyzing && 'Running MesoNet-4 + XceptionNet detection'}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Upload or Results */}
          <div className="bg-slate-900 border-2 border-slate-700 rounded-2xl p-8 shadow-2xl">

            {!result ? (
              <UploadZone
                onFileSelected={handleFileSelected}
                onUpload={handleUpload}
                uploading={uploading || analyzing}
              />
            ) : (
              <ResultsDisplay
                result={result}
                metadata={metadata}
                c2paData={c2paData}
                jobId={currentJobId}
                onReset={handleReset}
              />
            )}

          </div>

        </div>
      </main>

      {/* Footer */}
      <footer className="bg-slate-900 border-t-2 border-slate-700 py-6">
        <div className="max-w-7xl mx-auto px-6 text-center text-slate-500 text-sm">
          <p>AuthentiScan © 2024 | Capstone Project</p>
          <p className="mt-1">
            Powered by MesoNet-4 + XceptionNet Ensemble
          </p>
        </div>
      </footer>

    </div>
  );
}

export default App;