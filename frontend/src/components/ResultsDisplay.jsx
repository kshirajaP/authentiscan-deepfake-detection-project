import { useState, useEffect } from 'react';
import { CheckCircle, XCircle, AlertTriangle, BarChart3, Shield } from 'lucide-react';
import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

export default function ResultsDisplay({ result, metadata, c2paData, onReset, jobId }) {
  const [gradcamData, setGradcamData] = useState(null);
  const [gradcamLoading, setGradcamLoading] = useState(false);
  const [selectedFrame, setSelectedFrame] = useState(0);

  useEffect(() => {
    if (jobId) fetchGradCAM();
  }, [jobId]);

  const fetchGradCAM = async () => {
    setGradcamLoading(true);
    try {
      const response = await axios.get(`${API_URL}/gradcam/${jobId}`);
      setGradcamData(response.data);
    } catch (error) {
      console.error('Grad-CAM failed:', error);
    } finally {
      setGradcamLoading(false);
    }
  };

  const isFake = result.prediction === 'fake';
  const confidence = (result.confidence * 100).toFixed(1);

  // Confidence Ring Math
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (confidence / 100) * circumference;

  return (
    <div className="space-y-10">

      {/* Verdict Badge */}
      <div className="text-center">
        <div className="flex justify-center mb-4">
          {isFake ? (
            <XCircle className="w-20 h-20 text-red-500" />
          ) : (
            <CheckCircle className="w-20 h-20 text-green-500" />
          )}
        </div>

        <h2 className="text-3xl font-bold mb-3">
          Analysis Complete
        </h2>

        <span className={`px-8 py-3 rounded-full text-lg font-semibold shadow-xl
          ${isFake ? 'bg-red-600 text-white' : 'bg-green-600 text-white'}`}>
          {isFake ? 'FAKE CONTENT DETECTED' : 'AUTHENTIC CONTENT'}
        </span>
      </div>

      {/* Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

        {/* Confidence Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl text-center">
          <h3 className="font-semibold mb-6 text-slate-300">
            Detection Confidence
          </h3>

          <div className="relative w-48 h-48 mx-auto">
            <svg className="w-full h-full transform -rotate-90">
              <circle
                cx="96"
                cy="96"
                r={radius}
                stroke="#334155"
                strokeWidth="12"
                fill="transparent"
              />
              <circle
                cx="96"
                cy="96"
                r={radius}
                stroke={isFake ? "#ef4444" : "#22c55e"}
                strokeWidth="12"
                fill="transparent"
                strokeDasharray={circumference}
                strokeDashoffset={offset}
                strokeLinecap="round"
                className="transition-all duration-700 ease-out"
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center text-3xl font-bold">
              {confidence}%
            </div>
          </div>
        </div>

        {/* Metadata */}
        {metadata && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h3 className="font-semibold mb-4 text-slate-300">
              Metadata Analysis
            </h3>

            {metadata.inconsistencies?.length > 0 ? (
              <ul className="space-y-2 text-sm text-slate-400">
                {metadata.inconsistencies.map((warning, idx) => (
                  <li key={idx}>⚠ {warning}</li>
                ))}
              </ul>
            ) : (
              <p className="text-slate-500 text-sm">
                No metadata inconsistencies detected.
              </p>
            )}

            {metadata.risk_level && (
              <p className="mt-4 text-sm">
                Risk Level:{' '}
                <span className={`font-semibold
                  ${metadata.risk_level === 'high' ? 'text-red-500' :
                    metadata.risk_level === 'medium' ? 'text-yellow-400' :
                    'text-green-500'}`}>
                  {metadata.risk_level.toUpperCase()}
                </span>
              </p>
            )}
          </div>
        )}

        {/* C2PA */}
        {c2paData && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h3 className="font-semibold mb-4 text-slate-300 flex items-center">
              <Shield className="w-5 h-5 mr-2 text-indigo-400" />
              C2PA Verification
            </h3>

            {c2paData.c2pa_verification?.has_credentials ? (
              <p className="text-green-400 text-sm">
                ✓ Valid Content Credentials Found
              </p>
            ) : (
              <p className="text-slate-500 text-sm">
                No C2PA credentials found.
              </p>
            )}
          </div>
        )}
      </div>

      {/* Grad-CAM Section */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h3 className="font-semibold mb-4 text-slate-300">
          Grad-CAM Visualization
        </h3>

        {gradcamLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-indigo-400 border-r-2"></div>
          </div>
        ) : gradcamData?.heatmap_url ? (
          <img
            src={`${API_URL}${gradcamData.heatmap_url.replace('/api', '')}`}
            alt="GradCAM"
            className="rounded-xl border border-slate-700"
          />
        ) : (
          <p className="text-slate-500 text-sm">
            Grad-CAM unavailable for this file.
          </p>
        )}
      </div>

      {/* Reset Button */}
      <div className="text-center">
        <button
          onClick={onReset}
          className="px-10 py-3 bg-indigo-600 hover:bg-indigo-500 rounded-xl transition font-semibold shadow-lg"
        >
          Analyze Another File
        </button>
      </div>

    </div>
  );
}