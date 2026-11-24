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
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 flex items-center justify-between">

        <div>
          <p className="text-sm text-slate-400">Confidence Score</p>
          <p className="text-4xl font-bold text-white">
            {confidence}%
          </p>
        </div>

        <div className="relative w-32 h-32 flex items-center justify-center">
          <svg width="128" height="128" className="-rotate-90">
            <circle
              cx="64"
              cy="64"
              r="56"
              stroke="#334155"
              strokeWidth="10"
              fill="none"
            />
            <circle
              cx="64"
              cy="64"
              r="56"
              stroke="#6366f1"
              strokeWidth="10"
              fill="none"
              strokeDasharray={2 * Math.PI * 56}
              strokeDashoffset={
                2 * Math.PI * 56 * (1 - result.confidence)
              }
              strokeLinecap="round"
            />
          </svg>
          <span className="absolute text-xl font-bold">
            {confidence}%
          </span>
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

      {result.comparison && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
          <h3 className="text-slate-300 font-semibold mb-4">
            Ensemble Model Comparison
          </h3>

          <div className="space-y-4">

            {/* Model 1 */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-400">
                  {result.comparison.model1.name}
                </span>
                <span className="font-semibold text-white">
                  {(result.comparison.model1.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-slate-700 h-2 rounded-full">
                <div
                  className="bg-indigo-500 h-2 rounded-full"
                  style={{
                    width: `${result.comparison.model1.confidence * 100}%`
                  }}
                />
              </div>
            </div>

            {/* Model 2 */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-400">
                  {result.comparison.model2.name}
                </span>
                <span className="font-semibold text-white">
                  {(result.comparison.model2.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-slate-700 h-2 rounded-full">
                <div
                  className="bg-green-500 h-2 rounded-full"
                  style={{
                    width: `${result.comparison.model2.confidence * 100}%`
                  }}
                />
              </div>
            </div>

            {/* Ensemble */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-400 font-semibold">
                  Ensemble Output
                </span>
                <span className="font-bold text-indigo-400">
                  {(result.comparison.ensemble.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-slate-700 h-3 rounded-full">
                <div
                  className="bg-indigo-600 h-3 rounded-full"
                  style={{
                    width: `${result.comparison.ensemble.confidence * 100}%`
                  }}
                />
              </div>
            </div>

          </div>
        </div>
      )}

      {/* Grad-CAM Section */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h3 className="font-semibold mb-4 text-slate-300">
          Grad-CAM Visualization
        </h3>

        {gradcamLoading ? (
          <div className="flex flex-centre justify-center py-8">
            <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-indigo-400 border-r-2"></div>
          </div>
        ) : gradcamData?.heatmap_url ? (
          <img
            src={`${API_URL}${gradcamData.heatmap_url.replace('/api', '')}`}
            alt="GradCAM"
            className="w-full max-w-full object-contain mx-auto rounded-xl border border-slate-700"
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