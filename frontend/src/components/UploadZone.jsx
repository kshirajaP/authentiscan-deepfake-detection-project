import { useState } from 'react';
import { Upload, FileVideo, Image as ImageIcon } from 'lucide-react';

export default function UploadZone({ onFileSelected, onUpload, uploading }) {
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (selectedFile) => {
    const allowedTypes = [
      'video/mp4',
      'video/x-msvideo',
      'video/quicktime',
      'image/jpeg',
      'image/png'
    ];

    if (!allowedTypes.includes(selectedFile.type)) {
      alert('Invalid file type. Please upload MP4, AVI, MOV, JPG, or PNG files.');
      return;
    }

    if (selectedFile.size > 100 * 1024 * 1024) {
      alert('File too large. Maximum size is 100MB.');
      return;
    }

    setFile(selectedFile);
    onFileSelected(selectedFile);
  };

  const getFileIcon = () => {
    if (!file) return <Upload className="w-16 h-16 text-slate-400" />;
    
    if (file.type.startsWith('video/')) {
      return <FileVideo className="w-16 h-16 text-indigo-400" />;
    } else {
      return <ImageIcon className="w-16 h-16 text-emerald-400" />;
    }
  };

  return (
    <div className="rounded-2xl p-10 bg-slate-900 border border-slate-800 shadow-2xl">

      <h2 className="text-2xl font-semibold mb-8 text-center bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
        Upload Media for Analysis
      </h2>

      <div
        className={`border-2 border-dashed rounded-2xl p-16 text-center transition-all duration-300 ${
          dragActive
            ? 'border-indigo-500 bg-indigo-500/10'
            : 'border-slate-700 bg-slate-800'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center">

          {getFileIcon()}

          {!file ? (
            <>
              <p className="text-lg text-slate-300 mt-6 mb-3">
                Drag and drop your file here
              </p>

              <label className="cursor-pointer">
                <span className="text-indigo-400 hover:text-indigo-300 font-semibold transition">
                  Browse to upload
                </span>
                <input
                  type="file"
                  className="hidden"
                  onChange={handleChange}
                  accept="video/mp4,video/x-msvideo,video/quicktime,image/jpeg,image/png"
                />
              </label>

              <p className="text-sm text-slate-500 mt-6">
                Supported: MP4, AVI, MOV, JPG, PNG (Max 100MB)
              </p>
            </>
          ) : (
            <div className="space-y-4 mt-6">

              <p className="text-lg font-semibold text-white">
                {file.name}
              </p>

              <p className="text-sm text-slate-400">
                Size: {(file.size / (1024 * 1024)).toFixed(2)} MB
              </p>

              <div className="flex gap-6 justify-center mt-8">

                <button
                  onClick={() => {
                    setFile(null);
                    onFileSelected(null);
                  }}
                  className="px-6 py-3 border border-slate-600 rounded-xl hover:bg-slate-700 transition"
                  disabled={uploading}
                >
                  Change File
                </button>

                <button
                  onClick={() => onUpload(file)}
                  disabled={uploading}
                  className="px-8 py-3 bg-indigo-600 hover:bg-indigo-500 rounded-xl transition font-semibold shadow-lg disabled:bg-slate-700"
                >
                  {uploading ? 'Uploading...' : 'Upload & Analyze'}
                </button>

              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}