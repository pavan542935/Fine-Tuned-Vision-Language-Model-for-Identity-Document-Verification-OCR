import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, CheckCircle, AlertCircle, Fingerprint } from 'lucide-react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('active');
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('active');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('active');
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('image/')) {
      setFile(droppedFile);
      setPreview(URL.createObjectURL(droppedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleExtract = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'https://curly-cows-decide.loca.lt';
      const response = await fetch(`${apiUrl}/api/extract`, {
        method: 'POST',
        headers: {
          'Bypass-Tunnel-Reminder': 'true'
        },
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        throw new Error(errData?.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to connect to the backend server.");
    } finally {
      setLoading(false);
    }
  };

  const clearSelection = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>ID Verification <span>OCR</span></h1>
        <p>Ultra-precise identity extraction powered by Qwen2-VL LoRA</p>
      </header>

      <main className="main-content">
        <div className="panel">
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: 0, fontWeight: 600, letterSpacing: '-0.5px' }}>
            <FileText size={26} color="#ffffff" opacity={0.8} /> Document Upload
          </h2>

          {!file ? (
            <div
              className="upload-zone"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <UploadCloud className="upload-icon" />
              <p>Drag & drop your ID card here</p>
              <span>or click to browse files</span>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*"
                style={{ display: 'none' }}
              />
            </div>
          ) : (
            <div className="preview-container">
              <img src={preview} alt="Document preview" className="image-preview" />

              <div style={{ display: 'flex', gap: '1rem', width: '100%' }}>
                <button className="glass-btn" onClick={clearSelection} disabled={loading}>
                  Clear Selection
                </button>
                <button className="glass-btn primary" onClick={handleExtract} disabled={loading}>
                  {loading ? (
                    <span className="btn-content">
                      <span className="loader"></span> Extracting AI Data...
                    </span>
                  ) : (
                    <span className="btn-content">
                      <Fingerprint size={20} /> Extract Data
                    </span>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="panel">
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: 0, fontWeight: 600, letterSpacing: '-0.5px' }}>
            <CheckCircle size={26} color={result ? "#ff3333" : "#ffffff"} opacity={result ? 1 : 0.8} /> Extraction Results
          </h2>

          {error && (
            <div style={{ color: '#ffcccc', background: 'rgba(255, 51, 51, 0.1)', border: '1px solid rgba(255, 51, 51, 0.2)', padding: '1rem', borderRadius: '12px', display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
              <AlertCircle size={22} /> {error}
            </div>
          )}

          {!result && !error && !loading && (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#555555', fontStyle: 'italic' }}>
              Awaiting document upload...
            </div>
          )}

          {loading && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1.5rem' }}>
              <span className="loader" style={{ width: '48px', height: '48px', borderWidth: '4px', borderColor: 'rgba(255,255,255,0.05)', borderTopColor: '#ff3333' }}></span>
              <p style={{ color: '#888888', letterSpacing: '1px' }}>RUNNING NEURAL INFERENCE...</p>
            </div>
          )}

          {result && (
            <pre className="results-pre">
              {JSON.stringify(result, null, 2)}
            </pre>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
