import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  FileText, 
  FolderOpen, 
  Download, 
  Play, 
  Trash2, 
  ShieldCheck, 
  AlertCircle,
  RefreshCw,
  Sliders
} from 'lucide-react';

import { apiService } from './services/api';
import { 
  downloadFile, 
  generateAggregatedTextReport, 
  formatSingleResultReport 
} from './utils/helpers';

import Card from './components/common/Card';
import Loader from './components/common/Loader';
import Toast from './components/common/Toast';
import Modal from './components/common/Modal';
import Drawer from './components/common/Drawer';

import DragDropUpload from './components/upload/DragDropUpload';
import BatchQueue from './components/upload/BatchQueue';

import { PreprocessingSandbox, LLMConfigPanel } from './components/sandbox/ParameterSliders';
import ImageComparer from './components/sandbox/ImageComparer';

import StructuredCard from './components/dashboard/StructuredCard';
import ConfidenceGauge from './components/dashboard/ConfidenceGauge';
import TextHighlight from './components/dashboard/TextHighlight';
import MedicineTable from './components/dashboard/MedicineTable';

const DEFAULT_PREPROCESS_CONFIG = {
  resize_width: 1500,
  clahe_clip_limit: 2.0,
  clahe_tile_grid_size: [8, 8],
  deskew: true,
  denoise: true,
  adaptive_block_size: 11,
  adaptive_c: 2,
  morphology_kernel_size: [3, 3]
};

const DEFAULT_LLM_CONFIG = {
  enabled: false,
  provider: 'openai',
  model: 'gpt-4o-mini',
  api_key: '',
  temperature: 0.1,
  clinical_context: '',
  mode: 'ocr_refinement',
  api_url: 'http://localhost:11434/v1'
};

export default function App() {
  // Navigation
  const [activeMode, setActiveMode] = useState('single'); // 'single' | 'batch'
  
  // API Health status
  const [backendHealth, setBackendHealth] = useState('checking'); // 'checking' | 'online' | 'offline'
  
  // Sandbox state config
  const [prepConfig, setPrepConfig] = useState(DEFAULT_PREPROCESS_CONFIG);
  const [llmConfig, setLlmConfig] = useState(DEFAULT_LLM_CONFIG);
  const [isPrepDrawerOpen, setIsPrepDrawerOpen] = useState(false);

  // Single Upload State
  const [selectedFile, setSelectedFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [singleResult, setSingleResult] = useState(null);

  // Batch Upload State
  const [batchItems, setBatchItems] = useState([]);
  const [activeJobId, setActiveJobId] = useState(null);
  const [selectedBatchItem, setSelectedBatchItem] = useState(null);

  // Toast Notifications
  const [toast, setToast] = useState(null);

  // Check backend health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await apiService.checkHealth();
        setBackendHealth('online');
      } catch {
        setBackendHealth('offline');
        showToast('FastAPI Backend is currently offline. Please run the backend service.', 'error');
      }
    };
    checkHealth();
  }, []);

  // Poll batch status if active job ID exists
  useEffect(() => {
    let intervalId;
    if (activeJobId) {
      intervalId = setInterval(async () => {
        try {
          const res = await apiService.getBatchStatus(activeJobId);
          
          setBatchItems(prevItems => {
            const updated = prevItems.map(item => {
              const matchedResult = res.results.find(r => r.filename === item.filename);
              if (matchedResult) {
                if (matchedResult.result) {
                  return { ...item, status: 'success', result: matchedResult.result };
                } else if (matchedResult.error) {
                  return { ...item, status: 'failed', error: matchedResult.error };
                }
              }
              return item;
            });
            
            // If all items are resolved, clear job
            const allDone = updated.every(item => item.status === 'success' || item.status === 'failed');
            if (allDone) {
              setActiveJobId(null);
              showToast('Batch processing completed successfully!', 'success');
            }
            return updated;
          });
        } catch (err) {
          console.error("Failed to poll batch status", err);
        }
      }, 2500);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [activeJobId]);

  const showToast = (message, type = 'info') => {
    setToast({ message, type });
  };

  const handleResetConfig = () => {
    setPrepConfig(DEFAULT_PREPROCESS_CONFIG);
    showToast('Sandbox parameters reset to pipeline defaults.', 'info');
  };

  // Single OCR trigger
  const handleSingleFileSelect = async (file) => {
    setSelectedFile(file);
    setIsProcessing(true);
    setSingleResult(null);

    try {
      const data = await apiService.ocrSingle(file, prepConfig, llmConfig);
      setSingleResult(data);
      showToast('Prescription analysed successfully!', 'success');
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message || 'OCR processing failed';
      showToast(errorMsg, 'error');
      setSingleResult(null);
    } finally {
      setIsProcessing(false);
    }
  };

  // Single medicines edit handler
  const handleMedicinesChange = (updatedMeds) => {
    if (singleResult) {
      setSingleResult({
        ...singleResult,
        extracted_fields: {
          ...singleResult.extracted_fields,
          medicines: updatedMeds
        }
      });
    }
  };

  // Single field edit handler
  const handleFieldChange = (updatedFields) => {
    if (singleResult) {
      setSingleResult({
        ...singleResult,
        extracted_fields: updatedFields
      });
    }
  };

  // Single downloads
  const downloadSingleJSON = () => {
    if (!singleResult || !selectedFile) return;
    downloadFile(
      JSON.stringify(singleResult, null, 2),
      `${selectedFile.name.split('.')[0]}_ocr_results.json`,
      'application/json'
    );
  };

  const downloadSingleTXT = () => {
    if (!singleResult || !selectedFile) return;
    const txtReport = formatSingleResultReport(singleResult, selectedFile.name);
    downloadFile(
      txtReport,
      `${selectedFile.name.split('.')[0]}_clinical_summary.txt`,
      'text/plain'
    );
  };

  // Batch trigger
  const handleBatchFilesSelect = (files) => {
    const list = Array.from(files).map((file, idx) => ({
      id: `${Date.now()}-${idx}`,
      filename: file.name,
      file: file,
      status: 'queued',
      progress: 0
    }));
    setBatchItems(prev => [...prev, ...list]);
  };

  const startBatchProcessing = async () => {
    if (batchItems.length === 0) {
      showToast('Please add files to batch queue first.', 'warning');
      return;
    }
    
    // Set all items as processing
    setBatchItems(prev => prev.map(item => ({ ...item, status: 'processing' })));
    
    try {
      const files = batchItems.map(item => item.file);
      const res = await apiService.startBatchJob(files, prepConfig, llmConfig);
      setActiveJobId(res.job_id);
      showToast('Asynchronous batch job started. Processing files in the background...', 'success');
    } catch (err) {
      setBatchItems(prev => prev.map(item => ({ ...item, status: 'failed', error: err.message })));
      showToast('Failed to start batch job: ' + err.message, 'error');
    }
  };

  const clearBatchQueue = () => {
    setBatchItems([]);
    setActiveJobId(null);
    showToast('Batch queue cleared.', 'info');
  };

  // Batch downloads
  const downloadBatchJSON = () => {
    const output = batchItems.map(item => ({
      filename: item.filename,
      status: item.status,
      result: item.result,
      error: item.error
    }));
    downloadFile(
      JSON.stringify(output, null, 2),
      `batch_ocr_results_${Date.now()}.json`,
      'application/json'
    );
  };

  const downloadBatchTXT = () => {
    const summaryReport = generateAggregatedTextReport(batchItems);
    downloadFile(
      summaryReport,
      `batch_summary_report_${Date.now()}.txt`,
      'text/plain'
    );
  };

  return (
    <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
      
      {/* Top Banner Navigation */}
      <header style={{
        backgroundColor: 'rgba(255, 250, 245, 0.78)',
        borderBottom: '1px solid var(--border-color)',
        padding: '16px 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 100,
        backdropFilter: 'blur(18px) saturate(125%)',
        WebkitBackdropFilter: 'blur(18px) saturate(125%)'
      }} className="flex-between">
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '36px',
            height: '36px',
            borderRadius: '50%',
            backgroundColor: 'rgba(101, 184, 181, 0.16)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--accent-primary)',
            boxShadow: '0 8px 18px -14px rgba(43, 100, 117, 0.48)'
          }}>
            <Activity size={20} />
          </div>
          <div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 700, margin: 0, letterSpacing: 'normal' }}>
              Shoreline Prescription OCR
            </h1>
            <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
              Prescription Review Workspace
            </span>
          </div>
        </div>

        {/* Mode Selector Tabs */}
        <div style={{ display: 'flex', gap: '8px', backgroundColor: 'rgba(43, 100, 117, 0.08)', padding: '4px', borderRadius: 'var(--radius-sm)' }}>
          <button 
            onClick={() => { setActiveMode('single'); setSingleResult(null); setSelectedFile(null); }}
            className={`btn ${activeMode === 'single' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '6px 16px', fontSize: '0.8rem', gap: '6px' }}
          >
            <FileText size={14} /> Single Scan
          </button>
          <button 
            onClick={() => { setActiveMode('batch'); }}
            className={`btn ${activeMode === 'batch' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '6px 16px', fontSize: '0.8rem', gap: '6px' }}
          >
            <FolderOpen size={14} /> Batch Queue
          </button>
        </div>

        {/* Preprocessing toggle & API Health badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button 
            onClick={() => setIsPrepDrawerOpen(true)}
            className="btn btn-secondary"
            style={{ padding: '6px 12px', fontSize: '0.8rem', gap: '6px' }}
          >
            <Sliders size={14} />
            <span className="hide-on-mobile">Preprocessing</span>
          </button>
          
          {backendHealth === 'online' ? (
            <span className="badge badge-success" style={{ gap: '6px', fontSize: '0.7rem' }}>
              <ShieldCheck size={12} /> Server Connected
            </span>
          ) : backendHealth === 'offline' ? (
            <span className="badge badge-danger" style={{ gap: '6px', fontSize: '0.7rem', animation: 'pulse-glow 1.5s infinite' }}>
              <AlertCircle size={12} /> Server Offline
            </span>
          ) : (
            <span className="badge badge-primary" style={{ gap: '6px', fontSize: '0.7rem' }}>
              <RefreshCw size={10} className="spin" style={{ animation: 'spin 1s linear infinite' }} /> Connecting...
            </span>
          )}
        </div>
      </header>

      {/* Main Grid Workspace */}
      <main style={{ flexGrow: 1, padding: '32px 0' }}>
        <div className="container">
          <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: '32px', alignItems: 'start' }}>
            
            {/* LEFT COLUMN: PARAMETER SIDEBAR */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', position: 'sticky', top: '100px' }}>
              <LLMConfigPanel
                llmConfig={llmConfig}
                onLLMChange={setLlmConfig}
              />
              
              {activeMode === 'single' && selectedFile && (
                <ImageComparer 
                  originalFile={selectedFile}
                  processedImageUrl={singleResult?.preprocessing?.processed_image_url}
                  isProcessing={isProcessing}
                />
              )}
            </div>

            {/* RIGHT COLUMN: ACTION AND DATA REPORTS AREA */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              
              {activeMode === 'single' ? (
                /* SINGLE MODE WORKSPACE */
                <>
                  {!selectedFile && (
                    <DragDropUpload onFileSelect={handleSingleFileSelect} />
                  )}

                  {isProcessing && (
                    <Card style={{ height: '300px' }} className="flex-center">
                      <Loader 
                        message={
                          llmConfig.enabled && llmConfig.mode === 'direct_vision'
                            ? `Running direct Vision VLM extraction (${llmConfig.provider}: ${llmConfig.model})...`
                            : "Performing multi-engine OCR fallback analysis..."
                        } 
                      />
                    </Card>
                  )}

                  {!isProcessing && singleResult && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', animation: 'fade-in 0.4s ease-out' }}>
                      
                      {/* Action Header Row */}
                      <div className="flex-between">
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <h2 style={{ fontSize: '1.25rem', fontWeight: 700 }}>Analysis Report Output</h2>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>ID: {singleResult.request_id}</span>
                        </div>
                        <div style={{ display: 'flex', gap: '8px' }}>
                          <button onClick={downloadSingleJSON} className="btn btn-secondary" style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
                            <Download size={14} /> Download Raw JSON
                          </button>
                          <button onClick={downloadSingleTXT} className="btn btn-primary" style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
                            <Download size={14} /> Download Clinical Summary
                          </button>
                          <button 
                            onClick={() => { setSelectedFile(null); setSingleResult(null); }} 
                            className="btn btn-danger"
                            style={{ padding: '6px 12px', fontSize: '0.8rem' }}
                          >
                            <Trash2 size={14} /> Reset
                          </button>
                        </div>
                      </div>

                      {/* Stat Metrics Row */}
                      <div className="grid-cols-2">
                        <ConfidenceGauge 
                          ocrConfidence={singleResult.confidence?.ocr}
                          preprocessConfidence={singleResult.confidence?.preprocessing}
                        />
                        <StructuredCard 
                          fields={singleResult.extracted_fields}
                          onFieldChange={handleFieldChange}
                        />
                      </div>

                      {/* Medicines Grid Table */}
                      <MedicineTable 
                        medicines={singleResult.extracted_fields?.medicines}
                        validation={singleResult.medicine_validation}
                        onMedicinesChange={handleMedicinesChange}
                      />

                      {/* Raw text transcription */}
                      <TextHighlight 
                        rawText={singleResult.ocr?.raw_text}
                        wordBoxes={singleResult.ocr?.word_boxes}
                      />

                    </div>
                  )}
                </>
              ) : (
                /* BATCH MODE WORKSPACE */
                <>
                  <DragDropUpload 
                    onFileSelect={handleBatchFilesSelect}
                    accept="image/*,application/pdf"
                    multiple
                  />

                  {batchItems.length > 0 && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                      {/* Toolbar buttons */}
                      <div className="flex-between">
                        <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
                          Batch Files Queue ({batchItems.length} items loaded)
                        </span>
                        <div style={{ display: 'flex', gap: '8px' }}>
                          <button 
                            onClick={startBatchProcessing}
                            className="btn btn-primary"
                            style={{ padding: '8px 16px', fontSize: '0.85rem' }}
                            disabled={!!activeJobId || batchItems.every(i => i.status === 'success')}
                          >
                            <Play size={14} /> Start Batch Pipeline
                          </button>
                          <button 
                            onClick={clearBatchQueue}
                            className="btn btn-danger"
                            style={{ padding: '8px 16px', fontSize: '0.85rem' }}
                          >
                            <Trash2 size={14} /> Clear Queue
                          </button>
                        </div>
                      </div>

                      <BatchQueue 
                        items={batchItems}
                        onSelectItem={(item) => setSelectedBatchItem(item)}
                        onDownloadJSON={downloadBatchJSON}
                        onDownloadTXT={downloadBatchTXT}
                      />
                    </div>
                  )}
                </>
              )}

            </div>
          </div>
        </div>
      </main>

      {/* FOOTER */}
      <footer style={{
        backgroundColor: 'rgba(255, 250, 245, 0.72)',
        borderTop: '1px solid var(--border-color)',
        padding: '20px 24px',
        textAlign: 'center',
        fontSize: '0.8rem',
        color: 'var(--text-muted)',
        marginTop: 'auto'
      }}>
        Shoreline Doctor Prescription Parser. Built with OpenCV and FastAPI.
      </footer>

      {/* TOAST SYSTEM */}
      {toast && (
        <div style={{ position: 'fixed', bottom: '24px', right: '24px', zIndex: 9999 }}>
          <Toast 
            message={toast.message} 
            type={toast.type} 
            onClose={() => setToast(null)} 
          />
        </div>
      )}

      {/* BATCH DETAIL MODAL POPUP */}
      {selectedBatchItem && (
        <Modal 
          isOpen={!!selectedBatchItem} 
          onClose={() => setSelectedBatchItem(null)}
          title={`Batch Item OCR Result: ${selectedBatchItem.filename}`}
          size="lg"
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div className="grid-cols-2">
              <ConfidenceGauge 
                ocrConfidence={selectedBatchItem.result?.confidence?.ocr}
                preprocessConfidence={selectedBatchItem.result?.confidence?.preprocessing}
              />
              <StructuredCard 
                fields={selectedBatchItem.result?.extracted_fields}
                onFieldChange={(updatedFields) => {
                  setBatchItems(prev => prev.map(item => 
                    item.id === selectedBatchItem.id 
                      ? { ...item, result: { ...item.result, extracted_fields: updatedFields } }
                      : item
                  ));
                  setSelectedBatchItem(prev => ({
                    ...prev,
                    result: { ...prev.result, extracted_fields: updatedFields }
                  }));
                }}
              />
            </div>
            
            <MedicineTable 
              medicines={selectedBatchItem.result?.extracted_fields?.medicines}
              validation={selectedBatchItem.result?.medicine_validation}
              onMedicinesChange={(updatedMeds) => {
                setBatchItems(prev => prev.map(item => 
                  item.id === selectedBatchItem.id 
                    ? {
                        ...item,
                        result: {
                          ...item.result,
                          extracted_fields: { ...item.result.extracted_fields, medicines: updatedMeds }
                        }
                      }
                    : item
                ));
                setSelectedBatchItem(prev => ({
                  ...prev,
                  result: {
                    ...prev.result,
                    extracted_fields: { ...prev.result.extracted_fields, medicines: updatedMeds }
                  }
                }));
              }}
            />

            <TextHighlight 
              rawText={selectedBatchItem.result?.ocr?.raw_text}
              wordBoxes={selectedBatchItem.result?.ocr?.word_boxes}
            />
          </div>
        </Modal>
      )}

      {/* PREPROCESSING SANDBOX DRAWER */}
      <Drawer 
        isOpen={isPrepDrawerOpen} 
        onClose={() => setIsPrepDrawerOpen(false)}
        title="Image Preprocessing Sandbox"
      >
        <PreprocessingSandbox 
          config={prepConfig}
          onChange={setPrepConfig}
          onReset={handleResetConfig}
        />
      </Drawer>

      <style dangerouslySetInnerHTML={{__html: `
        @keyframes fade-in {
          0% { opacity: 0; transform: translateY(8px); }
          100% { opacity: 1; transform: translateY(0); }
        }
      `}} />

    </div>
  );
}
