import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds (useful for OCR fallback + LLM timeouts)
});

// Interceptor to add stored JWT bearer token to requests
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
}, (error) => {
  return Promise.reject(error);
});


/**
 * Builds the FormData with file and optional settings payload
 */
function buildFormData(fileOrFiles, settings) {
  const formData = new FormData();
  
  if (Array.isArray(fileOrFiles)) {
    fileOrFiles.forEach((file) => {
      formData.append('files', file);
    });
  } else {
    formData.append('file', fileOrFiles);
  }

  if (settings) {
    const formattedSettings = {};
    
    if (settings.preprocessing) {
      formattedSettings.preprocessing = settings.preprocessing;
    }
    
    if (settings.llm && settings.llm.enabled) {
      formattedSettings.llm = {
        provider: settings.llm.provider,
        model: settings.llm.model,
        mode: settings.llm.mode || 'ocr_refinement',
        api_key: settings.llm.api_key || '',
        api_url: settings.llm.api_url || 'http://localhost:11434/v1',
        clinical_context: settings.llm.clinical_context || '',
        api_key_env: settings.llm.provider === 'openai' ? 'OPENAI_API_KEY' : 'NVIDIA_API_KEY',
        timeout_seconds: 35,
        max_input_chars: 6000,
        ...(settings.llm.temperature !== undefined ? { temperature: settings.llm.temperature } : {})
      };
    }
    
    formData.append('settings', JSON.stringify(formattedSettings));
  }

  return formData;
}

export const apiService = {
  /**
   * Health status check
   */
  async checkHealth() {
    const response = await client.get('/health');
    return response.data;
  },

  /**
   * OCR a single prescription file
   */
  async ocrSingle(file, preprocessing, llm) {
    const formData = buildFormData(file, { preprocessing, llm });
    const response = await client.post('/ocr', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
    });
    return response.data;
  },

  /**
   * OCR a batch of files synchronously (for small batches)
   */
  async ocrBatchSync(files, preprocessing, llm) {
    const formData = buildFormData(files, { preprocessing, llm });
    const response = await client.post('/ocr/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
    });
    return response.data;
  },

  /**
   * Start an asynchronous background batch job
   */
  async startBatchJob(files, preprocessing, llm) {
    const formData = buildFormData(files, { preprocessing, llm });
    const response = await client.post('/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
    });
    return response.data;
  },

  /**
   * Get the status & results of a background batch job
   */
  async getBatchStatus(job_id) {
    const response = await client.get(`/batch/${job_id}`);
    return response.data;
  },

  /**
   * Get the OCR history of the logged in user
   */
  async fetchHistory() {
    const response = await client.get('/history');
    return response.data;
  }
};

