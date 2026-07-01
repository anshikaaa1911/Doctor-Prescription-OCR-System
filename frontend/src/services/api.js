import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds (useful for OCR fallback + LLM timeouts)
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
    
    if (settings.llm && settings.llm.enabled && settings.llm.api_key) {
      formattedSettings.llm = {
        provider: settings.llm.provider,
        model: settings.llm.model,
        api_key: settings.llm.api_key,
        clinical_context: settings.llm.clinical_context || '',
        api_key_env: settings.llm.provider === 'openai' ? 'OPENAI_API_KEY' : 'NVIDIA_API_KEY',
        timeout_seconds: 20,
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
  }
};
