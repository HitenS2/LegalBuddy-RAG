interface UploadResponse {
  success: boolean;
  fileId: string;
  message: string;
}

interface SummaryResponse {
  documentType: string;
  riskLevel: string;
  keyDates: string[];
  actionItems: string[];
}

interface ExtractResponse {
  parties: string[];
  dates: {
    effective: string;
    termination: string;
  };
  contractValue: string;
  governingLaw: string;
  criticalClauses: {
    type: string;
    description: string;
    severity: 'warning' | 'danger';
  }[];
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://127.0.0.1:5000/api/upload', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Upload failed');
  }

  return response.json();
}

export async function getDocumentSummary(fileId: string): Promise<SummaryResponse> {
  const response = await fetch(`/api/summary/${fileId}`);
  
  if (!response.ok) {
    throw new Error('Failed to get summary');
  }

  return response.json();
}

export async function extractDocumentInfo(fileId: string): Promise<ExtractResponse> {
  const response = await fetch(`/api/extract/${fileId}`);
  
  if (!response.ok) {
    throw new Error('Failed to extract information');
  }

  return response.json();
}