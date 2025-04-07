import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Download, ArrowLeft, Loader2 } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

interface ExtractContent {
  entities?: string[];
  dates?: string[];
  scope?: string;
  sla?: string;
  penalties?: string;
  confidentiality?: string;
  termination?: string;
  commercials?: string;
  risks?: string[] | string;
  [key: string]: any;
}

interface ExtractionData {
  id: number;
  created_at: string;
  document_id: number;
  document_name: string;
  content: ExtractContent;
}

export const ExtractResult: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [extraction, setExtraction] = useState<ExtractionData | null>(null);

  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login');
      return;
    }

    const queryParams = new URLSearchParams(location.search);
    const extractionId = queryParams.get('extraction');

    if (!extractionId) {
      setError('No extraction ID provided');
      setLoading(false);
      return;
    }

    const fetchExtraction = async () => {
      try {
        const token = localStorage.getItem('token');
        const response = await fetch(`http://127.0.0.1:5000/extractions/${extractionId}`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });

        if (!response.ok) {
          throw new Error('Failed to fetch extraction');
        }

        const data = await response.json();
        console.log('Extraction data:', data);
        setExtraction(data);
      } catch (err) {
        console.error('Error fetching extraction:', err);
        setError('Failed to load extraction details');
      } finally {
        setLoading(false);
      }
    };

    fetchExtraction();
  }, [isAuthenticated, location.search, navigate]);

  const downloadExtraction = () => {
    if (!extraction) return;
    
    const token = localStorage.getItem('token');
    window.open(`http://127.0.0.1:5000/download/extraction?extraction_id=${extraction.id}&token=${token}`, '_blank');
  };

  const formatArray = (arr: string[] | undefined) => {
    if (!arr || arr.length === 0) return 'None identified';
    return arr.map((item, index) => (
      <li key={index} className="mb-1">{item}</li>
    ));
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <button
        onClick={() => navigate('/history')}
        className="flex items-center text-gray-600 hover:text-gray-900 mb-4"
      >
        <ArrowLeft className="h-4 w-4 mr-1" />
        Back to History
      </button>

      {loading ? (
        <div className="flex justify-center items-center p-12">
          <Loader2 className="h-8 w-8 animate-spin text-[#2f5233]" />
          <span className="ml-2 text-lg text-gray-600">Loading extraction details...</span>
        </div>
      ) : error ? (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      ) : extraction ? (
        <div>
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-2xl font-bold">{extraction.document_name}</h1>
            <button
              onClick={downloadExtraction}
              className="flex items-center bg-[#2f5233] text-white px-4 py-2 rounded-md hover:bg-[#1e351f]"
            >
              <Download className="h-4 w-4 mr-2" />
              Download PDF
            </button>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6 mb-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Entities Section */}
              <div className="mb-6">
                <h2 className="text-lg font-semibold text-[#2f5233] mb-2">Entities</h2>
                <ul className="list-disc ml-6">
                  {formatArray(extraction.content.entities)}
                </ul>
              </div>

              {/* Key Dates */}
              <div className="mb-6">
                <h2 className="text-lg font-semibold text-[#2f5233] mb-2">Key Dates</h2>
                <ul className="list-disc ml-6">
                  {formatArray(extraction.content.dates)}
                </ul>
              </div>
            </div>

            {/* Contract Details */}
            <div className="grid grid-cols-1 gap-6 mt-6">
              {/* Scope */}
              <div>
                <h2 className="text-lg font-semibold text-[#2f5233] mb-2">Scope</h2>
                <p className="bg-gray-50 p-3 rounded">
                  {extraction.content.scope || 'None specified'}
                </p>
              </div>

              {/* SLA */}
              <div>
                <h2 className="text-lg font-semibold text-[#2f5233] mb-2">Service Level Agreements</h2>
                <p className="bg-gray-50 p-3 rounded">
                  {extraction.content.sla || 'None specified'}
                </p>
              </div>

              {/* Penalties */}
              <div>
                <h2 className="text-lg font-semibold text-[#2f5233] mb-2">Penalties</h2>
                <p className="bg-gray-50 p-3 rounded">
                  {extraction.content.penalties || 'None specified'}
                </p>
              </div>

              {/* Confidentiality */}
              <div>
                <h2 className="text-lg font-semibold text-[#2f5233] mb-2">Confidentiality</h2>
                <p className="bg-gray-50 p-3 rounded">
                  {extraction.content.confidentiality || 'None specified'}
                </p>
              </div>

              {/* Termination */}
              <div>
                <h2 className="text-lg font-semibold text-[#2f5233] mb-2">Termination</h2>
                <p className="bg-gray-50 p-3 rounded">
                  {extraction.content.termination || 'None specified'}
                </p>
              </div>

              {/* Commercials */}
              <div>
                <h2 className="text-lg font-semibold text-[#2f5233] mb-2">Commercials</h2>
                <p className="bg-gray-50 p-3 rounded">
                  {extraction.content.commercials || 'None specified'}
                </p>
              </div>

              {/* Risks */}
              <div>
                <h2 className="text-lg font-semibold text-[#2f5233] mb-2">Risks</h2>
                <div className="bg-gray-50 p-3 rounded">
                  {Array.isArray(extraction.content.risks) ? (
                    <ul className="list-disc ml-6">
                      {formatArray(extraction.content.risks as string[])}
                    </ul>
                  ) : (
                    <p>{extraction.content.risks || 'None identified'}</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center p-8">
          <p className="text-lg text-gray-600">No extraction data found</p>
        </div>
      )}
    </div>
  );
}; 