import React, { useEffect, useState } from "react";
import { Loader2, FileDown, History } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { useNavigate } from "react-router-dom";

interface ExtractionData {
  answer: {
    [section: string]: string | object;
  };
  time_taken: string;
  status: string;
  extraction_id?: number;
  sections_extracted?: string[];
}

interface Extraction {
  id: number;
  created_at: string;
  document_id: number;
  document_name: string;
}

interface ExtractInfoProps {
  selectedFile: File | null;
}

export const ExtractInfo: React.FC<ExtractInfoProps> = ({ selectedFile }) => {
  const [extractionData, setExtractionData] = useState<ExtractionData | null>(null);
  const [extractionId, setExtractionId] = useState<number | null>(null);
  const [existingExtractions, setExistingExtractions] = useState<Extraction[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isPdfLoading, setIsPdfLoading] = useState<boolean>(false);
  const navigate = useNavigate();

  // First fetch all existing extractions
  useEffect(() => {
    const fetchExistingExtractions = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/extractions", {
          method: "GET"
        });

        if (!response.ok) {
          throw new Error("Failed to fetch existing extractions");
        }

        const data = await response.json();
        setExistingExtractions(data.extractions || []);
        
        // If there are existing extractions, use the most recent one
        if (data.extractions && data.extractions.length > 0) {
          const latestExtraction = data.extractions[0]; // First one is the most recent
          setExtractionId(latestExtraction.id);
          fetchExtractionById(latestExtraction.id);
        } else {
          // If no existing extractions, proceed with generating a new one
          fetchNewExtraction();
        }
      } catch (err) {
        console.error("Error fetching existing extractions:", err);
        // If we can't fetch existing extractions, try doing a new extraction
        fetchNewExtraction();
      }
    };

    fetchExistingExtractions();
  }, []);

  // Fetch an extraction by ID
  const fetchExtractionById = async (id: number) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`http://127.0.0.1:5000/extractions/${id}`, {
        method: "GET"
      });

      if (!response.ok) {
        throw new Error("Failed to fetch extraction data");
      }

      const extractionData = await response.json();
      setExtractionData({
        answer: extractionData.content,
        time_taken: "loaded from history",
        status: "success",
        extraction_id: extractionData.id
      });
      
      console.log("✅ Loaded existing extraction:", extractionData);
    } catch (err: any) {
      console.error("❌ Error fetching extraction by ID:", err);
      setError(err.message || "An error occurred while fetching extraction data");
      // Try generating a new extraction if we couldn't load an existing one
      fetchNewExtraction();
    } finally {
      setLoading(false);
    }
  };

  // Generate a new extraction
  const fetchNewExtraction = async () => {
    setLoading(true);
    setError(null);
    try {
      console.log("📡 Generating new extraction from backend...");
      
      const response = await fetch("http://127.0.0.1:5000/extract", {
        method: "GET"
      });

      if (!response.ok) {
        throw new Error("❌ Failed to generate extraction data");
      }

      const data = await response.json();
      setExtractionData(data);
      if (data.extraction_id) {
        setExtractionId(data.extraction_id);
      }
      console.log("✅ New extraction data generated:", data);
    } catch (err: any) {
      console.error("❌ Error generating extraction:", err);
      setError(
        err.message || "An error occurred while generating extraction data"
      );
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPdf = async () => {
    if (!extractionData) return;
    
    setIsPdfLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:5000/download/extraction", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ extraction: extractionData.answer }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate PDF");
      }

      // Create a blob from the PDF Stream
      const blob = await response.blob();
      // Create a link element, set the download attribute and click it
      const link = document.createElement('a');
      link.href = window.URL.createObjectURL(blob);
      link.download = "contract_extraction.pdf";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err: any) {
      console.error("Error downloading PDF:", err);
      alert("Failed to download PDF: " + (err.message || "Unknown error"));
    } finally {
      setIsPdfLoading(false);
    }
  };

  const sections = [
    { label: "1) Entities Name & Address Details", key: "entities" },
    { label: "2) Contract Start Date & End Date", key: "dates" },
    { label: "3) Scope of Work", key: "scope" },
    { label: "4) SLA Clause", key: "sla" },
    { label: "5) Penalty Clause", key: "penalty" },
    { label: "6) Confidentiality Clause", key: "confidentiality" },
    { label: "7) Renewal & Termination Clause", key: "termination" },
    { label: "8) Commercials / Payment Terms", key: "commercials" },
    { label: "9) Risks / Assumptions", key: "risks" }
  ];

  return (
    <div className="w-full min-h-screen bg-gray-100 flex items-center justify-center px-4 py-10">
      <div className="animate-fade-in animate-slide-up w-full max-w-5xl bg-white p-10 rounded-2xl shadow-xl">
        <div className="flex justify-between items-center mb-8">
          <h2 className="text-4xl font-bold text-center text-[#8b4513]">
            Extracted Contract Information
          </h2>
          
          <div className="flex space-x-3">
            {existingExtractions.length > 0 && (
              <button
                onClick={() => navigate('/history')}
                className="flex items-center gap-2 bg-gray-200 text-gray-800 px-4 py-2 rounded-lg hover:bg-gray-300 transition-colors"
              >
                <History className="w-4 h-4" />
                View History
              </button>
            )}
            
            {extractionData && !loading && !error && (
              <button
                onClick={handleDownloadPdf}
                disabled={isPdfLoading}
                className="flex items-center gap-2 bg-[#2f5233] text-white px-4 py-2 rounded-lg hover:bg-[#1e351f] transition-colors"
              >
                {isPdfLoading ? (
                  <Loader2 className="animate-spin w-4 h-4" />
                ) : (
                  <FileDown className="w-4 h-4" />
                )}
                Download PDF
              </button>
            )}
          </div>
        </div>

        {loading ? (
          <div className="flex flex-col items-center gap-4">
            <Loader2 className="animate-spin w-10 h-10 text-blue-600" />
            <p className="text-gray-700 text-lg">
              {extractionId ? "Loading saved extraction..." : "Generating new extraction..."}
            </p>
          </div>
        ) : error ? (
          <p className="text-red-600 text-lg text-center font-medium">
            {error}
          </p>
        ) : extractionData && typeof extractionData.answer === "object" ? (
          <>
            <div className="max-h-[600px] overflow-y-auto text-gray-800 text-lg leading-relaxed p-4 bg-gray-50 border border-gray-300 rounded-lg space-y-6">
              {sections.map(({ label, key }) => {
                const content = extractionData.answer[key];
                if (!content) return null;
                
                return (
                  <div key={key}>
                    <h3 className="text-xl font-semibold text-indigo-700 mb-2 border-b pb-1">
                      {label}
                    </h3>
                    <ReactMarkdown>
                      {typeof content === "string"
                        ? content
                        : JSON.stringify(content, null, 2)}
                    </ReactMarkdown>
                  </div>
                );
              })}
            </div>

            <div className="mt-6 text-sm text-gray-500 flex justify-between">
              <p>
                {extractionId 
                  ? `Extraction #${extractionId} loaded from history` 
                  : `⏱️ Processing Time: ${extractionData.time_taken}`
                }
              </p>
              {extractionId && (
                <button 
                  onClick={fetchNewExtraction}
                  className="text-blue-500 hover:text-blue-700"
                >
                  Generate New Extraction
                </button>
              )}
            </div>
          </>
        ) : (
          <p className="text-gray-600 text-lg text-center">
            No extraction data available yet.
          </p>
        )}
      </div>
    </div>
  );
};

export default ExtractInfo;
