import React, { useEffect, useState } from "react";
import { Loader2, Clock, FileText } from "lucide-react";
import { toast, ToastContainer } from "react-toastify";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ExtractResultProps {
  selectedFile: File | null;
}

interface ExtractData {
  answer: {
    [key: string]: string;
  };
  time_taken: string;
}

export const ExtractResult: React.FC<ExtractResultProps> = ({ selectedFile }) => {
  const [extractData, setExtractData] = useState<ExtractData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isFileUploaded, setIsFileUploaded] = useState<boolean>(false);

  useEffect(() => {
    const fetchExtract = async () => {
      if (!selectedFile) {
        setError("Please upload a document first");
        return;
      }

      if (!isFileUploaded) {
        setLoading(true);
        setError(null);

        try {
          const formData = new FormData();
          formData.append('files', selectedFile);

          const uploadResponse = await fetch("http://127.0.0.1:5000/upload", {
            method: "POST",
            body: formData,
          });

          if (!uploadResponse.ok) {
            const errorData = await uploadResponse.json();
            throw new Error(errorData.error || "Failed to upload document");
          }

          const uploadData = await uploadResponse.json();
          if (uploadData.status !== "success") {
            throw new Error(uploadData.message || "Failed to process document");
          }

          setIsFileUploaded(true);
        } catch (err: any) {
          console.error("Error:", err);
          setError(err.message || "An error occurred while uploading the document");
          setLoading(false);
          return;
        }
      }

      setLoading(true);
      try {
        const extractResponse = await fetch("http://127.0.0.1:5000/extract", {
          method: "GET",
        });

        if (!extractResponse.ok) {
          const errorData = await extractResponse.json();
          throw new Error(errorData.error || "Failed to fetch extraction");
        }

        const data: ExtractData = await extractResponse.json();
        if (data.answer) {
          setExtractData(data);
        }
      } catch (err: any) {
        console.error("Error:", err);
        setError(err.message || "An error occurred while fetching the extraction");
      } finally {
        setLoading(false);
      }
    };

    fetchExtract();
  }, [selectedFile, isFileUploaded]);

  return (
    <div className="min-h-[60vh] bg-gradient-to-b from-gray-50 to-gray-100 p-8 flex flex-col items-center">
      <div className="animate-fade-in animate-slide-up max-w-screen-lg w-full bg-white p-8 rounded-2xl shadow-lg border border-gray-200">
        <div className="flex items-center justify-center gap-2 mb-6">
          <FileText className="w-6 h-6 text-[#8b4513]" />
          <h3 className="text-2xl font-bold text-[#8b4513] text-center">
            Document Extraction
          </h3>
        </div>

        {loading ? (
          <div className="flex flex-col items-center justify-center gap-4 py-12">
            <Loader2 className="animate-spin w-8 h-8 text-[#8b4513]" />
            <p className="text-gray-600 text-lg">
              {isFileUploaded ? "Generating extraction..." : "Uploading document..."}
            </p>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center gap-4 py-12">
            <div className="text-red-600 text-lg text-center bg-red-50 p-4 rounded-lg">
              {error}
            </div>
          </div>
        ) : extractData ? (
          <div className="space-y-6">
            <div className="bg-gray-50 p-6 rounded-xl border border-gray-200 prose prose-lg max-w-none">
              {Object.entries(extractData.answer).map(([section, content]) => (
                <div key={section} className="mb-6">
                  <h3 className="text-xl font-bold text-[#8b4513] mb-3 capitalize">
                    {section.replace(/_/g, ' ')}
                  </h3>
                  <ReactMarkdown 
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({node, ...props}) => <p className="text-gray-700 leading-relaxed" {...props} />,
                      strong: ({node, ...props}) => <strong className="font-bold text-[#8b4513]" {...props} />,
                      h1: ({node, ...props}) => <h1 className="text-2xl font-bold text-[#8b4513] mt-6 mb-4" {...props} />,
                      h2: ({node, ...props}) => <h2 className="text-xl font-bold text-[#8b4513] mt-5 mb-3" {...props} />,
                      h3: ({node, ...props}) => <h3 className="text-lg font-bold text-[#8b4513] mt-4 mb-2" {...props} />,
                      ul: ({node, ...props}) => <ul className="list-disc pl-6 space-y-2 marker:text-[#8b4513]" {...props} />,
                      ol: ({node, ...props}) => <ol className="list-decimal pl-6 space-y-2 marker:text-[#8b4513]" {...props} />,
                      li: ({node, ...props}) => <li className="text-gray-700" {...props} />,
                      blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-[#8b4513] pl-4 italic bg-gray-100 py-2" {...props} />,
                    }}
                  >
                    {content}
                  </ReactMarkdown>
                </div>
              ))}
            </div>
            <div className="flex items-center justify-center gap-2 text-gray-500 text-sm">
              <Clock className="w-4 h-4" />
              <span>Processing Time: {extractData.time_taken}</span>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center gap-4 py-12">
            <p className="text-gray-600 text-lg text-center">
              Please upload a document to generate extraction.
            </p>
          </div>
        )}
      </div>
      <ToastContainer 
        position="top-right" 
        autoClose={3000}
        toastClassName="bg-white text-gray-800"
        progressClassName="bg-[#8b4513]"
      />
    </div>
  );
};

export default ExtractResult; 