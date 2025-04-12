import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { FileUp, MessageCircle, FileSearch } from "lucide-react";

interface WelcomeProps {
  onFileSelect: (file: File) => void;
}

export const Welcome: React.FC<WelcomeProps> = ({ onFileSelect }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const navigate = useNavigate();

  // Handle file selection
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      onFileSelect(file);
    }
  };

  // Handle file upload
  const handleFileUpload = async () => {
    if (!selectedFile) {
      alert("Please select a file before uploading.");
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append("files", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Failed to upload file: ${errorData.error || response.statusText}`);
      }

      const data = await response.json();
      console.log("Upload Response:", data);
      alert("File uploaded successfully!");

      // Navigate to the /summary page after a successful upload
      navigate("/summary");

    } catch (error) {
      console.error("Upload Error:", error);
      alert("Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
      setSelectedFile(null);
    }
  };

  return (
    <div className="flex justify-center items-center bg-gray-100 p-6">
      <div className="max-w-screen-lg w-full space-y-8 animate-fade-in">
        <h1 className="text-5xl font-bold text-[#8b4513] animate-slide-up text-center">
          Welcome to Legal Buddy
        </h1>
        <div className="bg-white p-12 rounded-xl shadow-xl space-y-8 animate-slide-up-delayed">
          <h2 className="text-3xl font-semibold text-[#2f5233] text-center">
            Hello there! 👋
          </h2>
          <p className="text-gray-700 text-xl text-center">
            Your AI-powered legal document assistant. Let's analyze your documents together.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="p-8 bg-[#f8f5f2] rounded-lg border-l-4 border-[#2f5233]">
              <div className="flex items-center gap-3 mb-3">
                <FileUp className="w-8 h-8 text-[#2f5233]" />
                <h3 className="font-semibold text-[#8b4513] text-xl">Step 1</h3>
              </div>
              <h4 className="font-medium text-gray-800 mb-2 text-lg">Upload Document</h4>
              <p className="text-gray-600 text-lg">
                Start by uploading your legal document for analysis.
              </p>
              <div className="mt-4">
                <input
                  type="file"
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload"
                  accept=".pdf,.docx,.txt"
                />
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer bg-[#2f5233] text-white px-4 py-2 rounded-md hover:bg-[#1e3a23] transition-colors"
                >
                  Choose File
                </label>
                {selectedFile && (
                  <p className="mt-2 text-sm text-gray-600">
                    Selected: {selectedFile.name}
                  </p>
                )}
              </div>
            </div>
            <div className="p-8 bg-[#f8f5f2] rounded-lg border-l-4 border-[#2f5233]">
              <div className="flex items-center gap-3 mb-3">
                <FileSearch className="w-8 h-8 text-[#2f5233]" />
                <h3 className="font-semibold text-[#8b4513] text-xl">Step 2</h3>
              </div>
              <h4 className="font-medium text-gray-800 mb-2 text-lg">Analyze & Extract</h4>
              <p className="text-gray-600 text-lg">
                Our AI will analyze your document and extract key information.
              </p>
            </div>
            <div className="p-8 bg-[#f8f5f2] rounded-lg border-l-4 border-[#2f5233]">
              <div className="flex items-center gap-3 mb-3">
                <MessageCircle className="w-8 h-8 text-[#2f5233]" />
                <h3 className="font-semibold text-[#8b4513] text-xl">Step 3</h3>
              </div>
              <h4 className="font-medium text-gray-800 mb-2 text-lg">Chat & Explore</h4>
              <p className="text-gray-600 text-lg">
                Ask questions and get insights about your document.
              </p>
            </div>
          </div>
          <div className="flex justify-center mt-8">
            <button
              onClick={handleFileUpload}
              disabled={!selectedFile || isUploading}
              className={`px-6 py-3 rounded-lg text-lg font-semibold ${
                !selectedFile || isUploading
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-[#2f5233] hover:bg-[#1e3a23] text-white'
              } transition-colors`}
            >
              {isUploading ? 'Uploading...' : 'Upload & Analyze'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Welcome;
