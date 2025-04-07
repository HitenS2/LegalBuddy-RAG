import React, { useState } from "react";
import { useNavigate } from "react-router-dom"; // Import useNavigate
import { FileUp, MessageCircle, FileSearch } from "lucide-react";

export const Welcome: React.FC = () => {
  const [isUploading, setIsUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const navigate = useNavigate(); // Initialize navigate function

  // Handle file selection
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
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
      // Get authentication token from localStorage
      const token = localStorage.getItem('token');
      if (!token) {
        alert("You must be logged in to upload files.");
        navigate('/login');
        return;
      }

      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        headers: {
          'Authorization': `Bearer ${token}`  // Add token to headers
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload file");
      }

      const data = await response.json();
      console.log("Upload Response:", data);
      alert("File uploaded successfully!");

      // ✅ Navigate to the /summary page after a successful upload
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
            </div>

            <div className="p-8 bg-[#f8f5f2] rounded-lg border-l-4 border-[#2f5233]">
              <div className="flex items-center gap-3 mb-3">
                <FileSearch className="w-8 h-8 text-[#2f5233]" />
                <h3 className="font-semibold text-[#8b4513] text-xl">Step 2</h3>
              </div>
              <h4 className="font-medium text-gray-800 mb-2 text-lg">Review Analysis</h4>
              <p className="text-gray-600 text-lg">
                Get instant insights and key information extraction.
              </p>
            </div>

            <div className="p-8 bg-[#f8f5f2] rounded-lg border-l-4 border-[#2f5233]">
              <div className="flex items-center gap-3 mb-3">
                <MessageCircle className="w-8 h-8 text-[#2f5233]" />
                <h3 className="font-semibold text-[#8b4513] text-xl">Step 3</h3>
              </div>
              <h4 className="font-medium text-gray-800 mb-2 text-lg">Get Assistance</h4>
              <p className="text-gray-600 text-lg">
                Chat with our AI to understand your document better.
              </p>
            </div>
          </div>

          <div className="mt-10 flex flex-col items-center">
            <label
              htmlFor="file-upload"
              className={`inline-flex items-center gap-3 px-10 py-5 bg-[#2f5233] text-white rounded-xl cursor-pointer hover:bg-[#1e351f] transition-colors shadow-xl text-lg ${
                isUploading ? "opacity-50 cursor-not-allowed" : ""
              }`}
            >
              <FileUp className="w-6 h-6" />
              {selectedFile ? selectedFile.name : "Select a File"}
              <input
                type="file"
                id="file-upload"
                onChange={handleFileChange}
                className="hidden"
                disabled={isUploading}
              />
            </label>

            {selectedFile && (
              <button
                onClick={handleFileUpload}
                className="mt-4 px-8 py-4 bg-blue-600 text-white rounded-lg shadow-md hover:bg-blue-700 transition"
                disabled={isUploading}
              >
                {isUploading ? "Uploading..." : "Upload File"}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Welcome;
