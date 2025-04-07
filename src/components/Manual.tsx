import React from 'react';
import { BookOpen, Upload, Search, MessageCircle, FileText } from 'lucide-react';

export const Manual: React.FC = () => {
  return (
    <div className="bg-white p-8 rounded-lg shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <BookOpen className="w-8 h-8 text-[#2f5233]" />
        <h2 className="text-2xl font-semibold text-[#8b4513]">User Manual</h2>
      </div>

      <div className="space-y-8">
        <div className="space-y-4">
          <h3 className="text-xl font-medium text-[#2f5233]">Getting Started</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-6 bg-[#f8f5f2] rounded-lg border-l-4 border-[#2f5233]">
              <div className="flex items-center gap-3 mb-3">
                <Upload className="w-6 h-6 text-[#2f5233]" />
                <h4 className="font-semibold text-[#8b4513]">1. Upload Document</h4>
              </div>
              <p className="text-gray-600">Click the upload button to select and upload your legal document</p>
            </div>
            
            <div className="p-6 bg-[#f8f5f2] rounded-lg border-l-4 border-[#2f5233]">
              <div className="flex items-center gap-3 mb-3">
                <Search className="w-6 h-6 text-[#2f5233]" />
                <h4 className="font-semibold text-[#8b4513]">2. View Summary</h4>
              </div>
              <p className="text-gray-600">Review the automatically generated document summary</p>
            </div>
            
            <div className="p-6 bg-[#f8f5f2] rounded-lg border-l-4 border-[#2f5233]">
              <div className="flex items-center gap-3 mb-3">
                <MessageCircle className="w-6 h-6 text-[#2f5233]" />
                <h4 className="font-semibold text-[#8b4513]">3. Chat with AI</h4>
              </div>
              <p className="text-gray-600">Ask questions about your document using the chat interface</p>
            </div>
            
            <div className="p-6 bg-[#f8f5f2] rounded-lg border-l-4 border-[#2f5233]">
              <div className="flex items-center gap-3 mb-3">
                <FileText className="w-6 h-6 text-[#2f5233]" />
                <h4 className="font-semibold text-[#8b4513]">4. Extract Info</h4>
              </div>
              <p className="text-gray-600">Extract and view key information from your document</p>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-xl font-medium text-[#2f5233]">Features</h3>
          <div className="space-y-4">
            <div className="p-4 bg-[#f8f5f2] rounded-lg">
              <h4 className="font-medium text-[#8b4513] mb-2">Document Summary</h4>
              <p className="text-gray-600">Get an overview of your document including key points, dates, and parties involved</p>
            </div>
            
            <div className="p-4 bg-[#f8f5f2] rounded-lg">
              <h4 className="font-medium text-[#8b4513] mb-2">Legal Assistant Chat</h4>
              <p className="text-gray-600">Interactive AI chat to answer questions about your legal documents</p>
            </div>
            
            <div className="p-4 bg-[#f8f5f2] rounded-lg">
              <h4 className="font-medium text-[#8b4513] mb-2">Information Extraction</h4>
              <p className="text-gray-600">Extract and analyze specific information from your documents</p>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-xl font-medium text-[#2f5233]">Tips</h3>
          <ul className="list-disc list-inside space-y-2 text-gray-600 ml-4">
            <li>Use clear, specific questions when chatting with the AI</li>
            <li>Review the extracted information for accuracy</li>
            <li>Keep your documents organized by company name</li>
            <li>Save important insights for future reference</li>
          </ul>
        </div>
      </div>
    </div>
  );
};