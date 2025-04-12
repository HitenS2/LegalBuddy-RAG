import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

// Components
import { Welcome } from './components/Welcome';
import { Summary } from './components/Summary';
import { Chat } from './components/Chat';
import { ExtractInfo } from './components/ExtractInfo';
import { ExtractResult } from './components/ExtractResult';
import { Manual } from './components/Manual';
import Header from './components/Header';

// Types
interface ChatMessage {
  id: number;
  text: string;
  timestamp: string;
  sender: 'user' | 'bot';
}

interface SummaryResponse {
  answer: string;
  time_taken: string;
}

interface ExtractResponse {
  answer: {
    [section: string]: string | object;
  };
  time_taken: string;
  status: string;
  sections_extracted?: string[];
}

// Main Layout with Header
const MainLayout: React.FC<{children: React.ReactNode}> = ({ children }) => {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <Header />
      <main className="flex-grow">
        {children}
      </main>
      <ToastContainer position="top-right" autoClose={3000} />
    </div>
  );
};

// Custom Welcome wrapper to handle file upload
const WelcomeWrapper: React.FC<{onFileSelect: (file: File) => void}> = ({ onFileSelect }) => {
  return <Welcome onFileSelect={onFileSelect} />;
};

function AppContent() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [summaryData, setSummaryData] = useState<SummaryResponse | null>(null);
  const [extractData, setExtractData] = useState<ExtractResponse | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>(() => {
    const saved = localStorage.getItem('chatHistory');
    return saved ? JSON.parse(saved) : [];
  });
  const [currentMessage, setCurrentMessage] = useState('');

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setIsUploading(true);
      try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('http://127.0.0.1:5000/upload', {
          method: 'POST',
          body: formData,
        });
  
        if (!response.ok) {
          throw new Error('Error uploading file');
        }
  
        const data = await response.json();
        setFileId(data.file_id);
        toast.success('File uploaded successfully!');
      } catch (error) {
        console.error('Upload error:', error);
        toast.error('Failed to upload file');
      } finally {
        setIsUploading(false);
      }
    }
  };

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentMessage.trim()) return;

    const newMessage: ChatMessage = {
      id: Date.now(),
      text: currentMessage,
      timestamp: new Date().toISOString(),
      sender: 'user'
    };

    setChatHistory(prev => [...prev, newMessage]);
    setCurrentMessage('');

    // Save to localStorage
    localStorage.setItem('chatHistory', JSON.stringify([...chatHistory, newMessage]));
  };

  return (
    <Router>
      <MainLayout>
        <Routes>
          <Route path="/" element={<WelcomeWrapper onFileSelect={handleFileSelect} />} />
          <Route path="/summary" element={<Summary selectedFile={selectedFile} />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/extract" element={<ExtractInfo />} />
          <Route path="/extract-result" element={<ExtractResult />} />
          <Route path="/manual" element={<Manual />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </MainLayout>
    </Router>
  );
}

function App() {
  return <AppContent />;
}

export default App;
