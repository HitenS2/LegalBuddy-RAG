import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

// Components
import { Welcome } from './components/Welcome';
import { Summary } from './components/Summary';
import { Chat } from './components/Chat';
import { ExtractInfo } from './components/ExtractInfo';
import { ExtractResult } from './components/ExtractResult';
import { Manual } from './components/Manual';
import Login from './components/Login';
import Register from './components/Register';
import Header from './components/Header';
import History from './components/History';
import ProtectedRouteWrapper from './components/ProtectedRoute';

// Context
import { AuthProvider, useAuth } from './contexts/AuthContext';

// Types
interface ChatMessage {
  id: number;
  text: string;
  timestamp: string;
  sender: 'user' | 'bot';
}

interface SummaryResponse {
  // Define your summary response type here
  answer: string;
  time_taken: string;
}

interface ExtractResponse {
  // Define your extract response type here
  answer: {
    [section: string]: string | object;
  };
  time_taken: string;
  status: string;
  sections_extracted?: string[];
}

// Main Layout with Header
const MainLayout: React.FC<{children: React.ReactNode}> = ({ children }) => {
  const { isAuthenticated, logout } = useAuth();
  
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <Header isLoggedIn={isAuthenticated} onLogout={logout} />
      <main className="flex-grow">
        {children}
      </main>
      <ToastContainer position="top-right" autoClose={3000} />
    </div>
  );
};

// Custom Welcome wrapper to handle file upload
const WelcomeWrapper: React.FC = () => {
  // The actual Welcome component doesn't take props, so we just render it directly
  return <Welcome />;
};

function AppContent() {
  const { isAuthenticated } = useAuth();
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

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setIsUploading(true);
      try {
        const formData = new FormData();
        formData.append('file', file);
        
        const token = localStorage.getItem('token');
        if (!token) throw new Error('Authentication required');
  
        const response = await fetch('http://127.0.0.1:5000/upload', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`
          },
          body: formData,
        });
  
        if (!response.ok) {
          throw new Error('Error uploading file');
        }
  
        const uploadResponse = await response.json();
        setSelectedFile(file);
        setFileId(uploadResponse.document_id || uploadResponse.fileId);
  
        // The backend now automatically stores extractions for the user
        // No need to immediately call these APIs unless needed for display
        // We'll fetch them when navigating to the respective pages
        
        toast.success('File uploaded successfully!');
      } catch (error) {
        console.error('Error uploading file:', error);
        toast.error('Error uploading file. Please try again.');
      } finally {
        setIsUploading(false);
      }
    }
  };

  const handleLoginSuccess = () => {
    // Refresh authentication state
    toast.success('Logged in successfully!');
  };

  const handleRegisterSuccess = () => {
    toast.success('Account created successfully!');
  };
  
  // Handle sending chat messages
  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (currentMessage.trim()) {
      setChatHistory([
        ...chatHistory,
        {
          id: Date.now(),
          text: currentMessage,
          timestamp: new Date().toLocaleTimeString(),
          sender: 'user'
        }
      ]);
      setCurrentMessage('');
    }
  };
  
  return (
    <MainLayout>
      <Routes>
        {/* Public routes */}
        <Route 
          path="/login" 
          element={
            isAuthenticated ? (
              <Navigate to="/" replace />
            ) : (
              <Login onLoginSuccess={handleLoginSuccess} />
            )
          } 
        />
        <Route 
          path="/register" 
          element={
            isAuthenticated ? (
              <Navigate to="/" replace />
            ) : (
              <Register onRegisterSuccess={handleRegisterSuccess} />
            )
          } 
        />

        {/* Protected routes */}
        <Route 
          path="/" 
          element={
            <ProtectedRouteWrapper>
              <WelcomeWrapper />
            </ProtectedRouteWrapper>
          }
        />
        <Route
          path="/summary"
          element={
            <ProtectedRouteWrapper>
              <Summary selectedFile={selectedFile} />
            </ProtectedRouteWrapper>
          }
        />
        <Route
          path="/chat"
          element={
            <ProtectedRouteWrapper>
              <Chat fileId={fileId} />
            </ProtectedRouteWrapper>
          }
        />
        <Route
          path="/extract"
          element={
            <ProtectedRouteWrapper>
              <ExtractInfo selectedFile={selectedFile} />
            </ProtectedRouteWrapper>
          }
        />
        <Route
          path="/extract/result"
          element={
            <ProtectedRouteWrapper>
              <ExtractResult />
            </ProtectedRouteWrapper>
          }
        />
        <Route
          path="/history"
          element={
            <ProtectedRouteWrapper>
              <History />
            </ProtectedRouteWrapper>
          }
        />
        <Route 
          path="/manual" 
          element={
            <ProtectedRouteWrapper>
              <Manual />
            </ProtectedRouteWrapper>
          } 
        />

        {/* Fallback route */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </MainLayout>
  );
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;
