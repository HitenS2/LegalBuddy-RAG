import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText, MessageSquare, Calendar, Download, Trash2, ExternalLink } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { format } from 'date-fns';

interface ChatSession {
  id: number;
  created_at: string;
  preview: string;
  message_count: number;
}

interface Extraction {
  id: number;
  created_at: string;
  document_id: number;
  document_name: string;
}

// New interface for grouped extractions
interface DocumentWithExtractions {
  document_id: number;
  document_name: string;
  extractions: Extraction[];
}

const History: React.FC = () => {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<'chats' | 'extractions'>('chats');
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [extractions, setExtractions] = useState<Extraction[]>([]);
  const [groupedExtractions, setGroupedExtractions] = useState<DocumentWithExtractions[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Redirect to login if not authenticated
    if (!isAuthenticated) {
      navigate('/login');
      return;
    }

    // Fetch data based on active tab
    fetchData(activeTab);
  }, [isAuthenticated, activeTab, navigate]);

  // Group extractions by document
  useEffect(() => {
    if (extractions.length > 0) {
      const groupedByDoc: Record<number, DocumentWithExtractions> = {};
      
      extractions.forEach(extraction => {
        if (!groupedByDoc[extraction.document_id]) {
          groupedByDoc[extraction.document_id] = {
            document_id: extraction.document_id,
            document_name: extraction.document_name,
            extractions: []
          };
        }
        groupedByDoc[extraction.document_id].extractions.push(extraction);
      });
      
      // Convert to array and sort by most recent extraction
      const grouped = Object.values(groupedByDoc)
        .sort((a, b) => {
          const aLatest = a.extractions.sort((x, y) => 
            new Date(y.created_at).getTime() - new Date(x.created_at).getTime()
          )[0].created_at;
          const bLatest = b.extractions.sort((x, y) => 
            new Date(y.created_at).getTime() - new Date(x.created_at).getTime()
          )[0].created_at;
          return new Date(bLatest).getTime() - new Date(aLatest).getTime();
        });
      
      setGroupedExtractions(grouped);
    } else {
      setGroupedExtractions([]);
    }
  }, [extractions]);

  const fetchData = async (tab: 'chats' | 'extractions') => {
    setLoading(true);
    setError(null);
    
    try {
      const token = localStorage.getItem('token');
      if (!token) throw new Error('No authentication token');
      
      const endpoint = tab === 'chats' ? 'chat/sessions' : 'extractions';
      console.log(`Fetching ${tab} from endpoint: ${endpoint}`);
      
      const response = await fetch(`http://127.0.0.1:5000/${endpoint}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch data');
      }
      
      const data = await response.json();
      console.log(`Received ${tab} data:`, data);
      
      if (tab === 'chats') {
        setChatSessions(data.sessions || []);
        console.log('Updated chat sessions:', data.sessions);
      } else {
        setExtractions(data.extractions || []);
        console.log('Updated extractions:', data.extractions);
      }
    } catch (err) {
      setError('Error loading data. Please try again.');
      console.error(`Error fetching ${tab}:`, err);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteChatSession = async (sessionId: number) => {
    if (!confirm('Are you sure you want to delete this chat session?')) return;
    
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`http://127.0.0.1:5000/chat/sessions/${sessionId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) throw new Error('Failed to delete chat session');
      
      // Refresh data
      setChatSessions(prevSessions => 
        prevSessions.filter(session => session.id !== sessionId)
      );
    } catch (err) {
      console.error('Error deleting chat session:', err);
      alert('Failed to delete chat session. Please try again.');
    }
  };

  const handleDeleteExtraction = async (extractionId: number) => {
    if (!confirm('Are you sure you want to delete this extraction?')) return;
    
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`http://127.0.0.1:5000/extractions/${extractionId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) throw new Error('Failed to delete extraction');
      
      // Refresh data
      setExtractions(prevExtractions => 
        prevExtractions.filter(extraction => extraction.id !== extractionId)
      );
    } catch (err) {
      console.error('Error deleting extraction:', err);
      alert('Failed to delete extraction. Please try again.');
    }
  };

  const downloadChatHistory = async (sessionId: number) => {
    try {
      const token = localStorage.getItem('token');
      
      // Create a temporary anchor element for download
      const link = document.createElement('a');
      link.href = `http://127.0.0.1:5000/download/chat-history?session_id=${sessionId}&token=${token}`;
      link.setAttribute('download', `chat-history-${sessionId}.pdf`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err) {
      console.error('Error downloading chat history:', err);
      alert('Failed to download chat history. Please try again.');
    }
  };

  const formatDate = (dateString: string) => {
    try {
      return format(new Date(dateString), 'MMM d, yyyy h:mm a');
    } catch (e) {
      return dateString;
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Debug info */}
      {console.log('Rendering History, activeTab:', activeTab)}
      {console.log('Current extractions:', extractions)}
      {console.log('Current chatSessions:', chatSessions)}
      {console.log('Grouped extractions:', groupedExtractions)}
      
      <h1 className="text-3xl font-bold mb-6">Your History</h1>
      
      <div className="mb-6">
        <div className="flex border-b border-gray-200">
          <button
            className={`px-4 py-2 font-medium text-sm ${
              activeTab === 'chats'
                ? 'border-b-2 border-[#2f5233] text-[#2f5233]'
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab('chats')}
          >
            <MessageSquare className="inline-block h-5 w-5 mr-2" />
            Chat Sessions
          </button>
          <button
            className={`px-4 py-2 font-medium text-sm ${
              activeTab === 'extractions'
                ? 'border-b-2 border-[#2f5233] text-[#2f5233]'
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab('extractions')}
          >
            <FileText className="inline-block h-5 w-5 mr-2" />
            Document Extractions
          </button>
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center items-center p-8">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[#2f5233]"></div>
        </div>
      ) : error ? (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      ) : (
        <>
          {/* Chat Sessions Tab */}
          {activeTab === 'chats' && (
            <div>
              {chatSessions.length === 0 ? (
                <div className="text-center p-8 bg-gray-50 rounded-lg">
                  <MessageSquare className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-gray-600">You don't have any chat sessions yet.</p>
                  <button 
                    onClick={() => navigate('/chat')}
                    className="mt-4 px-4 py-2 bg-[#2f5233] text-white rounded-md hover:bg-[#1e351f]"
                  >
                    Start a new chat
                  </button>
                </div>
              ) : (
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {chatSessions.map((session) => (
                    <div key={session.id} className="bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
                      <div className="p-4">
                        <div className="flex items-start justify-between">
                          <div className="flex items-center mb-2">
                            <MessageSquare className="h-5 w-5 text-[#2f5233] mr-2" />
                            <h3 className="font-medium">Chat Session #{session.id}</h3>
                          </div>
                          <div className="flex items-center text-xs text-gray-500">
                            <Calendar className="h-4 w-4 mr-1" />
                            <span>{formatDate(session.created_at)}</span>
                          </div>
                        </div>
                        
                        <p className="text-gray-600 text-sm mb-3 line-clamp-2">{session.preview}</p>
                        
                        <div className="text-xs text-gray-500 mb-4">
                          {session.message_count} message{session.message_count !== 1 ? 's' : ''}
                        </div>
                        
                        <div className="flex justify-between">
                          <button
                            onClick={() => navigate(`/chat?session=${session.id}`)}
                            className="inline-flex items-center px-3 py-1.5 text-xs bg-gray-100 hover:bg-gray-200 rounded-md"
                          >
                            <ExternalLink className="h-4 w-4 mr-1" />
                            Open
                          </button>
                          
                          <div className="flex space-x-2">
                            <button
                              onClick={() => downloadChatHistory(session.id)}
                              className="inline-flex items-center px-3 py-1.5 text-xs bg-blue-50 text-blue-600 hover:bg-blue-100 rounded-md"
                            >
                              <Download className="h-4 w-4 mr-1" />
                              Download
                            </button>
                            
                            <button
                              onClick={() => handleDeleteChatSession(session.id)}
                              className="inline-flex items-center px-3 py-1.5 text-xs bg-red-50 text-red-600 hover:bg-red-100 rounded-md"
                            >
                              <Trash2 className="h-4 w-4 mr-1" />
                              Delete
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Extractions Tab - Grouped by Document */}
          {activeTab === 'extractions' && (
            <div>
              {extractions.length === 0 ? (
                <div className="text-center p-8 bg-gray-50 rounded-lg">
                  <FileText className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-gray-600">You don't have any document extractions yet.</p>
                  <button 
                    onClick={() => navigate('/extract')}
                    className="mt-4 px-4 py-2 bg-[#2f5233] text-white rounded-md hover:bg-[#1e351f]"
                  >
                    Extract a document
                  </button>
                </div>
              ) : (
                <div className="space-y-8">
                  {groupedExtractions.map((docGroup) => (
                    <div key={docGroup.document_id} className="bg-white rounded-lg border border-gray-200 shadow-md p-4">
                      <div className="flex items-center mb-4">
                        <FileText className="h-6 w-6 text-[#2f5233] mr-2" />
                        <h2 className="text-xl font-semibold">{docGroup.document_name}</h2>
                      </div>
                      
                      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                        {docGroup.extractions.map((extraction) => (
                          <div key={extraction.id} className="border border-gray-200 rounded-lg p-3 hover:border-[#2f5233] transition-colors">
                            <div className="flex items-center justify-between mb-2">
                              <h3 className="font-medium">Extraction #{extraction.id}</h3>
                              <div className="text-xs text-gray-500">
                                {formatDate(extraction.created_at)}
                              </div>
                            </div>
                            
                            <div className="flex justify-between mt-4">
                              <button
                                onClick={() => navigate(`/extract/result?extraction=${extraction.id}`)}
                                className="inline-flex items-center px-3 py-1.5 text-xs bg-gray-100 hover:bg-gray-200 rounded-md"
                              >
                                <ExternalLink className="h-4 w-4 mr-1" />
                                View
                              </button>
                              
                              <div className="flex space-x-2">
                                <button
                                  onClick={() => window.open(`http://127.0.0.1:5000/download/extraction?extraction_id=${extraction.id}&token=${localStorage.getItem('token')}`)}
                                  className="inline-flex items-center px-3 py-1.5 text-xs bg-blue-50 text-blue-600 hover:bg-blue-100 rounded-md"
                                >
                                  <Download className="h-4 w-4 mr-1" />
                                  Download
                                </button>
                                
                                <button
                                  onClick={() => handleDeleteExtraction(extraction.id)}
                                  className="inline-flex items-center px-3 py-1.5 text-xs bg-red-50 text-red-600 hover:bg-red-100 rounded-md"
                                >
                                  <Trash2 className="h-4 w-4 mr-1" />
                                  Delete
                                </button>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default History; 