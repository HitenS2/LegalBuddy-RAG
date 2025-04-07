import React, { useRef, useState, useEffect } from "react";
import { Send, Bot, User, FileDown, Loader2, Download, History } from "lucide-react";
import { useNavigate, useLocation } from "react-router-dom";

interface ChatMessage {
  id: number;
  text: string;
  timestamp: string;
  sender: "user" | "bot";
}

interface ChatProps {
  fileId: string | null;
}

function formatBotResponse(text: string): string {
  if (!text) return '';

  let formatted = text;

  // Bold using **text**
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Bullets: convert lines starting with - or • into list items
  formatted = formatted.replace(/(?:^|\n)[\-\•]\s+(.*?)(?=\n|$)/g, '<li>$1</li>');

  // Wrap list items in <ul>
  if (formatted.includes('<li>')) {
    formatted = `<ul class="list-disc pl-6 space-y-1">${formatted}</ul>`;
  }

  // Line breaks
  formatted = formatted.replace(/\n/g, '<br>');

  return formatted;
}

export const Chat: React.FC<ChatProps> = ({ fileId }) => {
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState("");
  const [isQuerying, setIsQuerying] = useState(false);
  const [isPdfLoading, setIsPdfLoading] = useState(false);
  const [isHistoryPdfLoading, setIsHistoryPdfLoading] = useState(false);
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const location = useLocation();

  // Get the token from localStorage
  const token = localStorage.getItem('token');

  // Load chat session from URL parameter if present
  useEffect(() => {
    const queryParams = new URLSearchParams(location.search);
    const sessionParam = queryParams.get('session');
    
    if (sessionParam && token) {
      const sessionIdNum = parseInt(sessionParam, 10);
      setSessionId(sessionIdNum);
      loadChatHistory(sessionIdNum);
    }
  }, [location.search, token]);

  // Load chat history for a given session
  const loadChatHistory = async (sessionIdToLoad: number) => {
    if (!token) return;
    
    setIsLoadingHistory(true);
    try {
      const response = await fetch(`http://127.0.0.1:5000/chat/sessions/${sessionIdToLoad}`, {
        headers: {
          "Authorization": `Bearer ${token}`
        }
      });
      
      if (!response.ok) {
        throw new Error("Failed to load chat history");
      }
      
      const data = await response.json();
      console.log('Loaded chat history:', data);
      
      if (data.messages && Array.isArray(data.messages)) {
        // Transform messages to the format expected by our component
        const formattedMessages: ChatMessage[] = data.messages.map((msg: any) => ({
          id: msg.id,
          text: msg.content,
          timestamp: new Date(msg.timestamp).toLocaleTimeString(),
          sender: msg.is_user ? "user" : "bot"
        }));
        
        setChatHistory(formattedMessages);
      }
    } catch (err) {
      console.error("Error loading chat history:", err);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  // Automatically scroll to the bottom when chatHistory updates.
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop =
        chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentMessage.trim()) return;
    if (!token) {
      alert("Please login to use the chat functionality");
      // Redirect to login page
      return;
    }

    // Create and append the user message.
    const userMessage: ChatMessage = {
      id: Date.now(),
      text: currentMessage,
      timestamp: new Date().toLocaleTimeString(),
      sender: "user",
    };
    setChatHistory((prev) => [...prev, userMessage]);
    setCurrentMessage("");
    setIsQuerying(true);

    try {
      // Hit the /ask endpoint with the user's query.
      const response = await fetch("http://127.0.0.1:5000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ 
          query: userMessage.text,
          session_id: sessionId
        }),
      });

      // If response is not OK, alert the error message.
      if (!response.ok) {
        alert(`Error: ${response.statusText}`);
        return;
      }

      const data = await response.json();
      
      // Update session ID if this is first message
      if (!sessionId && data.session_id) {
        setSessionId(data.session_id);
      }

      // Create the bot message with the API's answer.
      const botMessage: ChatMessage = {
        id: Date.now() + 1,
        text: data.answer,
        timestamp: new Date().toLocaleTimeString(),
        sender: "bot",
      };
      setChatHistory((prev) => [...prev, botMessage]);
    } catch (error: any) {
      alert(`Error querying the document: ${error.message}`);
      console.error(error);
    } finally {
      setIsQuerying(false);
    }
  };

  const handleDownloadPdf = async () => {
    if (chatHistory.length < 2) return;
    
    // Get the last user question and bot answer
    const lastBotMessage = [...chatHistory].reverse().find(msg => msg.sender === 'bot');
    const lastUserMessage = [...chatHistory].reverse().find(msg => msg.sender === 'user');
    
    if (!lastBotMessage || !lastUserMessage) return;
    
    setIsPdfLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:5000/download/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          question: lastUserMessage.text,
          answer: lastBotMessage.text
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate PDF");
      }

      // Create a blob from the PDF Stream
      const blob = await response.blob();
      // Create a link element, set the download attribute and click it
      const link = document.createElement('a');
      link.href = window.URL.createObjectURL(blob);
      link.download = "contract_qa.pdf";
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
  
  const handleDownloadFullHistory = async () => {
    if (!sessionId || !token) {
      alert("No chat session available");
      return;
    }
    
    setIsHistoryPdfLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:5000/download/chat-history", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ 
          session_id: sessionId
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate PDF");
      }

      // Create a blob from the PDF Stream
      const blob = await response.blob();
      // Create a link element, set the download attribute and click it
      const link = document.createElement('a');
      link.href = window.URL.createObjectURL(blob);
      link.download = `chat_history_${sessionId}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err: any) {
      console.error("Error downloading chat history PDF:", err);
      alert("Failed to download chat history: " + (err.message || "Unknown error"));
    } finally {
      setIsHistoryPdfLoading(false);
    }
  };

  return (
    <div className="flex  bg-gray-100 p-6  ">
      <div className="animate-fade-in animate-slide-up w-screen max-w-screen-lg  h-[calc(100vh-10rem)] bg-white rounded-xl shadow-xl flex flex-col">
        <div className="bg-[#2f5233] text-white p-6 rounded-t-xl flex justify-between items-center">
          <div>
            <h2 className="text-2xl font-bold flex items-center gap-3">
              <Bot className="w-7 h-7" />
              {sessionId ? `Chat Session #${sessionId}` : 'Legal Assistant Chat'}
            </h2>
            <p className="text-md text-gray-200">
              Ask questions about your legal documents
            </p>
          </div>
          
          {chatHistory.length > 0 && (
            <div className="flex gap-3">
              <button
                onClick={handleDownloadPdf}
                disabled={isPdfLoading}
                className="flex items-center gap-2 bg-white text-[#2f5233] px-4 py-2 rounded-lg hover:bg-gray-100 transition-colors"
                title="Download last question and answer"
              >
                {isPdfLoading ? (
                  <Loader2 className="animate-spin w-4 h-4" />
                ) : (
                  <FileDown className="w-4 h-4" />
                )}
                Last Q&A
              </button>
              
              {sessionId && (
                <button
                  onClick={handleDownloadFullHistory}
                  disabled={isHistoryPdfLoading}
                  className="flex items-center gap-2 bg-white text-[#2f5233] px-4 py-2 rounded-lg hover:bg-gray-100 transition-colors"
                  title="Download complete chat history"
                >
                  {isHistoryPdfLoading ? (
                    <Loader2 className="animate-spin w-4 h-4" />
                  ) : (
                    <History className="w-4 h-4" />
                  )}
                  Full History
                </button>
              )}
            </div>
          )}
        </div>
        <div
          ref={chatContainerRef}
          className="flex-1 overflow-y-auto p-6 space-y-5"
        >
          {isLoadingHistory ? (
            <div className="flex justify-center items-center h-full">
              <Loader2 className="w-8 h-8 animate-spin text-[#2f5233]" />
              <span className="ml-2 text-gray-600">Loading chat history...</span>
            </div>
          ) : chatHistory.length === 0 ? (
            <div className="flex justify-center items-center h-full text-gray-500">
              <p>No messages yet. Start a conversation!</p>
            </div>
          ) : (
            chatHistory.map((message) => (
              <div
                key={message.id}
                className={`flex ${
                  message.sender === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[75%] p-5 rounded-xl flex items-start gap-4 ${
                    message.sender === "user"
                      ? "bg-[#2f5233] text-white"
                      : "bg-[#f8f5f2] text-gray-800"
                  }`}
                >
                  {message.sender === "user" ? (
                    <User className="w-6 h-6 mt-1" />
                  ) : (
                    <Bot className="w-6 h-6 mt-1" />
                  )}
                  <div>
                    <div
                      className="text-base whitespace-pre-wrap"
                      dangerouslySetInnerHTML={{
                        __html: formatBotResponse(message.text),
                      }}
                    ></div>

                    <p
                      className={`text-xs mt-1 ${
                        message.sender === "user"
                          ? "text-gray-300"
                          : "text-gray-500"
                      }`}
                    >
                      {message.timestamp}
                    </p>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
        <div className="border-t border-gray-200 p-5 bg-white rounded-b-xl">
          <form
            onSubmit={handleSendMessage}
            className="flex items-center gap-4"
          >
            <input
              type="text"
              value={currentMessage}
              onChange={(e) => setCurrentMessage(e.target.value)}
              placeholder="Ask about your document..."
              className="flex-1 p-4 border border-gray-300 rounded-lg text-lg focus:outline-none focus:ring-2 focus:ring-[#2f5233]"
            />
            <button
              type="submit"
              disabled={!currentMessage.trim() || isQuerying}
              className="px-7 py-4 bg-[#2f5233] text-white rounded-lg hover:bg-[#1e351f] transition-colors flex items-center gap-2 text-lg"
            >
              <Send className="w-5 h-5" />
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Chat;
