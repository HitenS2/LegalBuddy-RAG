import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { LogOut, Menu, X, FileText, MessageSquare, History, Home } from 'lucide-react';

interface User {
  id: number;
  name: string;
  email: string;
}

interface HeaderProps {
  isLoggedIn: boolean;
  onLogout: () => void;
}

const Header: React.FC<HeaderProps> = ({ isLoggedIn, onLogout }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  
  useEffect(() => {
    // Get user data from localStorage if logged in
    if (isLoggedIn) {
      const userString = localStorage.getItem('user');
      if (userString) {
        try {
          setUser(JSON.parse(userString));
        } catch (e) {
          console.error('Error parsing user data', e);
        }
      }
    }
  }, [isLoggedIn]);
  
  const handleLogout = () => {
    // Clear localStorage
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    
    // Call logout callback
    onLogout();
    
    // Redirect to login page
    navigate('/login');
  };
  
  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };
  
  // Close menu when route changes
  useEffect(() => {
    setIsMenuOpen(false);
  }, [location.pathname]);
  
  return (
    <header className="bg-white shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <Link to="/" className="flex-shrink-0 flex items-center">
              <span className="text-2xl font-bold text-[#2f5233]">LegalAI</span>
            </Link>
          </div>
          
          {/* Desktop navigation */}
          <div className="hidden md:ml-6 md:flex md:items-center md:space-x-4">
            {isLoggedIn ? (
              <>
                <Link
                  to="/"
                  className={`px-3 py-2 rounded-md text-sm font-medium ${
                    location.pathname === '/' 
                      ? 'bg-[#2f5233] text-white' 
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <Home className="inline-block h-5 w-5 mr-1" />
                  Home
                </Link>
                <Link
                  to="/extract"
                  className={`px-3 py-2 rounded-md text-sm font-medium ${
                    location.pathname === '/extract' 
                      ? 'bg-[#2f5233] text-white' 
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <FileText className="inline-block h-5 w-5 mr-1" />
                  Extract
                </Link>
                <Link
                  to="/chat"
                  className={`px-3 py-2 rounded-md text-sm font-medium ${
                    location.pathname === '/chat' 
                      ? 'bg-[#2f5233] text-white' 
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <MessageSquare className="inline-block h-5 w-5 mr-1" />
                  Chat
                </Link>
                {/* <Link
                  to="/history"
                  className={`px-3 py-2 rounded-md text-sm font-medium ${
                    location.pathname === '/history' 
                      ? 'bg-[#2f5233] text-white' 
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <History className="inline-block h-5 w-5 mr-1" />
                  History
                </Link> */}
                <div className="ml-4 relative flex-shrink-0 flex items-center">
                  <div className="px-2 py-1 rounded-md bg-gray-100">
                    <span className="text-sm text-gray-700">{user?.name || 'User'}</span>
                  </div>
                  <button
                    type="button"
                    onClick={handleLogout}
                    className="ml-4 px-3 py-2 rounded-md text-sm font-medium text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                  >
                    <LogOut className="inline-block h-5 w-5 mr-1" />
                    Logout
                  </button>
                </div>
              </>
            ) : (
              <>
                <Link
                  to="/login"
                  className={`px-3 py-2 rounded-md text-sm font-medium ${
                    location.pathname === '/login' 
                      ? 'bg-[#2f5233] text-white' 
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  Login
                </Link>
                <Link
                  to="/register"
                  className={`px-3 py-2 rounded-md text-sm font-medium ${
                    location.pathname === '/register' 
                      ? 'bg-[#2f5233] text-white' 
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  Register
                </Link>
              </>
            )}
          </div>
          
          {/* Mobile menu button */}
          <div className="flex items-center md:hidden">
            <button
              onClick={toggleMenu}
              className="inline-flex items-center justify-center p-2 rounded-md text-gray-700 hover:text-black hover:bg-gray-100 focus:outline-none"
            >
              <span className="sr-only">Open main menu</span>
              {isMenuOpen ? (
                <X className="block h-6 w-6" aria-hidden="true" />
              ) : (
                <Menu className="block h-6 w-6" aria-hidden="true" />
              )}
            </button>
          </div>
        </div>
      </div>
      
      {/* Mobile menu */}
      {isMenuOpen && (
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            {isLoggedIn ? (
              <>
                <Link
                  to="/"
                  className={`block px-3 py-2 rounded-md text-base font-medium ${
                    location.pathname === '/' 
                      ? 'bg-[#2f5233] text-white' 
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <Home className="inline-block h-5 w-5 mr-2" />
                  Home
                </Link>
                <Link
                  to="/extract"
                  className={`block px-3 py-2 rounded-md text-base font-medium ${
                    location.pathname === '/extract' 
                      ? 'bg-[#2f5233] text-white' 
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <FileText className="inline-block h-5 w-5 mr-2" />
                  Extract
                </Link>
                <Link
                  to="/chat"
                  className={`block px-3 py-2 rounded-md text-base font-medium ${
                    location.pathname === '/chat' 
                      ? 'bg-[#2f5233] text-white' 
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <MessageSquare className="inline-block h-5 w-5 mr-2" />
                  Chat
                </Link>
                <Link
                  to="/history"
                  className={`block px-3 py-2 rounded-md text-base font-medium ${
                    location.pathname === '/history' 
                      ? 'bg-[#2f5233] text-white' 
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <History className="inline-block h-5 w-5 mr-2" />
                  History
                </Link>
                <div className="mt-3 px-3 py-3 border-t border-gray-200">
                  <div className="flex items-center">
                    <div className="px-2 py-1 rounded-md bg-gray-100">
                      <span className="text-sm text-gray-700">{user?.name || 'User'}</span>
                    </div>
                    <button
                      type="button"
                      onClick={handleLogout}
                      className="ml-auto px-3 py-2 rounded-md text-sm font-medium text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                    >
                      <LogOut className="inline-block h-5 w-5 mr-1" />
                      Logout
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <>
                <Link
                  to="/login"
                  className={`block px-3 py-2 rounded-md text-base font-medium ${
                    location.pathname === '/login' 
                      ? 'bg-[#2f5233] text-white' 
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  Login
                </Link>
                <Link
                  to="/register"
                  className={`block px-3 py-2 rounded-md text-base font-medium ${
                    location.pathname === '/register' 
                      ? 'bg-[#2f5233] text-white' 
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  Register
                </Link>
              </>
            )}
          </div>
        </div>
      )}
    </header>
  );
};

export default Header; 