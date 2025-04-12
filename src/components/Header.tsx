import React from 'react';
import { Link } from 'react-router-dom';
import { FileText, MessageCircle, FileSearch, BookOpen } from 'lucide-react';

interface HeaderProps {
  // Remove authentication-related props
}

const Header: React.FC<HeaderProps> = () => {
  return (
    <header className="bg-white shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <Link to="/" className="text-xl font-bold text-[#8b4513]">
                Legal Buddy
              </Link>
            </div>
            <nav className="ml-6 flex space-x-8">
              <Link
                to="/"
                className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-900 hover:text-[#8b4513]"
              >
                <FileText className="h-5 w-5 mr-1" />
                Upload
              </Link>
              <Link
                to="/summary"
                className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-900 hover:text-[#8b4513]"
              >
                <FileSearch className="h-5 w-5 mr-1" />
                Summary
              </Link>
              <Link
                to="/chat"
                className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-900 hover:text-[#8b4513]"
              >
                <MessageCircle className="h-5 w-5 mr-1" />
                Chat
              </Link>
              <Link
                to="/extract"
                className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-900 hover:text-[#8b4513]"
              >
                <FileSearch className="h-5 w-5 mr-1" />
                Extract
              </Link>
              <Link
                to="/manual"
                className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-900 hover:text-[#8b4513]"
              >
                <BookOpen className="h-5 w-5 mr-1" />
                Manual
              </Link>
            </nav>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header; 