import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [sampleDataLoaded, setSampleDataLoaded] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedTable, setSelectedTable] = useState("ad_sales");
  const [uploadLogs, setUploadLogs] = useState([]);
  const [activeTab, setActiveTab] = useState("chat"); // 'chat', 'upload', 'history'

  // Generate session ID on component mount
  useEffect(() => {
    const newSessionId = 'session-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    setSessionId(newSessionId);
  }, []);

  // Load chat history when session changes
  useEffect(() => {
    if (sessionId) {
      loadChatHistory();
    }
  }, [sessionId]);

  // Example queries for users to try
  const exampleQueries = [
    "What are my total sales?",
    "Calculate the Return on Ad Spend (RoAS)",
    "Which product had the highest Cost Per Click (CPC)?",
    "Show me products with highest total revenue",
    "What's the average conversion rate?",
    "Which products are performing best in advertising?"
  ];

  const loadChatHistory = async () => {
    try {
      const response = await axios.get(`${API}/chat-history/${sessionId}`);
      setChatHistory(response.data.messages.reverse()); // Reverse to show oldest first
    } catch (error) {
      console.error("Error loading chat history:", error);
    }
  };

  const loadUploadLogs = async () => {
    try {
      const response = await axios.get(`${API}/upload-logs`);
      setUploadLogs(response.data.logs);
    } catch (error) {
      console.error("Error loading upload logs:", error);
    }
  };

  const loadSampleData = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/sample-data`);
      setSampleDataLoaded(true);
      alert("Sample data loaded successfully! You can now try queries.");
    } catch (error) {
      console.error("Error loading sample data:", error);
      alert("Error loading sample data: " + error.response?.data?.detail);
    } finally {
      setLoading(false);
    }
  };

  const processQuery = async () => {
    if (!query.trim()) {
      alert("Please enter a question");
      return;
    }

    try {
      setLoading(true);
      const response = await axios.post(`${API}/query`, {
        question: query,
        session_id: sessionId
      });
      setResponse(response.data);
      
      // Reload chat history to show new message
      await loadChatHistory();
      
      // Clear the input
      setQuery("");
    } catch (error) {
      console.error("Error processing query:", error);
      alert("Error: " + error.response?.data?.detail);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFile) {
      alert("Please select a CSV file");
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('table_name', selectedTable);

    try {
      setLoading(true);
      const response = await axios.post(`${API}/upload-csv`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      alert(`Successfully uploaded ${response.data.records_uploaded} records to ${response.data.table_name}`);
      setSelectedFile(null);
      
      // Reload upload logs
      await loadUploadLogs();
      
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("Upload failed: " + error.response?.data?.detail);
    } finally {
      setLoading(false);
    }
  };

  const formatResults = (results) => {
    if (!results || results.length === 0) {
      return <p className="text-gray-500">No data found</p>;
    }

    const keys = Object.keys(results[0]);
    
    return (
      <div className="overflow-x-auto">
        <table className="min-w-full bg-white border border-gray-300 rounded-lg">
          <thead className="bg-gray-50">
            <tr>
              {keys.map((key) => (
                <th key={key} className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b">
                  {key.replace(/_/g, ' ').toUpperCase()}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {results.slice(0, 10).map((row, index) => (
              <tr key={index} className="hover:bg-gray-50">
                {keys.map((key) => (
                  <td key={key} className="px-4 py-2 text-sm text-gray-900 border-b">
                    {typeof row[key] === 'number' && key.includes('date') === false 
                      ? Number(row[key]).toLocaleString() 
                      : String(row[key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {results.length > 10 && (
          <p className="text-sm text-gray-500 mt-2">Showing first 10 of {results.length} results</p>
        )}
      </div>
    );
  };

  const renderChatTab = () => (
    <div className="space-y-6">
      {/* Sample Data Section */}
      {!sampleDataLoaded && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-blue-900 mb-2">
            Get Started
          </h3>
          <p className="text-blue-700 mb-3">
            Load sample e-commerce data to test the AI query system
          </p>
          <button
            onClick={loadSampleData}
            disabled={loading}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? "Loading..." : "Load Sample Data"}
          </button>
        </div>
      )}

      {/* Example Queries */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Example Questions You Can Ask:
        </h3>
        <div className="grid md:grid-cols-2 gap-3">
          {exampleQueries.map((example, index) => (
            <button
              key={index}
              onClick={() => setQuery(example)}
              className="text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-md text-sm text-gray-700 transition-colors"
            >
              "{example}"
            </button>
          ))}
        </div>
      </div>

      {/* Query Input */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Ask Your Question
        </h3>
        <div className="flex gap-4">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., What are my total sales?"
            className="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            onKeyPress={(e) => e.key === 'Enter' && processQuery()}
          />
          <button
            onClick={processQuery}
            disabled={loading || !query.trim()}
            className="bg-green-600 text-white px-6 py-2 rounded-md hover:bg-green-700 disabled:opacity-50 font-medium"
          >
            {loading ? "Processing..." : "Ask AI"}
          </button>
        </div>
      </div>

      {/* Chat History */}
      {chatHistory.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Chat History ({chatHistory.length} messages)
          </h3>
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {chatHistory.map((message, index) => (
              <div key={index} className="border-l-4 border-blue-400 pl-4 py-2">
                <div className="text-sm text-gray-600 mb-1">
                  {new Date(message.timestamp).toLocaleString()}
                </div>
                <div className="font-medium text-gray-900 mb-2">
                  Q: {message.user_question}
                </div>
                <div className="text-green-700 bg-green-50 p-2 rounded">
                  A: {message.ai_response}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Latest Response */}
      {response && (
        <div className="space-y-6">
          {/* Human-readable Answer */}
          <div className="bg-green-50 border border-green-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-green-900 mb-3">
              AI Answer
            </h3>
            <p className="text-green-800 leading-relaxed">
              {response.human_readable_answer}
            </p>
          </div>

          {/* Generated SQL */}
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">
              Generated SQL Query
            </h3>
            <pre className="bg-gray-900 text-green-400 p-4 rounded-md overflow-x-auto text-sm">
              {response.sql_query}
            </pre>
          </div>

          {/* Raw Results */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Query Results ({response.results?.length || 0} rows)
            </h3>
            {formatResults(response.results)}
          </div>
        </div>
      )}
    </div>
  );

  const renderUploadTab = () => (
    <div className="space-y-6">
      {/* File Upload Section */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Upload Your E-commerce Data
        </h3>
        <p className="text-gray-600 mb-4">
          Upload CSV files containing your e-commerce data. The system will automatically process and import the data.
        </p>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Table Type
            </label>
            <select
              value={selectedTable}
              onChange={(e) => setSelectedTable(e.target.value)}
              className="block w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="ad_sales">Ad Sales & Metrics</option>
              <option value="total_sales">Total Sales & Metrics</option>
              <option value="product_eligibility">Product Eligibility</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Choose CSV File
            </label>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setSelectedFile(e.target.files[0])}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
          </div>

          {selectedFile && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <p className="text-blue-800">
                <strong>Selected file:</strong> {selectedFile.name} ({Math.round(selectedFile.size / 1024)} KB)
              </p>
              <p className="text-blue-700 text-sm">
                <strong>Target table:</strong> {selectedTable}
              </p>
            </div>
          )}

          <button
            onClick={handleFileUpload}
            disabled={loading || !selectedFile}
            className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50 font-medium"
          >
            {loading ? "Uploading..." : "Upload CSV"}
          </button>
        </div>
      </div>

      {/* Column Requirements */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
        <h4 className="text-lg font-semibold text-yellow-900 mb-3">
          CSV Column Requirements
        </h4>
        <div className="space-y-3 text-sm">
          <div>
            <strong className="text-yellow-800">Ad Sales & Metrics:</strong>
            <p className="text-yellow-700">date, product_id, product_name, ad_spend, clicks, impressions, ad_sales, cpc, cpm, ctr, acos, roas</p>
          </div>
          <div>
            <strong className="text-yellow-800">Total Sales & Metrics:</strong>
            <p className="text-yellow-700">date, product_id, product_name, total_sales, units_sold, price_per_unit, total_revenue, organic_sales, sessions, conversion_rate</p>
          </div>
          <div>
            <strong className="text-yellow-800">Product Eligibility:</strong>
            <p className="text-yellow-700">product_id, product_name, category, brand, is_eligible_for_ads, is_active</p>
          </div>
        </div>
      </div>

      {/* Upload Logs */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            Recent Uploads
          </h3>
          <button
            onClick={loadUploadLogs}
            className="text-blue-600 hover:text-blue-700 text-sm font-medium"
          >
            Refresh
          </button>
        </div>
        
        {uploadLogs.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2 px-3 text-sm font-medium text-gray-600">File</th>
                  <th className="text-left py-2 px-3 text-sm font-medium text-gray-600">Table</th>
                  <th className="text-left py-2 px-3 text-sm font-medium text-gray-600">Records</th>
                  <th className="text-left py-2 px-3 text-sm font-medium text-gray-600">Status</th>
                  <th className="text-left py-2 px-3 text-sm font-medium text-gray-600">Date</th>
                </tr>
              </thead>
              <tbody>
                {uploadLogs.slice(0, 10).map((log, index) => (
                  <tr key={index} className="border-b hover:bg-gray-50">
                    <td className="py-2 px-3 text-sm text-gray-900">{log.filename}</td>
                    <td className="py-2 px-3 text-sm text-gray-900">{log.table_name}</td>
                    <td className="py-2 px-3 text-sm text-gray-900">{log.records_count}</td>
                    <td className="py-2 px-3 text-sm">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        log.status === 'success' 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {log.status}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-sm text-gray-900">
                      {new Date(log.upload_timestamp).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500 text-center py-4">No uploads yet</p>
        )}
      </div>
    </div>
  );

  useEffect(() => {
    if (activeTab === 'upload') {
      loadUploadLogs();
    }
  }, [activeTab]);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900">
            E-commerce Data Query AI Agent
          </h1>
          <p className="text-gray-600 mt-2">
            Ask questions about your e-commerce data in natural language
          </p>
          <div className="text-sm text-gray-500 mt-1">
            Session: {sessionId.split('-').pop()}
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="max-w-6xl mx-auto px-4 py-4">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8" aria-label="Tabs">
            <button
              onClick={() => setActiveTab('chat')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'chat'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Chat & Query
            </button>
            <button
              onClick={() => setActiveTab('upload')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'upload'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Data Upload
            </button>
          </nav>
        </div>
      </div>

      {/* Tab Content */}
      <div className="max-w-6xl mx-auto px-4 py-8">
        {activeTab === 'chat' && renderChatTab()}
        {activeTab === 'upload' && renderUploadTab()}
      </div>
    </div>
  );
}

export default App;