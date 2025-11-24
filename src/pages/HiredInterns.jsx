import React, { useState, useEffect } from "react";
import "./HiredInterns.css";
import Navbar from "../components/navbar/Navbar";
import axios from "axios";

const HiredInterns = () => {
  // State for pending and active interns
  const [pendingInterns, setPendingInterns] = useState([]);
  const [activeInterns, setActiveInterns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // State for search and filters
  const [pendingSearchTerm, setPendingSearchTerm] = useState("");
  const [activeSearchTerm, setActiveSearchTerm] = useState("");
  const [selectedDate, setSelectedDate] = useState("");
  
  // State for modals
  const [showStartDateModal, setShowStartDateModal] = useState(false);
  const [selectedIntern, setSelectedIntern] = useState(null);
  const [startDate, setStartDate] = useState("");
  
  useEffect(() => {
    fetchHiredInterns();
  }, []);
  
  // Get API base URL from environment variables
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://localhost:3000";
const API_TIMEOUT = process.env.REACT_APP_API_TIMEOUT || 30000;

// Configure axios instance with base URL and timeout
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: parseInt(API_TIMEOUT),
});

const fetchHiredInterns = async () => {
  setLoading(true);
  try {
    // Fetch pending interns
    const pendingResponse = await api.get("/hired-interns", {
      params: { status: "pending" }
    });
    
    setPendingInterns(pendingResponse.data.hired_interns || []);
    
    // Fetch active interns
    const activeResponse = await api.get("/hired-interns", {
      params: { status: "active" }
    });
    
    setActiveInterns(activeResponse.data.hired_interns || []);
    
  } catch (error) {
    console.error("Error fetching hired interns:", error);
    setError("Failed to load hired interns. Please try again later.");
  } finally {
    setLoading(false);
  }
};

const handleVerifyIntern = async (intern) => {
  try {
    const response = await api.post(`/hired-interns/${intern.cv_id}/verify`, {
      is_verified: true
    });
    
    // Update local state to reflect the verification
    const updatedInterns = pendingInterns.map(i => 
      i.cv_id === intern.cv_id ? { ...i, is_verified: true } : i
    );
    
    setPendingInterns(updatedInterns);
    alert(`${intern.name}'s documents have been verified.`);
    
  } catch (error) {
    console.error("Error verifying intern:", error);
    alert(`Failed to verify intern: ${error.message}`);
  }
};

const openStartDateModal = (intern) => {
  // Default to today's date
  const today = new Date().toISOString().split('T')[0];
  setStartDate(today);
  setSelectedIntern(intern);
  setShowStartDateModal(true);
};

const closeStartDateModal = () => {
  setShowStartDateModal(false);
  setSelectedIntern(null);
};

const handleAcceptIntern = async () => {
  if (!selectedIntern || !startDate) return;
  
  try {
    const response = await api.post(`/hired-interns/${selectedIntern.cv_id}/accept`, {
      start_date: startDate
    });
    
    const data = response.data;
    
    // Remove intern from pending list
    const updatedPendingInterns = pendingInterns.filter(i => i.cv_id !== selectedIntern.cv_id);
    setPendingInterns(updatedPendingInterns);
    
    // Add intern to active list with start/end dates
    const acceptedIntern = {
      ...selectedIntern,
      is_accepted: true,
      start_date: startDate,
      end_date: data.end_date
    };
    
    setActiveInterns([...activeInterns, acceptedIntern]);
    
    alert(`${selectedIntern.name} has been accepted to start on ${startDate}.`);
    closeStartDateModal();
    
  } catch (error) {
    console.error("Error accepting intern:", error);
    alert(`Failed to accept intern: ${error.response?.data?.detail || error.message}`);
  }
};

const handleSearchPending = (e) => {
  setPendingSearchTerm(e.target.value);
};

const handleSearchActive = (e) => {
  setActiveSearchTerm(e.target.value);
};

const handleDateFilter = async (e) => {
  const date = e.target.value;
  setSelectedDate(date);
  
  if (date) {
    try {
      setLoading(true);
      
      // Fetch active interns for the selected date
      const response = await api.get("/hired-interns", {
        params: { 
          status: "active", 
          date: date 
        }
      });
      
      setActiveInterns(response.data.hired_interns || []);
      
    } catch (error) {
      console.error("Error filtering by date:", error);
      setError("Failed to filter interns by date.");
    } finally {
      setLoading(false);
    }
  } else {
    // If no date is selected, fetch all active interns again
    fetchHiredInterns();
  }
};
  
  // Filter interns based on search terms
  const filteredPendingInterns = pendingSearchTerm 
    ? pendingInterns.filter(intern => 
        intern.name.toLowerCase().includes(pendingSearchTerm.toLowerCase())
      )
    : pendingInterns;
  
  const filteredActiveInterns = activeSearchTerm 
    ? activeInterns.filter(intern => 
        intern.name.toLowerCase().includes(activeSearchTerm.toLowerCase())
      )
    : activeInterns;
  
  return (
    <div className="hired-interns-page">
      <Navbar />
      
      <div className="content-container">
        <h1 className="page-title">Hired Interns Management</h1>
        
        {loading ? (
          <div className="loading-container">
            <div className="loader"></div>
            <p>Loading interns data...</p>
          </div>
        ) : error ? (
          <div className="error-container">
            <p className="error-message">{error}</p>
          </div>
        ) : (
          <div className="tables-container">
            {/* Pending Interns Table */}
            <div className="table-section">
              <div className="section-header">
                <h2>Pending Document Verification</h2>
                <div className="search-container">
                  <input
                    type="text"
                    placeholder="Search by intern name..."
                    value={pendingSearchTerm}
                    onChange={handleSearchPending}
                    className="search-input"
                  />
                </div>
              </div>
              
              {filteredPendingInterns.length === 0 ? (
                <div className="empty-state">
                  <p>No pending interns found.</p>
                </div>
              ) : (
                <div className="table-responsive">
                  <table className="interns-table">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Education</th>
                        <th>University</th>
                        <th>Contact</th>
                        <th>Internship Period</th>
                        <th>Document Deadline</th>
                        <th>Status</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredPendingInterns.map((intern) => (
                        <tr key={intern.cv_id}>
                          <td>{intern.name}</td>
                          <td>{intern.degree}</td>
                          <td>{intern.university}</td>
                          <td>{intern.contact_no}</td>
                          <td>{intern.internship_period}</td>
                          <td>{intern.document_deadline}</td>
                          <td>
                            <span className={`status-badge ${intern.is_verified ? "verified" : "pending"}`}>
                              {intern.is_verified ? "Verified" : "Awaiting Documents"}
                            </span>
                          </td>
                          <td>
                            {!intern.is_verified ? (
                              <button 
                                className="verify-button"
                                onClick={() => handleVerifyIntern(intern)}
                              >
                                Verify Documents
                              </button>
                            ) : (
                              <button 
                                className="accept-button"
                                onClick={() => openStartDateModal(intern)}
                              >
                                Set Start Date
                              </button>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
            
            {/* Active Interns Table */}
            <div className="table-section">
              <div className="section-header">
                <h2>Active Interns</h2>
                <div className="filter-container">
                  <div className="search-container">
                    <input
                      type="text"
                      placeholder="Search by intern name..."
                      value={activeSearchTerm}
                      onChange={handleSearchActive}
                      className="search-input"
                    />
                  </div>
                  <div className="date-filter">
                    <input
                      type="date"
                      value={selectedDate}
                      onChange={handleDateFilter}
                      className="date-input"
                      placeholder="Filter by start date"
                    />
                    {selectedDate && (
                      <button 
                        className="clear-filter"
                        onClick={() => {
                          setSelectedDate("");
                          fetchHiredInterns();
                        }}
                      >
                        Clear
                      </button>
                    )}
                  </div>
                </div>
              </div>
              
              {filteredActiveInterns.length === 0 ? (
                <div className="empty-state">
                  <p>No active interns found.</p>
                </div>
              ) : (
                <div className="table-responsive">
                  <table className="interns-table">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Education</th>
                        <th>University</th>
                        <th>Contact</th>
                        <th>Internship Period</th>
                        <th>Start Date</th>
                        <th>End Date</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredActiveInterns.map((intern) => (
                        <tr key={intern.cv_id}>
                          <td>{intern.name}</td>
                          <td>{intern.degree}</td>
                          <td>{intern.university}</td>
                          <td>{intern.contact_no}</td>
                          <td>{intern.internship_period}</td>
                          <td>{intern.start_date}</td>
                          <td>{intern.end_date}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
      
      {/* Start Date Modal */}
      {showStartDateModal && (
        <div className="modal-overlay">
          <div className="start-date-modal">
            <h2>Set Start Date</h2>
            <p>Select a start date for {selectedIntern?.name}'s internship:</p>
            
            <div className="date-input-container">
              <input 
                type="date" 
                value={startDate} 
                onChange={(e) => setStartDate(e.target.value)}
                min={new Date().toISOString().split('T')[0]} // Can't select dates in the past
                className="date-input"
              />
            </div>
            
            <div className="modal-buttons">
              <button className="cancel-button" onClick={closeStartDateModal}>
                Cancel
              </button>
              <button 
                className="confirm-button" 
                onClick={handleAcceptIntern}
                disabled={!startDate}
              >
                Confirm & Accept
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HiredInterns;