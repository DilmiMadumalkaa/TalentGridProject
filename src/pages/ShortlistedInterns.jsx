import "./ShortlistedInterns.css";
import React, { useState, useEffect, useRef } from "react";
import Navbar from "../components/navbar/Navbar";
import axios from "axios";

const ShortlistedInterns = () => {
  const [interns, setInterns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showEmailModal, setShowEmailModal] = useState(false);
  const [selectedIntern, setSelectedIntern] = useState(null);
  const [selectedDate, setSelectedDate] = useState("");
  const [emailSubject, setEmailSubject] = useState(
    "Internship Confirmation and Reporting Instructions – SLT Digital Platforms Development Section"
  );
  const [emailBody, setEmailBody] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [defaultFileAttached, setDefaultFileAttached] = useState(true);
  const fileInputRef = useRef(null);

  // Default email template
  const defaultEmailBody = `Dear [Intern Name],
**Congratulations!**
You have been selected for a **six-month Software Internship** with the **SLT – Digital Platforms Development Section Refno :SLTDPDS-20250520-08**.
We are excited to welcome you to Sri Lanka Telecom. Please read the following instructions carefully to ensure a smooth onboarding process.
 
**Reporting Details:**
**Location:** SLT Head Office, Lotus Road, Colombo 01 
**Date:** [SELECTED_DATE] 
**Time:** 9:30 AM
 
**Documents to Bring (Compulsory for Registration):**
Please bring **all** of the following documents. **Incomplete documentation will not be accepted**, and you will be required to visit the following **Tuesday instead**.
1. **Printed copy of this email**
2. **Your latest CV**
3. **Internship request letter** from your university *(Addressed to: Engineer - Talent Development Section, Sri Lanka Telecom. Must specify a six-month internship period.)*
4. **Police Report**
5. **Signed Trainee Guidelines document**
6. **Photocopy of your National Identity Card (NIC)**
 
**Visitor Entry Pass**
To arrange your visitor entry pass, please send the following details **within today**:
* Full Name
* NIC Number
* Contact Number
* Laptop Serial Number
Please reply to this email with the required information.
**Important Notice**
You are **required to visit SLT at least once a week** during your internship. If you are unable to meet this commitment, we kindly request that you **do not proceed** with this internship opportunity.
 
**Trainee Guidelines**
Please read the Trainee Guidelines document thoroughly. If you agree with the terms and conditions, you may proceed with confirming your participation in the internship.
 
We look forward to having you on board and wish you a rewarding internship experience at SLT!
Best Regards,
**Gayal Jayawardana** 
Digital Platform Development Section 
SLTMobitel`;

  // Calculate default date (next Tuesday)
  const getDefaultDate = () => {
    const today = new Date();
    const daysUntilNextTuesday = (9 - today.getDay()) % 7; // Tuesday is day 2
    const nextTuesday = new Date(today);
    nextTuesday.setDate(today.getDate() + daysUntilNextTuesday);
    return nextTuesday.toISOString().split('T')[0]; // Format as YYYY-MM-DD
  };

  useEffect(() => {
    fetchShortlistedInterns();
  }, []);

  useEffect(() => {
    // When modal is opened, set default values
    if (showEmailModal && selectedIntern) {
      setSelectedDate(getDefaultDate());
      // Format the email body with the intern's name
      const formattedBody = defaultEmailBody
        .replace("[Intern Name]", selectedIntern.name)
        .replace("[SELECTED_DATE]", new Date(getDefaultDate()).toLocaleDateString('en-US', {
          weekday: 'long',
          year: 'numeric',
          month: 'long',
          day: 'numeric'
        }));
      setEmailBody(formattedBody);
    }
  }, [showEmailModal, selectedIntern]);


  // Get API base URL from environment variables
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://localhost:3000";
const API_TIMEOUT = process.env.REACT_APP_API_TIMEOUT || 30000;

// Configure axios instance with base URL and timeout
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: parseInt(API_TIMEOUT),
});

  // Update date in email body when date changes
  useEffect(() => {
    if (selectedDate && emailBody.includes("[SELECTED_DATE]")) {
      const formattedDate = new Date(selectedDate).toLocaleDateString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
      });
      setEmailBody(prev => prev.replace("[SELECTED_DATE]", formattedDate));
    }
  }, [selectedDate]);

  const fetchShortlistedInterns = async () => {
  setLoading(true);
  try {
    const response = await api.get("/shortlisted-interns");
    
    setInterns(response.data.shortlisted_interns || []);
  } catch (error) {
    console.error("Error fetching shortlisted interns:", error);
    setError("Failed to load shortlisted interns. Please try again later.");
  } finally {
    setLoading(false);
  }
};

const handleClearAll = async () => {
  if (!window.confirm("Are you sure you want to delete all shortlisted interns?")) {
    return;
  }

  try {
    const response = await api.delete("/clear-shortlisted");

    // Refresh the list after clearing
    setInterns([]);
    alert("All shortlisted interns have been removed.");
  } catch (error) {
    console.error("Error clearing shortlisted interns:", error);
    alert("Failed to clear shortlisted interns. Please try again later.");
  }
};

  const openEmailModal = (intern) => {
    setSelectedIntern(intern);
    setShowEmailModal(true);
    setDefaultFileAttached(true);
    setSelectedFile(null); // Reset any custom file
  };

  const closeEmailModal = () => {
    setShowEmailModal(false);
    setSelectedIntern(null);
    setEmailSubject("Internship Confirmation and Reporting Instructions – SLT Digital Platforms Development Section");
    setEmailBody("");
    setSelectedFile(null);
    setDefaultFileAttached(true);
  };

  const handleDateChange = (e) => {
    const newDate = e.target.value;
    setSelectedDate(newDate);
    
    // Update date in email body
    if (newDate) {
      const formattedDate = new Date(newDate).toLocaleDateString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
      });
      
      setEmailBody(prev => {
        // If the email still has the placeholder
        if (prev.includes("[SELECTED_DATE]")) {
          return prev.replace("[SELECTED_DATE]", formattedDate);
        }
        
        // Try to find and replace an existing date
        const dateRegex = /\*\*Date:\*\* ([^*\n]+)/;
        return prev.replace(dateRegex, `**Date:** ${formattedDate}`);
      });
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setDefaultFileAttached(false);
    }
  };

  const toggleDefaultFile = () => {
    if (defaultFileAttached) {
      setDefaultFileAttached(false);
    } else {
      setDefaultFileAttached(true);
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = null;
      }
    }
  };

  const handleHireIntern = async () => {
  if (!selectedIntern || !selectedDate) return;
  
  if (!window.confirm(`Are you sure you want to hire ${selectedIntern.name}? An email will be sent to notify them and they will be removed from the shortlist.`)) {
    return;
  }

  try {
    setLoading(true);

    // Create form data to handle file upload
    const formData = new FormData();
    formData.append('deadline_date', selectedDate);
    formData.append('email_subject', emailSubject);
    formData.append('email_body', emailBody);
    
    // Append file if selected or default
    if (selectedFile) {
      formData.append('attachment', selectedFile);
    } else if (defaultFileAttached) {
      formData.append('use_default_attachment', 'true');
    }
    
    const response = await api.post(`/hire-intern/${selectedIntern.cv_id}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    
    const data = response.data;
    
    // Show success message
    alert(`${data.message}`);
    
    // Remove the intern from the local state since they've been hired and removed from the database
    setInterns(interns.filter(i => i.cv_id !== selectedIntern.cv_id));
    
    // Close the modal
    closeEmailModal();
    
  } catch (error) {
    console.error("Error hiring intern:", error);
    alert(`Failed to hire intern: ${error.response?.data?.detail || error.message}`);
  } finally {
    setLoading(false);
  }
};

const handleDeleteIntern = async (cv_id) => {
  if (!window.confirm("Are you sure you want to remove this intern from the shortlist?")) {
    return;
  }

  try {
    const response = await api.delete(`/remove-shortlisted/${cv_id}`);

    // Update the interns list after deletion
    setInterns(interns.filter(intern => intern.cv_id !== cv_id));
    alert("Intern removed from shortlist successfully.");
  } catch (error) {
    console.error("Error removing intern from shortlist:", error);
    alert("Failed to remove intern from shortlist. Please try again later.");
  }
};

  // 🔽 Sort interns by starting_date (latest first)
    const sortedInterns = [...interns].sort((a, b) => {
      const dateA = new Date(a.starting_date);
      const dateB = new Date(b.starting_date);
      return dateB - dateA; // Descending order
    });


  return (
    <div className="page-container2">
      <div className="nav-container">
        <Navbar />
      </div>
      <div className="content-wrapper">
        <div className="shortlisted-header">
          <div className="header-with-button">
            <div>
              <h1>Shortlisted Interns</h1>
              <p>Candidates you've selected for further consideration</p>
            </div>
            {interns.length > 0 && (
              <button className="clear-all-button" onClick={handleClearAll}>
                Clear All
              </button>
            )}
          </div>
        </div>
        
        {loading ? (
          <div className="loading-container">
            <div className="loader"></div>
            <p>Loading shortlisted interns...</p>
          </div>
        ) : error ? (
          <div className="error-container">
            <p className="error-message">{error}</p>
          </div>
        ) : interns.length === 0 ? (
          <div className="empty-state">
            <h2>No shortlisted interns yet</h2>
            <p>When you shortlist candidates, they will appear here</p>
          </div>
        ) : (
          <div className="container">
            <div className="interns-grid">
              {sortedInterns.map((intern, index) => {
                const allSkills = Object.values(intern.skills || {}).flat();
                return (
                  <div key={index} className="intern-card">
                    <div className="intern-role intern-role-shortlisted">
                      <p>
                        <a href={intern.cv_link} target="_blank" rel="noopener noreferrer">
                          <strong>Shortlisted Candidate</strong>
                        </a>
                      </p>
                    </div>
                    <div className="intern-details">
                      <h3>{intern.name}</h3>
                      <p><strong>Education:</strong> {intern.degree}</p>
                      <p><strong>Contact:</strong> {intern.contact_no}</p>
                      <p><strong>Email:</strong> {intern.email}</p>
                      <p><strong>Starting Date:</strong> {intern.starting_date}</p>
                      <p><strong>University:</strong> {intern.university}</p>
                      <p><strong>Possible Roles:</strong> {Array.isArray(intern.expected_role) ? intern.expected_role.join(", ") : JSON.stringify(intern.expected_role)}</p>
                      <p><strong>Year:</strong> {intern.current_year}</p>
                      <p><strong>Internship Period:</strong> {intern.internship_period}</p>
                      <p><strong>Working Mode:</strong> {intern.working_mode}</p>
                      <div className="skills-container">
                        {allSkills.map((skill, skillIndex) => (
                          <span key={skillIndex} className="skill-tag">
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="action-buttons">
                      <a href={intern.cv_link} target="_blank" rel="noopener noreferrer" className="view-cv">
                        View CV
                      </a>
                      <button 
                        className="hire-button" 
                        onClick={() => openEmailModal(intern)}
                      >
                        Hire
                      </button>
                      <button 
                        className="delete-button" 
                        onClick={() => handleDeleteIntern(intern.cv_id)}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
        
        {/* Email Customization Modal */}
        {showEmailModal && (
          <div className="modal-overlay">
            <div className="email-modal">
              <h2>Hire Intern: {selectedIntern?.name}</h2>
              
              <div className="modal-section">
                <h3>Reporting Date</h3>
                <p>Select the date when the intern should report to the office:</p>
                <input 
                  type="date" 
                  value={selectedDate} 
                  onChange={handleDateChange}
                  min={new Date().toISOString().split('T')[0]} // Can't select dates in the past
                  className="date-input"
                />
              </div>
              
              <div className="modal-section">
                <h3>Email Subject</h3>
                <input 
                  type="text"
                  value={emailSubject}
                  onChange={(e) => setEmailSubject(e.target.value)}
                  className="subject-input"
                  placeholder="Enter email subject"
                />
              </div>
              
              <div className="modal-section">
                <h3>Email Body</h3>
                <p className="formatting-note">
                  <strong>Note:</strong> You can use <code>**text**</code> for bold text. The email will be sent as HTML.
                </p>
                <textarea 
                  value={emailBody}
                  onChange={(e) => setEmailBody(e.target.value)}
                  className="body-input"
                  placeholder="Enter email body"
                  rows={12}
                />
              </div>
              
              <div className="modal-section">
                <h3>Attachment</h3>
                <div className="attachment-controls">
                  <div className="attachment-option">
                    <input 
                      type="checkbox" 
                      id="default-attachment" 
                      checked={defaultFileAttached} 
                      onChange={toggleDefaultFile}
                    />
                    <label htmlFor="default-attachment">
                      Use default Trainee Guidelines document
                    </label>
                  </div>
                  
                  <div className="attachment-upload">
                    <span>Or upload your own file:</span>
                    <input 
                      type="file" 
                      onChange={handleFileChange}
                      disabled={defaultFileAttached}
                      ref={fileInputRef}
                      className="file-input"
                    />
                    {selectedFile && (
                      <div className="selected-file">
                        Selected: {selectedFile.name}
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              <div className="modal-buttons">
                <button className="cancel-button" onClick={closeEmailModal}>
                  Cancel
                </button>
                <button 
                  className="confirm-button" 
                  onClick={handleHireIntern}
                  disabled={!selectedDate || !emailSubject || !emailBody}
                >
                  Send Email & Hire
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ShortlistedInterns;