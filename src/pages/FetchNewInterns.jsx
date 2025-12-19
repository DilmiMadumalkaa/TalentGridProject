import React, { useState, useEffect } from "react";
import axios from "axios";
import "./FetchNewInterns.css";
import Navbar from "../components/navbar/Navbar";

const FetchNewInterns = () => {
  const [emails, setEmails] = useState([]);
  const [latestTime, setLatestTime] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [sortBy, setSortBy] = useState("name");
  const [sortOrder, setSortOrder] = useState("asc");
  const [activeFilter, setActiveFilter] = useState(null);

 // Get API base URL from environment variables
  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://localhost:3000";
  const API_TIMEOUT = process.env.REACT_APP_API_TIMEOUT || 30000;
  
  // Configure axios instance with base URL and timeout
  const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: parseInt(API_TIMEOUT),
  });

  const fetchEmails = async () => {
  setIsLoading(true);
  try {
    // Use the configured api instance instead of axios directly
    const response = await api.get("/emails");
    if (Array.isArray(response.data)) {
      setEmails(response.data);
    } else {
      console.error("Unexpected API response format:", response.data);
      setEmails([]);
    }
  } catch (error) {
    console.error("Error fetching emails:", error);
    setEmails([]);
  } finally {
    setIsLoading(false);
  }
};

  const fetchLatestTime = async () => {
    try {
      const response = await api.get("/latest_uploaded_time");
      if (response.data.latest_uploaded_time) {
        setLatestTime(response.data.latest_uploaded_time);
      } else {
        setLatestTime("No previous fetch time available");
      }
    } catch (error) {
      console.error("Error fetching latest uploaded time:", error);
    }
  };

  useEffect(() => {
    fetchLatestTime();
  }, []);

  const saveEmailDataALL = async (emails) => {
  try {
    let response2 = "false";
    for (let i = 0; i < emails.length; i++) {
      // Use the configured api instance instead of axios directly
      const response = await api.post("/save_email_data", emails[i]);
      console.log(response.data.message);
      response2 = response.data;
    }
    
    if (response2 === "true") {
      showNotification("Success", "Email data saved successfully", "success");
    } else {
      showNotification("Error", response2, "error");
    }
  } catch (error) {
    console.error("Error saving email data:", error);
    showNotification("Error", "Failed to save email data", "error");
  }
};

  const showNotification = (title, message, type) => {
    const notification = document.createElement("div");
    notification.className = `notification ${type}`;
    notification.innerHTML = `
      <div class="notification-title">${title}</div>
      <div class="notification-message">${message}</div>
    `;
    document.body.appendChild(notification);
    
    setTimeout(() => {
      notification.classList.add("show");
    }, 10);
    
    setTimeout(() => {
      notification.classList.remove("show");
      setTimeout(() => {
        document.body.removeChild(notification);
      }, 300);
    }, 3000);
  };

  const handleSort = (key) => {
    if (sortBy === key) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortBy(key);
      setSortOrder("asc");
    }
  };

  const handleFilter = (key, value) => {
    if (activeFilter && activeFilter.key === key && activeFilter.value === value) {
      setActiveFilter(null);
    } else {
      setActiveFilter({ key, value });
    }
  };

  const filteredEmails = emails
    .filter(email => {
      // Search filter
      const searchFields = [
        email.name, 
        email.university, 
        email.degree, 
        email.expected_role && (Array.isArray(email.expected_role) ? email.expected_role.join(" ") : email.expected_role),
        email.email
      ].filter(Boolean).join(" ").toLowerCase();
      
      const matchesSearch = !searchTerm || searchFields.includes(searchTerm.toLowerCase());
      
      // Active filter
      const matchesFilter = !activeFilter || 
        (email[activeFilter.key] && 
         (Array.isArray(email[activeFilter.key]) 
          ? email[activeFilter.key].includes(activeFilter.value)
          : email[activeFilter.key] === activeFilter.value));
      
      return matchesSearch && matchesFilter;
    })
    .sort((a, b) => {
      const aValue = a[sortBy] || "";
      const bValue = b[sortBy] || "";
      
      if (typeof aValue === "string" && typeof bValue === "string") {
        return sortOrder === "asc" 
          ? aValue.localeCompare(bValue) 
          : bValue.localeCompare(aValue);
      }
      
      return sortOrder === "asc" 
        ? (aValue > bValue ? 1 : -1)
        : (bValue > aValue ? 1 : -1);
    });

  // Get unique values for filters
  const getUniqueValues = (key) => {
    const values = emails
      .map(email => email[key])
      .filter(Boolean)
      .flat();
    
    return Array.isArray(values[0]) 
      ? [...new Set(values.flat())]
      : [...new Set(values)];
  };

  const uniqueRoles = getUniqueValues("expected_role");
  const uniqueWorkingModes = getUniqueValues("working_mode");

  return (
    <div className="App">
      <Navbar />
    <div className="page-container">
      
    <div className="interns-container">
      <header className="interns-header">
        <h1>Internship Management System</h1>
        <p className="subtitle">Manage and track potential interns</p>
      </header>

      <div className="interns-actions">
        <div className="action-group">
          <button 
            onClick={fetchEmails} 
            className="btn btn-primary"
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <span className="spinner"></span>
                <span>Fetching...</span>
              </>
            ) : (
              <>
                <i className="icon-refresh"></i>
                <span>Fetch Emails</span>
              </>
            )}
          </button>
          
          {emails.length > 0 && (
            <button 
              onClick={() => saveEmailDataALL(emails)} 
              className="btn btn-secondary"
            >
              <i className="icon-save"></i>
              <span>Save New Email Data</span>
            </button>
          )}
        </div>
        
        {latestTime && (
          <div className="latest-time">
            <i className="icon-clock"></i>
            <span>Last Fetched: </span>
            <strong>{new Date(latestTime).toLocaleString()}</strong>
          </div>
        )}
      </div>

      {emails.length > 0 && (
        <div className="interns-controls">
          <div className="search-container">
            <input
              type="text"
              placeholder="Search interns..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
            <i className="icon-search"></i>
          </div>
          
          <div className="filters">
            <div className="filter-group">
              <label>Filter by Role:</label>
              <div className="filter-options">
                {uniqueRoles.slice(0, 5).map(role => (
                  <button
                    key={role}
                    className={`filter-btn ${activeFilter && activeFilter.key === 'expected_role' && activeFilter.value === role ? 'active' : ''}`}
                    onClick={() => handleFilter('expected_role', role)}
                  >
                    {role}
                  </button>
                ))}
              </div>
            </div>
            
            <div className="filter-group">
              <label>Filter by Mode:</label>
              <div className="filter-options">
                {uniqueWorkingModes.slice(0, 3).map(mode => (
                  <button
                    key={mode}
                    className={`filter-btn ${activeFilter && activeFilter.key === 'working_mode' && activeFilter.value === mode ? 'active' : ''}`}
                    onClick={() => handleFilter('working_mode', mode)}
                  >
                    {mode}
                  </button>
                ))}
              </div>
            </div>
          </div>
          
          <div className="sort-controls">
            <label>Sort by:</label>
            <div className="sort-options">
              <button
                className={`sort-btn ${sortBy === 'name' ? 'active' : ''}`}
                onClick={() => handleSort('name')}
              >
                Name {sortBy === 'name' && (sortOrder === 'asc' ? '↑' : '↓')}
              </button>
              <button
                className={`sort-btn ${sortBy === 'university' ? 'active' : ''}`}
                onClick={() => handleSort('university')}
              >
                Institute {sortBy === 'university' && (sortOrder === 'asc' ? '↑' : '↓')}
              </button>
              <button
                className={`sort-btn ${sortBy === 'internship_period' ? 'active' : ''}`}
                onClick={() => handleSort('internship_period')}
              >
                Duration {sortBy === 'internship_period' && (sortOrder === 'asc' ? '↑' : '↓')}
              </button>
            </div>
          </div>
        </div>
      )}

      {isLoading ? (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading emails, please wait...</p>
        </div>
      ) : emails.length > 0 ? (
        <>
          <div className="interns-count">
            <span>Total Interns: </span>
            <strong>{filteredEmails.length}</strong>
            {activeFilter && <span className="filtered-text"> (filtered)</span>}
          </div>
          
          <div className="interns-grid">
            {filteredEmails.map((email, index) => (
              <div key={index} className="intern-card">
                <div className="card-header">
                  <h2>{email.name || "No Name"}</h2>
                  {email.expected_role && (
                    <div className="role-tags">
                      {(Array.isArray(email.expected_role) ? email.expected_role : [email.expected_role]).map((role, idx) => (
                        <span key={idx} className="role-tag">{role}</span>
                      ))}
                    </div>
                  )}
                </div>

                <div className="card-body">
                  <div className="info-group">
                    <div className="info-item">
                      <i className="icon-school"></i>
                      <div>
                        <label>Institute</label>
                        <p>{email.university || "N/A"}</p>
                      </div>
                    </div>
                    
                    <div className="info-item">
                      <i className="icon-degree"></i>
                      <div>
                        <label>Course</label>
                        <p>{email.degree || "N/A"}</p>
                      </div>
                    </div>
                    
                    <div className="info-item">
                      <i className="icon-calendar"></i>
                      <div>
                        <label>Current Year</label>
                        <p>{email.current_year || "N/A"}</p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="info-group">
                    <div className="info-item">
                      <i className="icon-clock"></i>
                      <div>
                        <label>Internship Period</label>
                        <p>{email.internship_period || "N/A"} months</p>
                      </div>
                    </div>
                    
                    <div className="info-item">
                      <i className="icon-mode"></i>
                      <div>
                        <label>Working Mode</label>
                        <p>{Array.isArray(email.working_mode) ? email.working_mode.join(", ") : email.working_mode || "N/A"}</p>
                      </div>
                    </div>
                    
                    <div className="info-item">
                      <i className="icon-date"></i>
                      <div>
                        <label>Starting Date</label>
                        <p>{email.starting_date || "N/A"}</p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="contact-info">
                    <div className="contact-item">
                      <i className="icon-email"></i>
                      <a href={`mailto:${email.email}`}>{email.email || "N/A"}</a>
                    </div>
                    
                    <div className="contact-item">
                      <i className="icon-phone"></i>
                      <a href={`tel:${email.contact_no}`}>{email.contact_no || "N/A"}</a>
                    </div>
                  </div>
                  
                  <div className="skills-section">
                    <h3>Skills</h3>
                    <div className="skills-container">
                      {email.skills && typeof email.skills === "object" ? (
                        Object.entries(email.skills).map(([key, value], idx) => (
                          <div key={idx} className="skill-group">
                            <h4>{key}</h4>
                            <div className="skill-tags">
                              {Array.isArray(value) ? value.map((skill, i) => (
                                <span key={i} className="skill-tag">{skill}</span>
                              )) : (
                                <span className="skill-tag">{value}</span>
                              )}
                            </div>
                          </div>
                        ))
                      ) : (
                        <p>{email.skills || "No skills listed"}</p>
                      )}
                    </div>
                  </div>
                  
                  {email.statement && (
                    <div className="statement-section">
                      <h3>Personal Statement</h3>
                      <p className="statement">{email.statement}</p>
                    </div>
                  )}
                </div>

                <div className="card-footer">
                  {email.cv_link ? (
                    <a 
                      href={email.cv_link} 
                      target="_blank" 
                      rel="noopener noreferrer" 
                      className="btn-cv"
                    >
                      <i className="icon-document"></i>
                      <span>View CV</span>
                    </a>
                  ) : (
                    <span className="no-cv">No CV available</span>
                  )}
                  
                  
                </div>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="no-data-container">
          <div className="no-data-icon">📥</div>
          <h2>No Intern Data Available</h2>
          <p>Click the "Fetch Emails" button to retrieve intern applications.</p>
        </div>
      )}
    </div>
    </div>
    </div>
  );
};

export default FetchNewInterns;