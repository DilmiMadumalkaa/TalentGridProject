// src/pages/LandingPage.jsx
import React from "react";
import { useNavigate } from "react-router-dom";
import "./LandingPage.css";
import Navbar from "../components/navbar/Navbar";
import Footer from "../components/footer/Footer";
import backdropImage from "../assets/backdrop.png"; // Adjust path if needed

const LandingPage = () => {
  const navigate = useNavigate();

  const handleNavigation = (path) => {
    navigate(path);
  };

  return (
    <div className="page-container">
      <Navbar />
      
      <div className="landing-container" style={{ backgroundImage: `url(${backdropImage})` }}>
        <div className="landing-content">
          <h1>Welcome to the Intern Hiring System</h1>
          <p>Streamline your recruitment process and find the perfect interns for your organization.</p>
          
          <div className="feature-cards">
            <div className="feature-card">
              <div className="feature-icon">🔍</div>
              <h3>Find Candidates</h3>
              <p>Search through our database of qualified intern candidates.</p>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon">📝</div>
              <h3>Manage Applications</h3>
              <p>Track and manage all intern applications in one place.</p>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon">🤝</div>
              <h3>Streamline Hiring</h3>
              <p>Simplify the entire hiring process from start to finish.</p>
            </div>
          </div>
          
          <div className="buttons-container">
            <button 
              className="primary-button"
              onClick={() => handleNavigation("/get-new-interns")}
            >
              Get New Interns
            </button>
            <button 
              className="secondary-button"
              onClick={() => handleNavigation("/hire-new-interns")}
            >
              Hire New Interns
            </button>
          </div>
        </div>
      </div>
      
      <Footer />
    </div>
  );
};

export default LandingPage;