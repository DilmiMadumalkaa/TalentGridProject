// src/components/footer/Footer.jsx
import React from "react";
import "./Footer.css";
import logo from "../../assets/logo.png"; // Adjust the path if needed

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="footer-logo-section">
            <div className="footer__logo">
                <img src={logo} alt="Company Logo" className="footer-logo" />
            </div>
          <h3>Intern Hiring System</h3>
        </div>
        <div className="footer-links">
          <div className="footer-column">
            <h4>Quick Links</h4>
            <ul>
              <li><a href="/landing">Home</a></li>
              <li><a href="/get-new-interns">Get Interns</a></li>
              <li><a href="/hire-new-interns">Hire Interns</a></li>
            </ul>
          </div>
          <div className="footer-column">
            <h4>Support</h4>
            <ul>
              <li><a href="#">Help Center</a></li>
              <li><a href="#">Contact Us</a></li>
              <li><a href="#">FAQ</a></li>
            </ul>
          </div>
        </div>
      </div>
      <div className="footer-bottom">
        <p>&copy; {new Date().getFullYear()} Intern Hiring System. All rights reserved.</p>
      </div>
    </footer>
  );
};

export default Footer;