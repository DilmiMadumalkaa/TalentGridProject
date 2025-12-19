// src/components/navbar/Navbar.jsx
import React from "react";
import { Link } from "react-router-dom";
import "./Navbar.css";
import logo from "../../assets/logo.png"; // Adjust the path if needed

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <div className="navbar__logo">
            <img src={logo} alt="Company Logo" className="navbar-logo" />
        </div>
        <h1 className="navbar-title">Intern Hiring System</h1>
      </div>
      <ul className="navbar-links">
        <li><Link to="/home">Home</Link></li>
        <li><Link to="/get-new-interns">Get New Interns</Link></li>
        <li><Link to="/hire-new-interns">Hire New Interns</Link></li>
        <li><Link to="/shortlisted-interns">Shortlisted Interns</Link></li>
        <li><Link to="/hired-interns">Hired Interns</Link></li>
      </ul>
    </nav>
  );
};

export default Navbar;