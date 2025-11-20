import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import "./Auth.css"; // We'll create a shared CSS file for both login & signup
import { FaUser, FaLock, FaSignInAlt } from "react-icons/fa";

const Login = () => {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };
// Get API base URL from environment variables
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://127.0.0.1:5000";
const API_TIMEOUT = process.env.REACT_APP_API_TIMEOUT || 30000;

// Configure axios instance with base URL and timeout
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: parseInt(API_TIMEOUT),
});
  const handleSubmit = async (e) => {
  e.preventDefault();
  setError("");
  setLoading(true);

  try {
    // Create FormData object for OAuth2 format
    const formDataObj = new FormData();
    formDataObj.append("username", formData.email); // OAuth2 uses 'username', not 'email'
    formDataObj.append("password", formData.password);
    
    const response = await api.post(
      "/login", 
      formDataObj, 
      {
        headers: {
          "Content-Type": "application/x-www-form-urlencoded"
        }
      }
    );
    
    // Store token and user data based on the actual API response structure
    localStorage.setItem("token", response.data.access_token);
    localStorage.setItem("user", JSON.stringify(response.data.user));
    
    // Redirect to the upload page
    navigate("/landing");
  } catch (err) {
    console.error("Login failed", err);
    
    // Properly handle the error object
    if (err.response && err.response.data) {
      // If error.data is an object with 'detail', use that
      if (typeof err.response.data === 'object' && err.response.data.detail) {
        setError(err.response.data.detail);
      } 
      // If it's a string, use it directly
      else if (typeof err.response.data === 'string') {
        setError(err.response.data);
      }
      // Otherwise fallback to a generic message
      else {
        setError("Login failed. Please check your credentials and try again.");
      }
    } else {
      setError("Login failed. Please check your credentials and try again.");
    }
  } finally {
    setLoading(false);
  }
};

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <h2>Welcome Back</h2>
          <p>Sign in to your account to continue</p>
        </div>

        {error && (
          <div className="auth-error">
            <p>{error}</p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label htmlFor="email">
              <FaUser /> Email
            </label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              placeholder="Enter your email"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">
              <FaLock /> Password
            </label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              placeholder="Enter your password"
              required
            />
          </div>

          <div className="form-footer">
            <button 
              type="submit" 
              className="auth-button" 
              disabled={loading}
            >
              {loading ? (
                "Signing In..."
              ) : (
                <>
                  <FaSignInAlt /> Sign In
                </>
              )}
            </button>
          </div>
        </form>

        <div className="auth-alternate">
          <p>
            Don't have an account? <Link to="/signup">Sign Up</Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;