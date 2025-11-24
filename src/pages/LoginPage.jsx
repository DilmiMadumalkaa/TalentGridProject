import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import "./Auth.css"; // We'll create a shared CSS file for both login & signup
import { FaUser, FaLock, FaSignInAlt, FaEye, FaEyeSlash } from "react-icons/fa";

const Login = () => {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false); // 👁 password toggle
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  // Get API base URL from environment variables
  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://127.0.0.1:3000";
  const API_TIMEOUT = process.env.REACT_APP_API_TIMEOUT || 30000;

  // Configure axios instance with base URL and timeout
  const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: parseInt(API_TIMEOUT),
  });

  const handleSubmit = (e) => {
  e.preventDefault();
  setError("");
  setLoading(true);

  const email = formData.email.trim();
  const password = formData.password.trim();

  // ✅ HARD-CODED CHECK
  if (email === "digitalplatforms@slt.lk" && password === "digitalplatform@456") {
    localStorage.setItem("token", "hardcoded_token_value");
    localStorage.setItem("user", JSON.stringify({ email }));

    navigate("/"); // Go to home page
  } else {
    setError("Invalid username or password."); // Show error only if wrong
  }

  setLoading(false);
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

          {/* PASSWORD FIELD WITH SHOW/HIDE ICON */}
          <div className="form-group password-group">
            <label htmlFor="password">
              <FaLock /> Password
            </label>

            <div className="password-wrapper">
              <input
                type={showPassword ? "text" : "password"} // 👁 toggle
                id="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="Enter your password"
                required
              />

              {/* 👁 Show/Hide Password Icon */}
              <span
                className="password-toggle-icon"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <FaEyeSlash /> : <FaEye />}
              </span>
            </div>
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

        {/* Optional: Signup link hidden */}
        {/* 
        <div className="auth-alternate">
          <p>
            Don't have an account? <Link to="/signup">Sign Up</Link>
          </p>
        </div> 
        */}
      </div>
    </div>
  );
};

export default Login;
