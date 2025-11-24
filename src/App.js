import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import LoginPage from "./pages/LoginPage";
import LandingPage from "./pages/LandingPage";
import FetchNewInterns from "./pages/FetchNewInterns";
import HireNewInterns from "./pages/HireNewInterns";
import EnhancedFiltering from "./pages/EnhancedFiltering";
import ShortlistedInterns from "./pages/ShortlistedInterns";
import ProtectedRoute from "./components/ProtectedRoute";
import Signup from "./pages/Signup";
import HiredInterns from "./pages/HiredInterns";

function App() {
  return (
    <Router>
      <Routes>

        {/* Public routes */}
        <Route path="/login" element={<LoginPage />} />
        {/* <Route path="/signup" element={<Signup />} /> */}

        {/* Protected routes */}
        <Route 
          path="/" 
          element={
            <ProtectedRoute>
              <LandingPage />
            </ProtectedRoute>
          }
        />

        <Route 
          path="/home" 
          element={
            <ProtectedRoute>
              <LandingPage />
            </ProtectedRoute>
          }
        />

        <Route 
          path="/get-new-interns" 
          element={
            <ProtectedRoute>
              <FetchNewInterns />
            </ProtectedRoute>
          }
        />

        <Route 
          path="/hire-new-interns" 
          element={
            <ProtectedRoute>
              <HireNewInterns />
            </ProtectedRoute>
          }
        />

        <Route 
          path="/enhanced-filtering" 
          element={
            <ProtectedRoute>
              <EnhancedFiltering />
            </ProtectedRoute>
          }
        />

        <Route 
          path="/shortlisted-interns" 
          element={
            <ProtectedRoute>
              <ShortlistedInterns />
            </ProtectedRoute>
          }
        />

        <Route 
          path="/hired-interns" 
          element={
            <ProtectedRoute>
              <HiredInterns />
            </ProtectedRoute>
          }
        />

        {/* Unknown routes → redirect to login */}
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    </Router>
  );
}

export default App;
