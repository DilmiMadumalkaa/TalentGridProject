import React from "react";
import { Navigate } from "react-router-dom";

const ProtectedRoute = ({ children }) => {
  // Check if user is logged in
  // const isAuthenticated = localStorage.getItem("token") !== null;

  // if (!isAuthenticated) {
  //   // Redirect to login if not authenticated
  //   return <Navigate to="/" replace />;
  // }

  return children;
};

export default ProtectedRoute;