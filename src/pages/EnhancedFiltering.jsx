// import "./EnhancedFiltering.css";
// import React, { useState } from "react";
// import { useLocation } from "react-router-dom";
// import Navbar from "../components/navbar/Navbar";

// const cities = [
//   "Colombo", "Gampaha", "Kandy", "Galle", "Matara", "Jaffna", "Negombo", "Anuradhapura",
//   "Badulla", "Kurunegala", "Ratnapura", "Trincomalee", "Batticaloa", "Ampara", "Nuwara Eliya",
//   "Polonnaruwa", "Hambantota", "Vavuniya", "Kilinochchi", "Mullaitivu", "Mannar", "Kalutara",
//   "Puttalam", "Matale", "Monaragala", "Kegalle", "Gampola", "Wattala", "Moratuwa",
//   "Dehiwala", "Ja-Ela", "Maharagama", "Panadura", "Avissawella", "Horana", "Chilaw",
//   "Gampaha", "Hatton", "Balangoda", "Tangalle", "Deniyaya"
// ];

// const RankedCV = () => {
//   const location = useLocation();
//   const { filteredInterns } = location.state || { filteredInterns: [] }; 

//   const [inputText, setInputText] = useState("");
//   const [selectedCity, setSelectedCity] = useState("");
//   const [errorMessage, setErrorMessage] = useState("");
//   const [interns, setInterns] = useState([]);
//   const [showContainer, setShowContainer] = useState(false);
//   const [loading, setLoading] = useState(false);

//   const fetchRankedCVs = async () => {
//     if (!inputText.trim() && !selectedCity) {
//       setErrorMessage("Please enter either a job role or select a city.");
//       return;
//     }
//     setErrorMessage("");
//     setLoading(true);
  
//     try {
//       const response = await fetch("http://127.0.0.1:5000/rank_cvs", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ 
//           prompt: inputText, 
//           city: selectedCity,
//           filteredInterns: filteredInterns 
//         }),
//       });
  
//       if (!response.ok) throw new Error("Failed to fetch");
//       const data = await response.json();
//       setInterns(data.ranked_cvs || []);
//       setShowContainer(true);
//     } catch (error) {
//       console.error("Error fetching ranked CVs:", error);
//     } finally {
//       setLoading(false);
//     }
//   };  

//   const handleHire = async (intern) => {
//     try {
//         const response = await fetch(`http://127.0.0.1:5000/shortlist/${intern.cvId}`, {
//             method: "POST",
//             headers: { "Content-Type": "application/json" }
//         });

//         if (!response.ok) throw new Error("Failed to shortlist");

//         alert(`${intern.name} has been shortlisted.`);
//         setInterns(interns.filter(i => i.cvId !== intern.cvId));  // Remove from UI
//     } catch (error) {
//         console.error("Error shortlisting intern:", error);
//     }
//   };

//   const handleRemove = async (intern) => {
//     try {
//         const response = await fetch(`http://127.0.0.1:5000/remove/${intern.cvId}`, {
//             method: "DELETE"
//         });

//         if (!response.ok) throw new Error("Failed to remove");

//         alert(`${intern.name} has been removed.`);
//         setInterns(interns.filter(i => i.cvId !== intern.cvId));  // Remove from UI
//     } catch (error) {
//         console.error("Error removing intern:", error);
//     }
//   };


//   return (
//     <div className="page-container2">
//       <div className="nav-container">
//         <Navbar />
//       </div>
//       <div className="content-wrapper">
//         <div className={`input-container ${showContainer ? "input-container-moved" : ""}`}>
//           <input
//             type="text"
//             placeholder="Enter job role..."
//             className="prompt-text"
//             value={inputText}
//             onChange={(e) => setInputText(e.target.value)}
//           />
          
//           <select 
//             className="city-dropdown" 
//             value={selectedCity} 
//             onChange={(e) => setSelectedCity(e.target.value)}
//           >
//             <option value="">Select a city</option>
//             {cities.map((city, index) => (
//               <option key={index} value={city}>{city}</option>
//             ))}
//           </select>
          
//           {errorMessage && <p className="error-message">{errorMessage}</p>}
  
//           <button 
//             onClick={fetchRankedCVs} 
//             className="submit-prompt-text" 
//             disabled={loading}
//           >
//             {loading ? "Searching..." : "Submit"}
//           </button>
  
//           {loading && <div className="loader"></div>}
//         </div>
  
//         {showContainer && (
//           <div className="container">
//             <h2 className="h2-ranked-interns">Ranked Interns</h2>
//             <div className="interns-grid">
//               {interns.map((intern, index) => {
//                 const allSkills = Object.values(intern.skills).flat();
//                 return (
//                   <div key={index} className="intern-card">
//                     <div className="intern-role">
//                       <p>
//                         <a href={intern.cvLink} target="_blank" rel="noopener noreferrer">
//                           <strong>Rank: </strong> {intern.rank}
//                         </a>
//                       </p>
//                     </div>
//                     <div className="intern-details">
//                       <h3>{intern.name}</h3>
//                       <p><strong>Education:</strong> {intern.education}</p>
//                       <p><strong>Contact:</strong> {intern.contactNo}</p>
//                       <p><strong>Email:</strong> {intern.email}</p>
//                       <p><strong>Starting Date:</strong> {intern.startingDate}</p>
//                       <p><strong>University:</strong> {intern.institute}</p>
//                       <p><strong>Possible Roles:</strong> {Array.isArray(intern.possibleJobRoles) ? intern.possibleJobRoles.join(", ") : JSON.stringify(intern.possibleJobRoles)}</p>
//                       <p><strong>Year:</strong> {intern.year}</p>
//                       <p><strong>Internship Period:</strong> {intern.internshipPeriod}</p>
//                       <p><strong>Working Mode:</strong> {intern.workingMode}</p>
//                       <p><strong>City:</strong> {intern.city || "N/A"}</p>
//                       <div className="skills-container">
//                         {allSkills.map((skill, skillIndex) => (
//                           <span key={skillIndex} className="skill-tag">
//                             {skill}
//                           </span>
//                         ))}
//                       </div>
//                     </div>
//                     <div className="action-buttons">
//                       <button
//                         className="hire-button"
//                         onClick={() => handleHire(intern)}
//                       >
//                         ShortList
//                       </button>
//                       <button
//                         className="remove-button"
//                         onClick={() => handleRemove(intern)}
//                       >
//                         Remove
//                       </button>
//                     </div>
//                   </div>
//                 );
//               })}
//             </div>
//           </div>
//         )}
//       </div>
//     </div>
//   );
// };

// export default RankedCV

import "./EnhancedFiltering.css";
import React, { useState } from "react";
import { useLocation } from "react-router-dom";
import Navbar from "../components/navbar/Navbar";
import axios from "axios";

const cities = [
  "Colombo", "Gampaha", "Kandy", "Galle", "Matara", "Jaffna", "Negombo", "Anuradhapura",
  "Badulla", "Kurunegala", "Ratnapura", "Trincomalee", "Batticaloa", "Ampara", "Nuwara Eliya",
  "Polonnaruwa", "Hambantota", "Vavuniya", "Kilinochchi", "Mullaitivu", "Mannar", "Kalutara",
  "Puttalam", "Matale", "Monaragala", "Kegalle", "Gampola", "Wattala", "Moratuwa",
  "Dehiwala", "Ja-Ela", "Maharagama", "Panadura", "Avissawella", "Horana", "Chilaw",
  "Gampaha", "Hatton", "Balangoda", "Tangalle", "Deniyaya"
];

const RankedCV = () => {
  const location = useLocation();
  const { filteredInterns } = location.state || { filteredInterns: [] }; 

  const [inputText, setInputText] = useState("");
  const [selectedCity, setSelectedCity] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [interns, setInterns] = useState([]);
  const [showContainer, setShowContainer] = useState(false);
  const [loading, setLoading] = useState(false);



     // Get API base URL from environment variables
  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;
  const API_TIMEOUT = process.env.REACT_APP_API_TIMEOUT || 30000;
  
  // Configure axios instance with base URL and timeout
  const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: parseInt(API_TIMEOUT),
  });

  
  const fetchRankedCVs = async () => {
    if (!inputText.trim() && !selectedCity) {
      setErrorMessage("Please enter either  role or select a city.");
      return;
    }
    setErrorMessage("");
    setLoading(true);
  
    try {
      const response = await fetch("http://127.0.0.1:5000/rank_cvs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          prompt: inputText, 
          city: selectedCity,
          filteredInterns: filteredInterns 
        }),
      });
  
      if (!response.ok) throw new Error("Failed to fetch");
      const data = await response.json();
      
      if (data.ranked_cvs && data.ranked_cvs.length > 0) {
        console.log("Ranked CVs received:", data.ranked_cvs);
        setInterns(data.ranked_cvs);
        setShowContainer(true);
      } else {
        setErrorMessage(data.message || "No matching interns found");
        setInterns([]);
      }
    } catch (error) {
      console.error("Error fetching ranked CVs:", error);
      setErrorMessage("Error fetching ranked CVs. Please try again.");
    } finally {
      setLoading(false);
    }
  };  

  const handleHire = async (intern) => {
    try {
        const response = await fetch(`http://127.0.0.1:5000/shortlist/${intern.cvId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });

        if (!response.ok) throw new Error("Failed to shortlist");

        alert(`${intern.name} has been shortlisted.`);
        setInterns(interns.filter(i => i.cvId !== intern.cvId));  // Remove from UI
    } catch (error) {
        console.error("Error shortlisting intern:", error);
    }
  };

  const handleRemove = async (intern) => {
    try {
        const response = await fetch(`http://127.0.0.1:5000/remove/${intern.cvId}`, {
            method: "DELETE"
        });

        if (!response.ok) throw new Error("Failed to remove");

        alert(`${intern.name} has been removed.`);
        setInterns(interns.filter(i => i.cvId !== intern.cvId));  // Remove from UI
    } catch (error) {
        console.error("Error removing intern:", error);
    }
  };



  return (
    <div className="page-container2">
      <div className="nav-container">
        <Navbar />
      </div>
      <div className="content-wrapper">
        <div className={`input-container ${showContainer ? "input-container-moved" : ""}`}>
          <input
            type="text"
            placeholder="Enter job role..."
            className="prompt-text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
          />
          
          <select 
            className="city-dropdown" 
            value={selectedCity} 
            onChange={(e) => setSelectedCity(e.target.value)}
          >
            <option value="">Select a city</option>
            {cities.map((city, index) => (
              <option key={index} value={city}>{city}</option>
            ))}
          </select>
          
          {errorMessage && <p className="error-message">{errorMessage}</p>}
  
          <button 
            onClick={fetchRankedCVs} 
            className="submit-prompt-text" 
            disabled={loading}
          >
            {loading ? "Searching..." : "Submit"}
          </button>
  
          {loading && <div className="loader"></div>}
        </div>
  
        {showContainer && (
          <div className="container">
            <h2 className="h2-ranked-interns">Ranked Interns</h2>
            <div className="interns-grid">
              {interns.map((intern, index) => {
                const allSkills = Object.values(intern.skills || {}).flat();
                return (
                  <div key={index} className="intern-card">
                    <div className="intern-role">
                      <p>
                        <a href={intern.cvLink} target="_blank" rel="noopener noreferrer">
                          <strong>Rank: </strong> {intern.rank}
                        </a>
                      </p>
                    </div>
                    <div className="intern-details">
                      <h3>{intern.name || "Unknown Name"}</h3>
                      <p><strong>Education:</strong> {intern.education || "N/A"}</p>
                      <p><strong>Contact:</strong> {intern.contactNo || "N/A"}</p>
                      {intern.email && <p><strong>Email:</strong> {intern.email}</p>}
                      <p><strong>Starting Date:</strong> {intern.startingDate || "N/A"}</p>
                      <p><strong>University:</strong> {intern.institute || "N/A"}</p>
                      <p><strong>Possible Roles:</strong> {Array.isArray(intern.possibleJobRoles) ? intern.possibleJobRoles.join(", ") : (typeof intern.possibleJobRoles === 'string' ? intern.possibleJobRoles : "N/A")}</p>
                      <p><strong>Year:</strong> {intern.year || "N/A"}</p>
                      <p><strong>Internship Period:</strong> {intern.internshipPeriod || "N/A"}</p>
                      <p><strong>Working Mode:</strong> {Array.isArray(intern.workingMode) ? intern.workingMode.join(", ") : (typeof intern.workingMode === 'string' ? intern.workingMode : "N/A")}</p>
                      <p><strong>City:</strong> {intern.city || "N/A"}</p>
                      {allSkills && allSkills.length > 0 && (
                        <div className="skills-container">
                          {allSkills.map((skill, skillIndex) => (
                            <span key={skillIndex} className="skill-tag">
                              {skill}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                    <div className="action-buttons">
                      <button
                        className="hire-button"
                        onClick={() => handleHire(intern)}
                      >
                        ShortList
                      </button>
                      <button
                        className="remove-button"
                        onClick={() => handleRemove(intern)}
                      >
                        Remove
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default RankedCV