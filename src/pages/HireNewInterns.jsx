import "./HireNewInterns.css";
import React, { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/navbar/Navbar";
import ClearOldInternsButton from "../components/ClearOldInterns/ClearOldInternsButton";

const HireNewInterns = () => {
  const [formData, setFormData] = useState({
    institute: "",
    degree: "",
    academicYear: "All",
    internshipPeriod: "",
    workingMode: "",
    role: "",
    startingDate: "",
    skills: [],
  });

  const [interns, setInterns] = useState([]);
  const [filteredInterns, setFilteredInterns] = useState([]);

  const navigate = useNavigate();

  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://127.0.0.1:5000";

  const API_TIMEOUT = process.env.REACT_APP_API_TIMEOUT || 30000;

  const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: parseInt(API_TIMEOUT),
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSkillsChange = (e) => {
    const selectedOptions = Array.from(e.target.selectedOptions).map(
      (option) => option.value
    );
    setFormData({ ...formData, skills: selectedOptions });
  };

  // Fetch all interns
  useEffect(() => {
    fetchInterns();
  }, []);

  useEffect(() => {
    setFilteredInterns(interns);
  }, [interns]);

  const fetchInterns = async () => {
    try {
      const response = await api.get("/interns");
      setInterns(response.data);
      console.log("Fetched interns:", response.data);
    } catch (error) {
      console.error("Error fetching interns:", error);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const results = interns.filter((intern) => {
      const matchesInstitute =
        formData.institute === "" ||
        intern.university.toLowerCase().includes(formData.institute.toLowerCase());
      const matchesDegree =
        formData.degree === "" ||
        intern.degree.toLowerCase().includes(formData.degree.toLowerCase());
      const matchesYear =
        formData.academicYear === "All" ||
        formData.academicYear === "" ||
        intern.current_year === formData.academicYear;
      const matchesPeriod =
        formData.internshipPeriod === "" ||
        intern.internship_period === formData.internshipPeriod;
      const matchesMode =
        formData.workingMode === "" || intern.working_mode.includes(formData.workingMode);
      const matchesRole =
        formData.role === "" || intern.expected_role.includes(formData.role);
      const matchesDate =
        formData.startingDate === "" ||
        new Date(intern.starting_date) <= new Date(formData.startingDate);
      const matchesSkills =
        formData.skills.length === 0 ||
        formData.skills.some((skill) =>
          Object.values(intern.skills).some((category) => category.includes(skill))
        );

      return (
        matchesInstitute &&
        matchesDegree &&
        matchesYear &&
        matchesPeriod &&
        matchesMode &&
        matchesRole &&
        matchesDate &&
        matchesSkills
      );
    });

    setFilteredInterns(results);
  };

  const handleHire = async (intern) => {
    try {
      await api.post(`/shortlist/${intern.cvid}`);
      alert(`${intern.name} has been shortlisted.`);
      const updatedInterns = interns.filter((i) => i.cv_id !== intern.cv_id);
      setInterns(updatedInterns);
      setFilteredInterns(filteredInterns.filter((i) => i.cv_id !== intern.cv_id));
    } catch (error) {
      console.error("Error shortlisting intern:", error);
    }
  };

  const handleRemove = async (intern) => {
    try {
      await api.delete(`/remove/${intern.cv_id}`);
      alert(`${intern.name} has been removed.`);
      const updatedInterns = interns.filter((i) => i.cv_id !== intern.cv_id);
      setInterns(updatedInterns);
      setFilteredInterns(filteredInterns.filter((i) => i.cv_id !== intern.cv_id));
    } catch (error) {
      console.error("Error removing intern:", error);
    }
  };

  const handleEnhancedFiltering = () => {
    navigate("/enhanced-filtering", { state: { filteredInterns } });

    const sortedByStartingDate = [...interns].sort((a, b) => {
    return new Date(b.starting_date) - new Date(a.starting_date);
  });

  setFilteredInterns(sortedByStartingDate);

  };

  // Check if at least one filter is applied
  const isFilterApplied = () => {
    return (
      formData.institute !== "" ||
      formData.degree !== "" ||
      formData.academicYear !== "All" ||
      formData.internshipPeriod !== "" ||
      formData.workingMode !== "" ||
      formData.role !== "" ||
      formData.startingDate !== "" ||
      formData.skills.length > 0
    );
  };

  // Sort filtered interns by latest starting date (descending)
  const sortedFilteredInterns = [...filteredInterns].sort(
    (a, b) => new Date(b.starting_date) - new Date(a.starting_date)
  );


  return (
    <div className="hire-container">
      <Navbar />
      <div className="btn-adjustment">
        <h1 className="hire-title">Filter Interns</h1>
        <ClearOldInternsButton onCleared={fetchInterns}/>
      </div>


      <form className="hire-form" onSubmit={handleSubmit}>
        {/* Educational Institute */}
        <label className="hire-label">
          Educational Institute:
          <input
            type="text"
            name="institute"
            value={formData.institute}
            onChange={handleInputChange}
            className="hire-input"
          />
        </label>

        {/* Degree/Course */}
        <label className="hire-label">
          Degree/Course:
          <input
            type="text"
            name="degree"
            value={formData.degree}
            onChange={handleInputChange}
            className="hire-input"
          />
        </label>

        {/* Academic Year */}
        <label className="hire-label">
          Current Academic Year:
          <select
            name="academicYear"
            value={formData.academicYear}
            onChange={handleInputChange}
            className="hire-select"
          >
            <option value="All">All</option>
            <option value="1st year">1st Year</option>
            <option value="2nd year">2nd Year</option>
            <option value="3rd year">3rd Year</option>
            <option value="4th year">4th Year</option>
          </select>
        </label>

        {/* Internship Period */}
        <label className="hire-label">
          Internship Period (in months):
          <select
            name="internshipPeriod"
            value={formData.internshipPeriod}
            onChange={handleInputChange}
            className="hire-select"
          >
            <option value="">Select</option>
            {Array.from({ length: 9 }, (_, i) => i + 3).map((month) => (
              <option key={month} value={month}>
                {month}
              </option>
            ))}
          </select>
        </label>

        {/* Working Mode */}
        <label className="hire-label">
          Working Mode:
          <select
            name="workingMode"
            value={formData.workingMode}
            onChange={handleInputChange}
            className="hire-select"
          >
            <option value="">Select</option>
            <option value="Work from home">Work from Home</option>
            <option value="Work from office">Work from Office</option>
            <option value="Hybrid (office & home)">Hybrid</option>
          </select>
        </label>

        {/* Role */} 
        <label className="hire-label"> 
          Role: 
          <select name="role" value={formData.role} onChange={handleInputChange} className="hire-select" > 
            <option value="">Select</option> 
            <option value="Account Manager">Account Manager</option> 
            <option value="Agricultural Engineer">Agricultural Engineer</option> 
            <option value="Agronomist">Agronomist</option> 
            <option value="AI/ML Engineer">AI/ML Engineer</option> 
            <option value="AI/ML Engineer (Computer Vision)">AI/ML Engineer (Computer Vision)</option> 
            <option value="AI/ML Ops Engineer">AI/ML Ops Engineer</option> 
            <option value="AI Research Scientist">AI Research Scientist</option> 
            <option value="AI Robotics Engineer">AI Robotics Engineer</option> 
            <option value="AI Technical Lead">AI Technical Lead</option> 
            <option value="Biomedical Engineer">Biomedical Engineer</option> 
            <option value="Business Analyst">Business Analyst</option> 
            <option value="Business Development Manager">Business Development Manager</option> 
            <option value="Cloud Engineer">Cloud Engineer</option> 
            <option value="Computer Vision Engineer">Computer Vision Engineer</option> 
            <option value="Computer Vision Solution Architect">Computer Vision Solution Architect</option> 
            <option value="Contact Center Manager">Contact Center Manager</option> 
            <option value="Customer Experience Manager">Customer Experience Manager</option> 
            <option value="Data Entry Operator">Data Entry Operator</option> 
            <option value="Data Analyst">Data Analyst</option> 
            <option value="Data Engineer">Data Engineer</option> 
            <option value="Data Scientist">Data Scientist</option> 
            <option value="DevOps Engineer">DevOps Engineer</option> 
            <option value="Digital Content Creator">Digital Content Creator</option> 
            <option value="Digital Marketing Manager">Digital Marketing Manager</option> 
            <option value="Digital Media Manager">Digital Media Manager</option> 
            <option value="Edge Computing Engineer">Edge Computing Engineer</option> 
            <option value="Electrical Engineer">Electrical Engineer</option> 
            <option value="Embedded Computer Vision Engineer">Embedded Computer Vision Engineer</option> 
            <option value="Embedded System Engineer">Embedded System Engineer</option> 
            <option value="Field Operation Engineer">Field Operation Engineer</option> 
            <option value="Graphic Designer">Graphic Designer</option> 
            <option value="Hardware QA Engineer">Hardware QA Engineer</option> 
            <option value="Help Desk Manager">Help Desk Manager</option> 
            <option value="Innovation Manager">Innovation Manager</option> 
            <option value="IOT AI/ML Engineer">IOT AI/ML Engineer</option> 
            <option value="IOT Firmware Engineer">IOT Firmware Engineer</option> 
            <option value="IOT Hardware Engineer">IOT Hardware Engineer</option> 
            <option value="IOT Implementation Engineer">IOT Implementation Engineer</option> 
            <option value="IOT Mechanical Engineer">IOT Mechanical Engineer</option> 
            <option value="IOT R&D Engineer">IOT R&D Engineer</option> 
            <option value="IOT Security Specialist">IOT Security Specialist</option> 
            <option value="IOT Solution Architect">IOT Solution Architect</option> 
            <option value="IOT Support Engineer">IOT Support Engineer</option> 
            <option value="IOT System Integration Engineer">IOT System Integration Engineer</option> 
            <option value="IOT Technical Lead">IOT Technical Lead</option> 
            <option value="Marketing Analyst">Marketing Analyst</option> 
            <option value="Marketing Manager">Marketing Manager</option> 
            <option value="Network Automation Engineer">Network Automation Engineer</option> 
            <option value="Network Engineer">Network Engineer</option> 
            <option value="Product Manager">Product Manager</option> 
            <option value="Project Manager">Project Manager</option> 
            <option value="Power Electronics Engineer">Power Electronics Engineer</option> 
            <option value="Research Analyst">Research Analyst</option> 
            <option value="RF Engineer">RF Engineer</option> 
            <option value="Robotics Engineer">Robotics Engineer</option> 
            <option value="R&D Scientist">R&D Scientist</option> 
            <option value="R&D Engineer">R&D Engineer</option> 
            <option value="Robot Simulation Engineer">Robot Simulation Engineer</option> 
            <option value="Sales & Marketing Executive">Sales & Marketing Executive</option> 
            <option value="Sales Manager">Sales Manager</option> 
            <option value="Site Reliability Engineer">Site Reliability Engineer</option> 
            <option value="Software Architect">Software Architect</option> 
            <option value="Software Developer (Android Mobile App)">Software Developer (Android Mobile App)</option> 
            <option value="Software Developer (iOS Mobile App)">Software Developer (iOS Mobile App)</option> 
            <option value="Software Developer (API)">Software Developer (API)</option> 
            <option value="Software Developer (AR)">Software Developer (AR)</option> 
            <option value="Software Developer (Backend)">Software Developer (Backend)</option> 
            <option value="Software Developer (Blockchain)">Software Developer (Blockchain)</option> 
            <option value="Software Developer (Frontend)">Software Developer (Frontend)</option> 
            <option value="Software Developer (FullStack)">Software Developer (FullStack)</option> 
            <option value="Software Developer (Metaverse)">Software Developer (Metaverse)</option> 
            <option value="Software Engineer">Software Engineer</option> 
            <option value="Software QA Engineer">Software QA Engineer</option> 
            <option value="Software Technical Lead">Software Technical Lead</option> 
            <option value="Solution Architect">Solution Architect</option> 
            <option value="Solution Support Engineer">Solution Support Engineer</option> 
            <option value="System Integration Engineer">System Integration Engineer</option> 
            <option value="Technical Support Engineer">Technical Support Engineer</option> 
            <option value="Technical Writer">Technical Writer</option> 
            <option value="Technician">Technician</option> 
            <option value="UAV Engineer">UAV Engineer</option> 
            <option value="UI/UX Designer">UI/UX Designer</option> 
            <option value="UI/UX Developer">UI/UX Developer</option> 
            <option value="VoIP Engineer">VoIP Engineer</option> 
            <option value="Wireless R&D Engineer">Wireless R&D Engineer</option> 
            </select> 
        </label>

        {/* Starting Date */}
        <label className="hire-label">
          Starting Date:
          <input
            type="date"
            name="startingDate"
            value={formData.startingDate}
            onChange={handleInputChange}
            className="hire-input"
          />
        </label>

        {/* Skills */}
        <label className="hire-label">
          Skills:
          <select
            name="skills"
            multiple
            value={formData.skills}
            onChange={handleSkillsChange}
            className="hire-multi-select"
          >
            <option value=".Net">.Net</option>
            <option value="C#">C#</option>
            <option value="Python">Python</option>
            <option value="Java">Java</option>
            <option value="React">React</option>
            <option value="Flutter">Flutter</option>
            <option value="Docker">Docker</option>
            <option value="Kubernetes">Kubernetes</option>
            <option value="TensorFlow">TensorFlow</option>
          </select>
        </label>

        {/* Submit Button */}
        <button
          type="submit"
          className={`hire-submit ${!isFilterApplied() ? "disabled" : ""}`}
          disabled={!isFilterApplied()}
        >
          Apply Filters
        </button>
      </form>

      {/* Enhanced Filtering Button */}
      <button
        type="button"
        className="ai-filter-button"
        onClick={() => handleEnhancedFiltering(filteredInterns)}
      >
        Enhanced CV Filtering Using AI
      </button>

      <div className="results-container">
        <h2>Filtered Interns</h2>
        {filteredInterns.length > 0 ? (
          <div className="interns-grid">

            {sortedFilteredInterns.map((intern, index) => (

              <div key={index} className="intern-card">
                <a
                  href={intern.cv_link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="thumbnail"
                >
                  <div className="thumbnail-text">View CV</div>
                </a>
                <div className="intern-details">
                  <h3>{intern.name}</h3>
                  <p>{intern.degree} at {intern.university}</p>
                  <p><strong>Year:</strong> {intern.current_year}</p>
                  <p><strong>University:</strong> {intern.university}</p>
                  <p><strong>Contact:</strong> {intern.contact_no}</p>
                  <p><strong>Starting Date:</strong> {intern.starting_date}</p>
                  <p><strong>Email:</strong> {intern.email}</p>
                  <p><strong>Internship Period:</strong> {intern.internship_period} months</p>
                  <p><strong>Working Mode:</strong> {intern.working_mode.join(", ")}</p>
                  <p><strong>Role:</strong> {intern.expected_role.join(", ")}</p>

                  <div className="skills-container2">
                    {Object.values(intern.skills).flat().map((skill, i) => (
                      <span key={i} className="skill-badge2">{skill}</span>
                    ))}
                  </div>

                  <div className="action-buttons">
                    <button className="hire-button" onClick={() => handleHire(intern)}>
                      ShortList
                    </button>
                    <button className="remove-button" onClick={() => handleRemove(intern)}>
                      Remove
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p>No interns match the selected criteria.</p>
        )}
      </div>
    </div>
  );
};

export default HireNewInterns;
