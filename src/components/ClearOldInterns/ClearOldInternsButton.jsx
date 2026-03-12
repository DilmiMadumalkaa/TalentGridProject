import React, { useState } from "react";
import axios from "axios";
import "./ClearOldInterns.css";

export default function ClearOldInternsButton({ onCleared }) {
  const [loading, setLoading] = useState(false);
  const [resultMsg, setResultMsg] = useState("");

  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;

  const handleClear = async () => {
    const ok = window.confirm(
      "This will permanently delete interns whose starting date is older than 3 months. Continue?"
    );
    if (!ok) return;

    setLoading(true);
    setResultMsg("");

    try {
      const res = await axios.delete(`${API_BASE_URL}/interns/clear-old`, {
        timeout: 30000,
      });

      const deleted = res?.data?.deleted_count ?? 0;
      const cutoff = res?.data?.cutoff_date ?? "";
      setResultMsg(`Deleted ${deleted} interns (cutoff: ${cutoff}).`);

      // optional: refresh list in parent
      if (typeof onCleared === "function"){
        await onCleared();
        window.location.reload();
      } 
      
    } catch (err) {
      const msg =
        err?.response?.data?.detail ||
        err?.message ||
        "Failed to clear old interns.";
      setResultMsg(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <button
        onClick={handleClear}
        disabled={loading}
        className="clear-all-button"
      >
        {loading ? "Clearing..." : "Clear Old Interns"}
      </button>

      {resultMsg ? <p className="text-sm text-gray-700">{resultMsg}</p> : null}
    </div>
  );
}
