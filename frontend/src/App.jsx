import { useState } from 'react'
import reactLogo from './assets/react.svg'
 

const initialForm = {
  CreditScore: "",
  Balance: "",
  EstimatedSalary: "",
  NumOfProducts: "",
  IsActiveMember: "",
  Age: "",
  Tenure: "",
  Gender: "",
  Geography: "",
};

function computeEngineeredFeatures(form) {
  const gender = parseInt(form.Gender);
  const geography = form.Geography;
  const balance = parseFloat(form.Balance);
  const salary = parseFloat(form.EstimatedSalary);
  const numProducts = parseFloat(form.NumOfProducts);
  const isActive = parseFloat(form.IsActiveMember);
  const age = parseFloat(form.Age);
  const tenure = parseFloat(form.Tenure);


/*this is one hot encoding. if germany is 0 and spain is 0, then it must be france. we do not need to create a separate column because  */

  const Geography_Germany = geography === "Germany" ? 1 : 0;
  const Geography_Spain = geography === "Spain" ? 1 : 0;

  /* we do /salary+1 to avoid division by zero */
  const BalanceSalaryRatio = balance / (salary + 1);
  const BalanceZero = balance === 0 ? 1 : 0;
  const ProductUsage = numProducts * isActive;
  const Male_Germany = gender * Geography_Germany;
  const Male_Spain = gender * Geography_Spain;

  const ageGroups = {
      AgeGroup_26_35: age >= 26 && age <= 35 ? 1 : 0,
      AgeGroup_36_45: age >= 36 && age <= 45 ? 1 : 0,
      AgeGroup_46_55: age >= 46 && age <= 55 ? 1 : 0,
      AgeGroup_56_65: age >= 56 && age <= 65 ? 1 : 0,
      AgeGroup_66_75: age >= 66 && age <= 75 ? 1 : 0,
      AgeGroup_76_85: age >= 76 && age <= 85 ? 1 : 0,
      AgeGroup_86_95: age >= 86 && age <= 95 ? 1 : 0,
    };

  const tenureGroups = {
    TenureGroup_3_4: tenure >= 3 && tenure <= 4 ? 1 : 0,
    TenureGroup_5_6: tenure >= 5 && tenure <= 6 ? 1 : 0,
    TenureGroup_7_8: tenure >= 7 && tenure <= 8 ? 1 : 0,
    TenureGroup_9_10: tenure >= 9 && tenure <= 10 ? 1 : 0,
  };

  return {
    CreditScore: parseFloat(form.CreditScore),
    Balance: balance,
    EstimatedSalary: salary,
    NumOfProducts: numProducts,
    IsActiveMember: isActive,
    BalanceSalaryRatio,
    BalanceZero,
    ProductUsage,
    Male_Germany,
    Male_Spain,
    ...ageGroups,
    ...tenureGroups,
  };

}

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // runs every time the user types in a field or changes a dropdown
  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  // runs when the user clicks Predict Churn
  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = computeEngineeredFeatures(form);
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error("Prediction failed");
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Could not connect to the backend. Make sure it is running.");
    } finally {
      setLoading(false);
    }

    // checks if every field has been filled in
    const isFormComplete = Object.values(form).every((v) => v !== "");
  };

 const isFormComplete = Object.values(form).every((v) => v !== "");

  return (
    <div className="page">
      <div className="container">

        {/* Header */}
        <div className="header">
          <div className="header-tag">AI-Powered</div>
          <h1 className="title">Churn <span className="accent">Predictor</span></h1>
          <p className="subtitle">Enter customer details to predict the likelihood of churn</p>
        </div>

        {/* Form */}
        <div className="card">
          <h2 className="section-title">Customer Profile</h2>
          <div className="form-grid">

            <div className="field">
              <label>Credit Score</label>
              <input
                type="number"
                name="CreditScore"
                placeholder="e.g. 650"
                value={form.CreditScore}
                onChange={handleChange}
              />
            </div>

            <div className="field">
              <label>Age</label>
              <input
                type="number"
                name="Age"
                placeholder="e.g. 35"
                value={form.Age}
                onChange={handleChange}
              />
            </div>

            <div className="field">
              <label>Tenure (years)</label>
              <input
                type="number"
                name="Tenure"
                placeholder="e.g. 5"
                value={form.Tenure}
                onChange={handleChange}
              />
            </div>

            <div className="field">
              <label>Balance</label>
              <input
                type="number"
                name="Balance"
                placeholder="e.g. 50000"
                value={form.Balance}
                onChange={handleChange}
              />
            </div>

            <div className="field">
              <label>Estimated Salary</label>
              <input
                type="number"
                name="EstimatedSalary"
                placeholder="e.g. 75000"
                value={form.EstimatedSalary}
                onChange={handleChange}
              />
            </div>

            <div className="field">
              <label>Number of Products</label>
              <input
                type="number"
                name="NumOfProducts"
                placeholder="1 - 4"
                min="1"
                max="4"
                value={form.NumOfProducts}
                onChange={handleChange}
              />
            </div>

            <div className="field">
              <label>Gender</label>
              <select name="Gender" value={form.Gender} onChange={handleChange}>
                <option value="">Select gender</option>
                <option value="1">Male</option>
                <option value="0">Female</option>
              </select>
            </div>

            <div className="field">
              <label>Geography</label>
              <select name="Geography" value={form.Geography} onChange={handleChange}>
                <option value="">Select country</option>
                <option value="France">France</option>
                <option value="Germany">Germany</option>
                <option value="Spain">Spain</option>
              </select>
            </div>

            <div className="field">
              <label>Active Member</label>
              <select name="IsActiveMember" value={form.IsActiveMember} onChange={handleChange}>
                <option value="">Select status</option>
                <option value="1">Yes</option>
                <option value="0">No</option>
              </select>
            </div>

          </div>

          <button
            className={`btn ${!isFormComplete || loading ? "btn-disabled" : ""}`}
            onClick={handleSubmit}
            disabled={!isFormComplete || loading}
          >
            {loading ? "Analysing..." : "Predict Churn"}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="error-card">
            <span>⚠</span> {error}
          </div>
        )}

        {/* Result */}
        {result && (
          <div className={`result-card ${result.churn ? "result-churn" : "result-safe"}`}>
            <div className="result-icon">{result.churn ? "⚠" : "✓"}</div>
            <div className="result-text">
              <h3>{result.churn ? "High Churn Risk" : "Low Churn Risk"}</h3>
              <p>{result.churn
                ? "This customer is likely to leave. Consider a retention strategy."
                : "This customer is likely to stay."
              }</p>
            </div>
            <div className="result-prob">
              <div className="prob-value">{Math.round(result.probability * 100)}%</div>
              <div className="prob-label">Churn Probability</div>
              <div className="prob-bar-bg">
                <div
                  className="prob-bar-fill"
                  style={{ width: `${result.probability * 100}%` }}
                />
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}