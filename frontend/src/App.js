import { useState } from "react";
import "./App.css";

const API_URL = process.env.REACT_APP_API_URL;
const API_KEY = process.env.REACT_APP_API_KEY;

function App() {
  const [image, setImage] = useState(null);      // selected file
  const [preview, setPreview] = useState(null);  // image preview URL
  const [caption, setCaption] = useState("");    // returned caption
  const [loading, setLoading] = useState(false); // loading state
  const [error, setError] = useState("");        // error message

  function handleImageChange(e) {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setCaption("");
    setError("");
  }

  async function handleSubmit() {
    if (!image) return;

    const formData = new FormData();
    formData.append("file", image);

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/caption`, {
        method: "POST",
        headers: { "X-API-Key": API_KEY },
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setCaption(data.caption);
      } else {
        setError(data.detail || "Something went wrong.");
      }
    } catch (err) {
      setError("Could not reach the API. Is it running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <h1>BLIP Image Captioning</h1>
      <p>Upload an image and get an AI-generated caption</p>

      <input type="file" accept="image/*" onChange={handleImageChange} />

      {preview && <img src={preview} alt="preview" className="preview" />}

      <button onClick={handleSubmit} disabled={!image || loading}>
        {loading ? "Generating..." : "Generate Caption"}
      </button>

      {caption && <div className="caption"><strong>Caption:</strong> {caption}</div>}
      {error && <div className="error">{error}</div>}
    </div>
  );
}

export default App;
