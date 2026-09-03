import React from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

function App() {
  return <div><h1>Career Matcher</h1><p>Frontend is running.</p></div>;
}

createRoot(document.getElementById('root')).render(<App />);
