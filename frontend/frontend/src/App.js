import React, { useState, useEffect, useRef } from 'react';
import { BrowserRouter as Router, Route, Link, Routes } from 'react-router-dom';
import './App.css';
import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';


// Generate or retrieve the session ID
const getSessionId = () => {
  let sessionId = localStorage.getItem('sessionId');
  if (!sessionId) {
    sessionId = uuidv4();
    localStorage.setItem('sessionId', sessionId);
  }
  return sessionId;
};


// Mock send message to agent (simulated response)

const sendMessageToAgent = async (message, sessionId) => {
  try {
    const response = await axios.post('http://localhost:8000/api/message', {
      message,
      sessionId,
    });

    return { text: response.data.text };
  } catch (error) {
    console.error('❌ Error contacting agent:', error);
    return { text: "Sorry, couldn't reach the agent." };
  }
};

function Disclaimer() {
  return (
    <div className="disclaimer">
      CouchGPT is a chatbot and cannot replace a therapist. Please seek immediate help in emergencies.
    </div>
  );
}

// About Us component
function AboutUs({ resetChat }) {
  const handleStartNewChat = () => {
    resetChat(); // Reset chat
    window.location.href = '/'; // Navigate back to Home using window.location
  };

  return (
    <div>
      <div className="about-us-container">
        <div className="team">
          <img className="chat-box-image" src="/couchgpt.png" alt="Chatbox Top Image" />
          <div className="team-member">
            <a href="https://www.linkedin.com/in/karim-gowani" target="_blank" rel="noopener noreferrer">
              <img src="/karim.png" alt="Karim Gowani" />
            </a>
            <h3>Karim Gowani</h3>
          </div>

          <div className="team-member">
            <a href="https://www.linkedin.com/in/imran-naskani" target="_blank" rel="noopener noreferrer">
              <img src="/imran.png" alt="Imran Naskani" />
            </a>
            <h3>Imran Naskani</h3>
          </div>

          <div className="team-member">
            <a href="https://www.linkedin.com/in/sandra-forro" target="_blank" rel="noopener noreferrer">
              <img src="/sandra.png" alt="Sandra Forro" />
            </a>
            <h3>Sandra Forro</h3>
          </div>
        </div>

        <div className="about-us">
          <p>CouchGPT is a chatbot created to provide mental health support, brought to you by students at Harvard Extension School.</p>
        </div>
      </div>
      <Disclaimer />
    </div>
  );
}

function Sources() {
  return (
    <div>
      <div className="sources-container">
        <img className="chat-box-image" src="/couchgpt.png" alt="Chatbox Top Image" />
        <div className="sources-row">
          <div className="resource-box">
            <a href="https://www.amazon.de/-/en/David-D-Burns-M-D/dp/0380731762" target="_blank" rel="noopener noreferrer">
              <img src="/feelinggood.jpg" alt="Book Title 1" />
              <div className="resource-text">
                <h3>Feeling Good | The New Mood Therapy 1</h3>
              </div>
            </a>
          </div>

          <div className="resource-box">
            <a href="https://www.amazon.com/Practical-Psychology-Karl-Schofield-Bernhardt/dp/B004MTGMK2" target="_blank" rel="noopener noreferrer">
              <img src="/practicalpsych.jpg" alt="Book Title 2" />
              <div className="resource-text">
                <h3>Practical PsychologyBook Title 2</h3>
              </div>
            </a>
          </div>
        </div>

        <div className="sources-row">
          <div className="resource-box">
            <a href="https://github.com/thu-coai/Emotional-Support-Conversation?tab=readme-ov-file" target="_blank" rel="noopener noreferrer">
              <img src="/escon.JPG" alt="GitHub Repository 1" />
              <div className="resource-text">
                <h3>Emotional Support Conversation Dataset</h3>
              </div>
            </a>
          </div>

          <div className="resource-box">
            <a href="https://github.com/uccollab/AnnoMI?tab=readme-ov-file" target="_blank" rel="noopener noreferrer">
              <img src="/AnnoMI.png" alt="GitHub Repository 2" />
              <div className="resource-text">
                <h3>Annotated Motivational Interviewing Dataset</h3>
              </div>
            </a>
          </div>

          <div className="resource-box">
            <a href="https://github.com/CAS-SIAT-XinHai/CPsyCoun" target="_blank" rel="noopener noreferrer">
              <img src="/cpsycoun.png" alt="GitHub Repository 3" />
              <div className="resource-text">
                <h3>Chinese Psychological Counseling Dialog Dataset</h3>
              </div>
            </a>
          </div>
        </div>
      </div>
      <Disclaimer />
    </div>
  );
}

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [messages, setMessages] = useState(() => {
    const savedMessages = localStorage.getItem('messages');
    return savedMessages ? JSON.parse(savedMessages) : [
      { text: 'Welcome to the Couch! How are you feeling today?', isUser: false, timestamp: new Date().toISOString() },
    ];
  });
  const [newMessage, setNewMessage] = useState('');
  const [sessionId, setSessionId] = useState(getSessionId());
  const chatBoxRef = useRef(null);
  const [allSessions, setAllSessions] = useState([]);

  useEffect(() => {
    const storedSessions = Object.keys(localStorage)
      .filter((key) => key.startsWith('session_'))
      .map((key) => ({
        id: key.replace('session_', ''),
        messages: JSON.parse(localStorage.getItem(key)),
      }));

    setAllSessions(storedSessions);
    setSessionId(localStorage.getItem('sessionId') || getSessionId());
  }, []);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const handleSendMessage = async () => {
    if (newMessage.trim()) {
      const userMessage = { text: newMessage, isUser: true, timestamp: new Date().toISOString() };
      const updatedMessages = [...messages, userMessage];
      setMessages(updatedMessages);
      setNewMessage('');

      localStorage.setItem(`session_${sessionId}`, JSON.stringify(updatedMessages));

      const response = await sendMessageToAgent(newMessage, sessionId);
      const botMessage = { text: response.text, isUser: false, timestamp: new Date().toISOString() };

      const finalMessages = [...updatedMessages, botMessage];
      setMessages(finalMessages);

      localStorage.setItem(`session_${sessionId}`, JSON.stringify(finalMessages));
    }
  };

  const resetChat = () => {
    const initialMessage = [{ text: 'Welcome to the Couch! How are you feeling today?', isUser: false, timestamp: new Date().toISOString() }];
    const newId = uuidv4();
    setSessionId(newId);
    setMessages(initialMessage);
    localStorage.setItem('sessionId', newId);
    localStorage.setItem(`session_${newId}`, JSON.stringify(initialMessage));
  };

  const handleSessionClick = (sessionId) => {
    const sessionMessages = JSON.parse(localStorage.getItem(`session_${sessionId}`));
    setSessionId(sessionId);
    setMessages(sessionMessages);
  };

  const getPreview = (messages) => {
    const userMessage = messages.find((msg) => msg.isUser && msg.text.length > 5);
    return userMessage ? userMessage.text.slice(0, 40) : 'Untitled Conversation';
  };

  const loadSession = (id) => {
    const savedMessages = localStorage.getItem(`session_${id}`);
    if (savedMessages) {
      setMessages(JSON.parse(savedMessages));
      setSessionId(id);
    }
  };

  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <Router>
      <div className="app">
        <div className={`sidebar ${isSidebarOpen ? '' : 'closed'}`}>
          <h2>Past Conversations</h2>
          <ul>
            {allSessions.map((session) => (
              <li key={session.id} onClick={() => loadSession(session.id)}>
                {getPreview(session.messages)}
              </li>
            ))}
          </ul>
        </div>

        <button className="toggle-sidebar-btn" onClick={toggleSidebar}>
          <i className="fas fa-bars"></i>
        </button>

        <div className={`logo-container ${isSidebarOpen ? 'logo-expanded' : ''}`}>
          <button className="new-chat-btn" onClick={() => { resetChat(); window.location.href = '/'; }}>
            <i className="fa-solid fa-comment"></i>
          </button>
          <span className="logo-text">CouchGPT</span>
        </div>

        <div className="links-container">
          <Link to="/" className="link">Home</Link>
          <Link to="/about" className="link">About Us</Link>
          <Link to="/sources" className="link">Sources</Link>
        </div>

        <Routes>
          <Route path="/about" element={<AboutUs resetChat={resetChat} />} />
          <Route path="/sources" element={<Sources />} />
          <Route path="/" element={
            <div className="chat-container">
              <div style={{
                fontSize: '18px',
                color: '#fff',
                marginBottom: '10px',
                textAlign: 'center',
              }}>
                Session ID: {sessionId}
              </div>
              <img className="chat-box-image" src="/couchgpt.png" alt="Chatbox Top Image" />
              <div className="chat-box" ref={chatBoxRef}>
                {messages.map((msg, index) => (
                  <div key={index} className={`chat-message ${msg.isUser ? 'user-message' : 'bot-message'}`}>
                    <div className="message-text">{msg.text}</div>
                    <div className="timestamp">
                      {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                ))}
              </div>

              <div className="chat-input">
                <input
                  type="text"
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  placeholder="Type a message..."
                />
                <button onClick={handleSendMessage}>Send</button>
              </div>

              <Disclaimer />
            </div>
          } />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
