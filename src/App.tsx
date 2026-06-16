import { useState, useRef, useEffect } from 'react';
import './App.css'

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'ai';
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, text: "Hello! How can I help you today?", sender: 'ai' }
  ]);
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isThinking) return;

    const newMessage: Message = {
      id: Date.now(),
      text: input,
      sender: 'user',
    };

    setIsThinking(true);

    setMessages([...messages, newMessage]);
    setInput('');

    // Simulate AI response
    setTimeout(() => {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        text: "This is a clean, minimal response.",
        sender: 'ai'
      }]);
      setIsThinking(false);
    }, 1000);
  };

  return (
    <div className="app-wrapper">
      <div className="chat-container">
        <header className="header">
          <h1>Therapy Chat</h1>
          <div style={{ fontSize: '0.8rem', color: '#999' }}>This is a AI and not a human so take things with a gain of salt</div>
        </header>

        <div className="message-list">
          {messages.map((msg) => (
            <div key={msg.id} className={`message-bubble ${msg.sender === 'user' ? 'user-message' : 'ai-message'}`}>
              {msg.text}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSend} className="input-form">
          <div className="input-wrapper">
            <input
              type="text"
              className="text-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={isThinking ? "AI is thinking..." : "Type a message..."}
              disabled={isThinking}
            />
            <button type="submit" className="send-button" disabled={!input.trim() || isThinking}>
              Send
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default App;
