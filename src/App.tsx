import { useState, useRef, useEffect } from 'react';
import './App.css'
import { generateReplyStream } from './ai';

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

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isThinking) return;

    const userText = input.trim();
    const userMsg = { id: Date.now(), text: userText, sender: 'user' } as Message;
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setIsThinking(true);

    const aiMsgId = Date.now() + 1;
    setMessages((prev) => [...prev, { id: aiMsgId, text: '', sender: 'ai' }]);

    try {
      const finalText = await generateReplyStream(
        userText,
        (_tokenId: number, tokenText: string) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === aiMsgId ? { ...m, text: m.text + tokenText } : m)),
          );
        },
        120,
      );

      // Ensure final text is reflected (in case decoding differs from streamed pieces)
      if (finalText && finalText.length > 0) {
        setMessages((prev) =>
          prev.map((m) => (m.id === aiMsgId ? { ...m, text: finalText } : m)),
        );
      }
    } catch (error) {
      console.error('generateReplyStream failed', error);
      setMessages((prev) =>
        prev.map((m) => (m.id === aiMsgId ? { ...m, text: 'Sorry, something went wrong.' } : m)),
      );
    } finally {
      setIsThinking(false);
    }
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
