import { useState, useRef, useEffect } from 'react';
import './App.css'
import { generateReplyStream, type Message } from './ai';

function App() {
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, content: "I am an AI agent which runs locally on you machine!", role: 'assistant' },
    { id: 2, content: "I done this so all data privacy stays with you!", role: 'assistant' },
    { id: 3, content: "I am a an empathetic, non-judgmental therapist. Who listen, validate feelings, and ask open-ended questions to guide self-reflection. Keep responses concise, supportive, and focused on the user. I do not give medical advice.", role: 'system' },
    { id: 4, content: "Hello! My name is Linda. How can I help you today?", role: 'assistant' }
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
    const userMsg = { id: Date.now(), content: userText, role: 'user' } as Message;
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setIsThinking(true);

    const aiMsgId = Date.now() + 1;
    setMessages((prev) => [...prev, { id: aiMsgId, content: '', role: 'assistant' }]);

    try {
      const finalText = await generateReplyStream(
        messages,
        userText,
        (_tokenId: number, tokenText: string) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === aiMsgId ? { ...m, content: m.content + tokenText } : m)),
          );
        },
        120,
      );

      // Ensure final text is reflected (in case decoding differs from streamed pieces)
      if (finalText && finalText.length > 0) {
        setMessages((prev) =>
          prev.map((m) => (m.id === aiMsgId ? { ...m, content: finalText } : m)),
        );
      }
    } catch (error) {
      console.error('generateReplyStream failed', error);
      setMessages((prev) =>
        prev.map((m) => (m.id === aiMsgId ? { ...m, content: 'Sorry, something went wrong.' } : m)),
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
            <div key={msg.id} className={`message-bubble ${msg.role === 'user' ? 'user-message' : 'ai-message'}`}>
              {msg.content}
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
