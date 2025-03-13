"use client";

import { useState, useRef, useEffect } from "react";
import { Send, MessageCircle, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export default function ChatbotInterface() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [typingMessage, setTypingMessage] = useState(null);
  const [displayedText, setDisplayedText] = useState("");
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
    }
  }, [messages, isOpen, displayedText]);

  // Typing animation effect
  useEffect(() => {
    if (!typingMessage) return;

    const fullText = typingMessage.content;
    let currentIndex = 0;
    const typingSpeed = 20; // milliseconds per character

    const typeNextChar = () => {
      if (currentIndex < fullText.length) {
        setDisplayedText(fullText.substring(0, currentIndex + 1));
        currentIndex++;
        setTimeout(typeNextChar, typingSpeed);
      } else {
        // Animation complete, add message to regular messages
        setMessages(prev => [...prev, typingMessage]);
        setTypingMessage(null);
        setDisplayedText("");
      }
    };

    typeNextChar();
  }, [typingMessage]);

  const handleSend = async () => {
    if (!input.trim()) return;

    setIsLoading(true);
    const userMessage = input;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ user_message: userMessage }),
      });

      const data = await response.json();
      
      // Start typing animation
      setTypingMessage({ role: "assistant", content: data.response });
      
    } catch (error) {
      console.error("Error sending message:", error);
      setTypingMessage({ 
        role: "assistant", 
        content: "Sorry, I encountered an error processing your request."
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed bottom-4 right-4 z-50">
      {isOpen ? (
        <Card className="w-96 shadow-xl">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>AI Assistant</CardTitle>
            <Button 
              variant="ghost" 
              size="icon" 
              onClick={() => setIsOpen(false)}
            >
              <X className="h-4 w-4" />
            </Button>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col h-[400px]">
              <div className="flex-1 mb-4 p-4 border rounded-md overflow-y-auto">
                <div className="space-y-4">
                  {messages.length === 0 && !typingMessage ? (
                    <p className="text-sm text-muted-foreground">
                      Ask me about social media trends and insights...
                    </p>
                  ) : (
                    <>
                      {messages.map((message, index) => (
                        <div
                          key={index}
                          className={`flex ${
                            message.role === "user" ? "justify-end" : "justify-start"
                          }`}
                        >
                          <div
                            className={`max-w-[80%] rounded-lg p-3 ${
                              message.role === "user"
                                ? "bg-primary text-primary-foreground text-right"
                                : "bg-muted"
                            }`}
                          >
                            <p className="text-sm">{message.content}</p>
                          </div>
                        </div>
                      ))}
                      
                      {/* Typing animation message */}
                      {typingMessage && (
                        <div className="flex justify-start">
                          <div className="max-w-[80%] rounded-lg p-3 bg-muted">
                            <p className="text-sm">{displayedText}</p>
                          </div>
                        </div>
                      )}
                    </>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              </div>
              <div className="flex gap-2">
                <Input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Type your question..."
                  className="flex-1"
                  onKeyPress={(e) => e.key === "Enter" && !isLoading && handleSend()}
                  disabled={isLoading || typingMessage !== null}
                />
                <Button 
                  size="icon" 
                  onClick={handleSend} 
                  disabled={isLoading || typingMessage !== null || !input.trim()}
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      ) : (
        <Button
          className="rounded-full w-12 h-12 shadow-lg"
          onClick={() => setIsOpen(true)}
        >
          <MessageCircle className="h-6 w-6" />
        </Button>
      )}
    </div>
  );
}