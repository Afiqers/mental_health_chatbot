document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const messagesContainer = document.getElementById('messages-container');
    const newChatBtn = document.getElementById('new-chat-btn');
    
    let isFirstMessage = true;
    let conversationHistory = []; // [{role: 'user'|'bot', content: '...'}]
    const MAX_HISTORY = 10;       // keep last 10 turns (5 back-and-forth exchanges)

    // Initialize UI
    addWelcomeMessage();

    function addWelcomeMessage() {
        messagesContainer.innerHTML = '';
        const msg = document.createElement('div');
        msg.className = 'message bot-message';
        msg.innerHTML = '<div class="message-content">Hello! I\'m Mindbot. How are you feeling today?</div>';
        messagesContainer.appendChild(msg);
    }

    // Function to add a message to the chat
    function addMessage(text, isUser = true, meta = null) {
        if (!text.trim()) return;

        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');

        let cleanText = text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
        cleanText = cleanText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        messageContent.innerHTML = cleanText;

        messageDiv.appendChild(messageContent);

        // Add metadata (emotion/confidence) if provided
        if (meta && meta.emotion && !isUser) {
            const metaDiv = document.createElement('div');
            metaDiv.classList.add('message-meta');
            const confidencePercent = Math.round(meta.confidence * 100);
            metaDiv.textContent = `Detected: ${meta.emotion} (${confidencePercent}%)`;
            messageDiv.appendChild(metaDiv);
        }

        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function startNewChat() {
        messagesContainer.innerHTML = '';
        isFirstMessage = true;
        conversationHistory = [];   // reset context on new chat
        addWelcomeMessage();
    }

    newChatBtn.addEventListener('click', startNewChat);

    // Function to handle sending message
    function handleSendMessage() {
        const text = messageInput.value;
        if (text.trim()) {
            addMessage(text, true);
            messageInput.value = '';

            // Show typing indicator
            const typingIndicator = document.createElement('div');
            typingIndicator.classList.add('message', 'bot-message', 'typing-indicator');
            typingIndicator.innerHTML = `
                <div class="message-content">
                    <span class="dot"></span>
                    <span class="dot"></span>
                    <span class="dot"></span>
                </div>
            `;
            messagesContainer.appendChild(typingIndicator);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;

            // Add user message to history
            conversationHistory.push({ role: 'user', content: text });
            if (conversationHistory.length > MAX_HISTORY) conversationHistory.shift();

            const headers = { 'Content-Type': 'application/json' };

            // Fetch response from backend — include last N turns as history
            fetch('http://127.0.0.1:5000/chat', {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({
                    message: text,
                    isFirstMessage: isFirstMessage,
                    history: conversationHistory.slice(0, -1) // all turns BEFORE current message
                })
            })
            .then(response => {
                if (!response.ok) throw new Error("HTTP error " + response.status);
                return response.json();
            })
            .then(data => {
                if (messagesContainer.contains(typingIndicator)) {
                    messagesContainer.removeChild(typingIndicator);
                }
                isFirstMessage = false;

                if (data.response) {
                    // Add bot reply to history
                    conversationHistory.push({ role: 'bot', content: data.response });
                    if (conversationHistory.length > MAX_HISTORY) conversationHistory.shift();

                    addMessage(data.response, false, { emotion: data.emotion, confidence: data.confidence });
                } else {
                    addMessage("I'm having trouble understanding right now. Please try again.", false);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                if (messagesContainer.contains(typingIndicator)) {
                    messagesContainer.removeChild(typingIndicator);
                }
                addMessage("Sorry, I'm having trouble connecting to the server.", false);
            });
        }
    }

    sendBtn.addEventListener('click', handleSendMessage);
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSendMessage();
    });

    messageInput.focus();
});
