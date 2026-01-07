document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const messagesContainer = document.getElementById('messages-container');

    // Function to add a message to the chat
    function addMessage(text, isUser = true, meta = null) {
        if (!text.trim()) return;

        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');

        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        messageContent.textContent = text;

        messageDiv.appendChild(messageContent);

        // Add metadata (emotion/confidence) if provided
        if (meta && !isUser) {
            const metaDiv = document.createElement('div');
            metaDiv.classList.add('message-meta');
            const confidencePercent = Math.round(meta.confidence * 100);
            metaDiv.textContent = `Detected: ${meta.emotion} (${confidencePercent}%)`;
            messageDiv.appendChild(metaDiv);
        }

        messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

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

            // Fetch response from backend
            fetch('http://127.0.0.1:5000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: text })
            })
                .then(response => response.json())
                .then(data => {
                    // Remove typing indicator
                    messagesContainer.removeChild(typingIndicator);

                    // Add bot response with meta data
                    if (data.response) {
                        addMessage(data.response, false, {
                            emotion: data.emotion,
                            confidence: data.confidence
                        });
                    } else {
                        addMessage("I'm having trouble understanding right now. Please try again.", false);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    messagesContainer.removeChild(typingIndicator);
                    addMessage("Sorry, I'm having trouble connecting to the server.", false);
                });
        }
    }

    // Event listeners
    sendBtn.addEventListener('click', handleSendMessage);

    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSendMessage();
        }
    });

    // Focus input on load
    messageInput.focus();
});
