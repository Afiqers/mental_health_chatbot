document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const messagesContainer = document.getElementById('messages-container');

    // Function to add a message to the chat
    function addMessage(text, isUser = true) {
        if (!text.trim()) return;

        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');

        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        messageContent.textContent = text;

        messageDiv.appendChild(messageContent);
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

            // Simulate bot response after a delay
            setTimeout(() => {
                const responses = [
                    "That sounds interesting. Tell me more.",
                    "I understand. Take a deep breath.",
                    "How does that make you feel?",
                    "Remember, you are doing your best.",
                    "I'm here to listen."
                ];
                const randomResponse = responses[Math.floor(Math.random() * responses.length)];
                addMessage(randomResponse, false);
            }, 1000);
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
