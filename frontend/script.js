document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const messagesContainer = document.getElementById('messages-container');
    const newChatBtn = document.getElementById('new-chat-btn');
    
    // Auth UI elements
    const authSection = document.getElementById('auth-section');
    const userProfile = document.getElementById('user-profile');
    const usernameDisplay = document.getElementById('username-display');
    const loginBtn = document.getElementById('login-modal-btn');
    const registerBtn = document.getElementById('register-modal-btn');
    const logoutBtn = document.getElementById('logout-btn');
    
    const authModal = document.getElementById('auth-modal');
    const closeModalBtn = document.getElementById('close-modal');
    const modalTitle = document.getElementById('modal-title');
    const authSubmitBtn = document.getElementById('auth-submit-btn');
    const authUsernameInput = document.getElementById('auth-username');
    const authPasswordInput = document.getElementById('auth-password');
    const authError = document.getElementById('auth-error');

    let currentAuthMode = 'login'; // 'login' or 'register'
    let isFirstMessage = true;

    // Check if logged in on load
    checkAuthStatus();

    function checkAuthStatus() {
        const token = localStorage.getItem('chat_token');
        const username = localStorage.getItem('chat_username');
        if (token && username) {
            authSection.classList.add('hidden');
            userProfile.classList.remove('hidden');
            usernameDisplay.textContent = username;
            loadHistory();
        } else {
            authSection.classList.remove('hidden');
            userProfile.classList.add('hidden');
            addWelcomeMessage();
        }
    }

    function addWelcomeMessage() {
        messagesContainer.innerHTML = '';
        const msg = document.createElement('div');
        msg.className = 'message bot-message';
        msg.innerHTML = '<div class="message-content">Hello! I\'m Qbot. How are you feeling today?</div>';
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
        addWelcomeMessage();
    }

    newChatBtn.addEventListener('click', startNewChat);

    // Call API /history
    function loadHistory() {
        const token = localStorage.getItem('chat_token');
        if (!token) return;

        fetch('http://127.0.0.1:5000/history', {
            method: 'GET',
            headers: { 'Authorization': `Bearer ${token}` }
        })
        .then(res => res.json())
        .then(data => {
            if (data.history && data.history.length > 0) {
                messagesContainer.innerHTML = '';
                isFirstMessage = false; 
                data.history.forEach(msg => {
                    addMessage(msg.content, msg.role === 'user', { emotion: msg.emotion, confidence: msg.confidence });
                });
            } else {
                addWelcomeMessage();
            }
        })
        .catch(err => {
            console.error('Failed to load history', err);
            addWelcomeMessage();
        });
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

            const token = localStorage.getItem('chat_token');
            const headers = { 'Content-Type': 'application/json' };
            if (token) {
                headers['Authorization'] = `Bearer ${token}`;
            }

            // Fetch response from backend
            fetch('http://127.0.0.1:5000/chat', {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({ message: text, isFirstMessage: isFirstMessage })
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

    // --- Modal Logic ---
    function openModal(mode) {
        currentAuthMode = mode;
        modalTitle.textContent = mode === 'login' ? 'Log In' : 'Sign Up';
        authModal.style.display = 'flex';
        authError.style.display = 'none';
        authUsernameInput.value = '';
        authPasswordInput.value = '';
        authUsernameInput.focus();
    }

    function closeModal() {
        authModal.style.display = 'none';
    }

    loginBtn.addEventListener('click', () => openModal('login'));
    registerBtn.addEventListener('click', () => openModal('register'));
    closeModalBtn.addEventListener('click', closeModal);

    authSubmitBtn.addEventListener('click', () => {
        const username = authUsernameInput.value.trim();
        const password = authPasswordInput.value;
        
        if (!username || !password) {
            authError.textContent = "Please fill in all fields.";
            authError.style.display = 'block';
            return;
        }
        
        authSubmitBtn.disabled = true;
        authSubmitBtn.textContent = 'Loading...';

        const endpoint = currentAuthMode === 'login' ? '/login' : '/register';
        
        fetch(`http://127.0.0.1:5000${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        })
        .then(async res => {
            const data = await res.json();
            if (!res.ok) throw new Error(data.message || 'Authentication failed');
            return data;
        })
        .then(data => {
            if (currentAuthMode === 'register') {
                openModal('login');
                authUsernameInput.value = username;
                authError.textContent = "Registration successful. Please log in.";
                authError.style.color = "green";
                authError.style.display = 'block';
            } else {
                localStorage.setItem('chat_token', data.access_token);
                localStorage.setItem('chat_username', data.username);
                closeModal();
                checkAuthStatus();
            }
        })
        .catch(err => {
            authError.textContent = err.message;
            authError.style.color = "red";
            authError.style.display = 'block';
        })
        .finally(() => {
            authSubmitBtn.disabled = false;
            authSubmitBtn.textContent = 'Submit';
        });
    });

    logoutBtn.addEventListener('click', () => {
        localStorage.removeItem('chat_token');
        localStorage.removeItem('chat_username');
        checkAuthStatus();
        startNewChat();
    });

    messageInput.focus();
});
