/* Mindbot SPA — auth, chat with persisted sessions, and a mood dashboard. */

const API = 'http://127.0.0.1:5000';

const EMOTION_COLORS = {
    NORMAL: '#30a46c', NEUTRAL: '#8a90a6', GREETING: '#6c8cff', FAREWELL: '#9d7bff',
    STRESS: '#f5a623', ANXIETY: '#f5a623', DEPRESSION: '#e08e0b',
    SUICIDAL: '#e5484d', HIGH_RISK: '#e5484d', UNKNOWN: '#8a90a6',
};

const State = {
    token: localStorage.getItem('mindbot_token') || null,
    user: null,
    conversationId: null,
    authMode: 'login',
    moodChart: null,
    emotionChart: null,
};

/* ───────────── API helper ───────────── */
async function api(path, { method = 'GET', body = null, auth = true } = {}) {
    const headers = { 'Content-Type': 'application/json' };
    if (auth && State.token) headers['Authorization'] = 'Bearer ' + State.token;
    const res = await fetch(API + path, {
        method, headers, body: body ? JSON.stringify(body) : null,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || ('Request failed (' + res.status + ')'));
    return data;
}

/* ───────────── DOM refs ───────────── */
const $ = (id) => document.getElementById(id);

document.addEventListener('DOMContentLoaded', () => {
    setupAuthUI();
    setupAppUI();
    boot();
});

/* ───────────── Boot ───────────── */
async function boot() {
    if (State.token) {
        try {
            const { user } = await api('/auth/me');
            State.user = user;
            enterApp();
            return;
        } catch (_) {
            localStorage.removeItem('mindbot_token');
            State.token = null;
        }
    }
    showAuth();
}

function showAuth() {
    $('auth-view').style.display = 'flex';
    $('app-view').style.display = 'none';
}

function enterApp() {
    $('auth-view').style.display = 'none';
    $('app-view').style.display = 'flex';
    $('user-badge').textContent = State.user ? State.user.name : '';
    switchView('chat');
    loadConversations();
    startNewChat();
}

/* ───────────── Auth UI ───────────── */
function setupAuthUI() {
    document.querySelectorAll('.auth-tab').forEach((tab) => {
        tab.addEventListener('click', () => {
            State.authMode = tab.dataset.tab;
            document.querySelectorAll('.auth-tab').forEach((t) => t.classList.remove('active'));
            tab.classList.add('active');
            $('name-field').style.display = State.authMode === 'register' ? 'block' : 'none';
            $('auth-submit').textContent = State.authMode === 'register' ? 'Create account' : 'Log in';
            $('auth-error').textContent = '';
        });
    });

    $('auth-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        $('auth-error').textContent = '';
        const btn = $('auth-submit');
        btn.disabled = true;
        try {
            const payload = {
                email: $('auth-email').value.trim(),
                password: $('auth-password').value,
            };
            let data;
            if (State.authMode === 'register') {
                payload.name = $('auth-name').value.trim();
                data = await api('/auth/register', { method: 'POST', body: payload, auth: false });
            } else {
                data = await api('/auth/login', { method: 'POST', body: payload, auth: false });
            }
            State.token = data.token;
            State.user = data.user;
            localStorage.setItem('mindbot_token', data.token);
            enterApp();
        } catch (err) {
            $('auth-error').textContent = err.message;
        } finally {
            btn.disabled = false;
        }
    });
}

/* ───────────── App UI wiring ───────────── */
function setupAppUI() {
    $('new-chat-btn').addEventListener('click', () => { switchView('chat'); startNewChat(); });
    $('logout-btn').addEventListener('click', logout);

    document.querySelectorAll('.nav-btn').forEach((btn) => {
        btn.addEventListener('click', () => switchView(btn.dataset.view));
    });

    $('send-btn').addEventListener('click', sendMessage);
    $('user-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
}

function logout() {
    localStorage.removeItem('mindbot_token');
    State.token = null; State.user = null; State.conversationId = null;
    showAuth();
}

function switchView(view) {
    document.querySelectorAll('.nav-btn').forEach((b) =>
        b.classList.toggle('active', b.dataset.view === view));
    $('chat-panel').style.display = view === 'chat' ? 'flex' : 'none';
    $('dashboard-panel').style.display = view === 'dashboard' ? 'flex' : 'none';
    if (view === 'dashboard') loadDashboard();
}

/* ───────────── Conversations ───────────── */
async function loadConversations() {
    try {
        const { conversations } = await api('/conversations');
        const list = $('conv-list');
        list.innerHTML = '';
        conversations.forEach((c) => {
            const item = document.createElement('div');
            item.className = 'conv-item' + (c.id === State.conversationId ? ' active' : '');
            item.innerHTML = `<span class="title">${escapeHtml(c.title)}</span>
                <button class="del" title="Delete">&times;</button>`;
            item.querySelector('.title').addEventListener('click', () => openConversation(c.id));
            item.querySelector('.del').addEventListener('click', (e) => {
                e.stopPropagation();
                deleteConversation(c.id);
            });
            list.appendChild(item);
        });
    } catch (err) {
        console.error('loadConversations', err);
    }
}

async function openConversation(id) {
    switchView('chat');
    try {
        const { conversation } = await api('/conversations/' + id);
        State.conversationId = id;
        const box = $('messages-container');
        box.innerHTML = '';
        conversation.messages.forEach((m) => {
            addMessage(m.content, m.role === 'user', m.role === 'user' ? {
                emotion: m.emotion, confidence: m.confidence, is_high_risk: m.is_high_risk,
            } : null);
        });
        loadConversations();
    } catch (err) {
        console.error('openConversation', err);
    }
}

async function deleteConversation(id) {
    if (!confirm('Delete this conversation?')) return;
    try {
        await api('/conversations/' + id, { method: 'DELETE' });
        if (State.conversationId === id) startNewChat();
        loadConversations();
    } catch (err) {
        console.error('deleteConversation', err);
    }
}

function startNewChat() {
    State.conversationId = null;
    const box = $('messages-container');
    box.innerHTML = '';
    addMessage("Hello! I'm Mindbot. How are you feeling today?", false);
    loadConversations();
    $('user-input').focus();
}

/* ───────────── Chat ───────────── */
function addMessage(text, isUser, meta = null) {
    if (!text || !text.trim()) return;
    const box = $('messages-container');
    const wrap = document.createElement('div');
    wrap.className = 'message ' + (isUser ? 'user-message' : 'bot-message');
    if (!isUser && meta && meta.is_high_risk) wrap.classList.add('crisis');

    let clean = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    clean = clean.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = clean;
    wrap.appendChild(content);

    // Emotion chip on the *user's* message so they see how they were read.
    if (isUser && meta && meta.emotion) {
        const m = document.createElement('div');
        m.className = 'message-meta';
        const chip = document.createElement('span');
        chip.className = 'emo-chip';
        chip.style.background = EMOTION_COLORS[meta.emotion] || '#8a90a6';
        const pct = meta.confidence != null ? ` ${Math.round(meta.confidence * 100)}%` : '';
        chip.textContent = meta.emotion + pct;
        m.appendChild(chip);
        wrap.appendChild(m);
    }
    box.appendChild(wrap);
    box.scrollTop = box.scrollHeight;
    return wrap;
}

function showTyping() {
    const box = $('messages-container');
    const t = document.createElement('div');
    t.className = 'message bot-message typing-indicator';
    t.id = 'typing';
    t.innerHTML = '<div class="message-content"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>';
    box.appendChild(t);
    box.scrollTop = box.scrollHeight;
}
function hideTyping() { const t = $('typing'); if (t) t.remove(); }

async function sendMessage() {
    const input = $('user-input');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';

    // Render user message immediately (chip filled in after analysis).
    const userBubble = addMessage(text, true);
    showTyping();

    try {
        const data = await api('/chat', {
            method: 'POST',
            body: { message: text, conversation_id: State.conversationId },
        });
        hideTyping();
        State.conversationId = data.conversation_id;

        // Attach emotion chip to the user bubble retroactively.
        if (userBubble && data.emotion) {
            const m = document.createElement('div');
            m.className = 'message-meta';
            const chip = document.createElement('span');
            chip.className = 'emo-chip';
            chip.style.background = EMOTION_COLORS[data.emotion] || '#8a90a6';
            chip.textContent = data.emotion + ` ${Math.round(data.confidence * 100)}%`;
            m.appendChild(chip);
            userBubble.appendChild(m);
        }

        addMessage(data.response, false, { is_high_risk: data.is_high_risk });
        loadConversations();
    } catch (err) {
        hideTyping();
        addMessage("Sorry, I'm having trouble connecting to the server.", false);
        console.error('sendMessage', err);
    }
}

/* ───────────── Dashboard ───────────── */
async function loadDashboard() {
    try {
        const [summary, mood, emotions] = await Promise.all([
            api('/analytics/summary'),
            api('/analytics/mood?days=30'),
            api('/analytics/emotions'),
        ]);
        renderStatCards(summary);
        renderMoodChart(mood.series);
        renderEmotionChart(emotions.distribution);
    } catch (err) {
        console.error('loadDashboard', err);
    }
}

function renderStatCards(s) {
    const trendArrow = s.trend > 0.02 ? '↑ improving' : s.trend < -0.02 ? '↓ declining' : '→ steady';
    const cards = [
        { label: 'Average mood', value: s.average_sentiment.toFixed(2), sub: s.mood_band },
        { label: 'Recent trend', value: trendArrow, sub: 'last 5 vs previous 5' },
        { label: 'Messages analysed', value: s.total_messages, sub: `${s.total_conversations} conversations` },
        { label: 'High-risk flags', value: s.high_risk_messages, sub: s.high_risk_messages ? 'please seek support' : 'none detected' },
    ];
    $('stat-cards').innerHTML = cards.map((c) =>
        `<div class="stat-card"><div class="label">${c.label}</div>
         <div class="value">${c.value}</div><div class="sub">${escapeHtml(String(c.sub))}</div></div>`
    ).join('');
}

function renderMoodChart(series) {
    const ctx = $('mood-chart');
    if (State.moodChart) State.moodChart.destroy();
    if (!series.length) {
        ctx.parentElement.querySelector('h3').insertAdjacentHTML('afterend',
            '<div class="empty-hint">Chat a little and your mood trend will appear here.</div>');
        return;
    }
    State.moodChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: series.map((p) => p.date),
            datasets: [{
                label: 'Avg sentiment',
                data: series.map((p) => p.average_sentiment),
                borderColor: '#6c8cff',
                backgroundColor: 'rgba(108,140,255,0.12)',
                fill: true, tension: 0.35, pointRadius: 4,
            }],
        },
        options: {
            scales: { y: { min: -1, max: 1, ticks: { stepSize: 0.5 } } },
            plugins: { legend: { display: false } },
        },
    });
}

function renderEmotionChart(distribution) {
    const ctx = $('emotion-chart');
    if (State.emotionChart) State.emotionChart.destroy();
    if (!distribution.length) return;
    State.emotionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: distribution.map((d) => d.emotion),
            datasets: [{
                data: distribution.map((d) => d.count),
                backgroundColor: distribution.map((d) => EMOTION_COLORS[d.emotion] || '#8a90a6'),
            }],
        },
        options: { plugins: { legend: { position: 'right' } } },
    });
}

/* ───────────── utils ───────────── */
function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
        ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
