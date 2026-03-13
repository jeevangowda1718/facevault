// FaceVault — Global JavaScript Utilities

// Handle page unload - stop cameras gracefully
window.addEventListener('beforeunload', () => {
    const path = window.location.pathname;
    if (path === '/register') {
        navigator.sendBeacon('/stop_camera/register');
    } else if (path === '/detect') {
        navigator.sendBeacon('/stop_camera/detect');
    }
});

// Keyboard shortcut: Enter key on name input triggers camera start (register page)
document.addEventListener('DOMContentLoaded', () => {
    const nameInput = document.getElementById('name-input');
    if (nameInput) {
        nameInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const startBtn = document.getElementById('start-btn');
                if (startBtn && startBtn.style.display !== 'none') {
                    startCamera();
                } else {
                    captureFrames();
                }
            }
        });
    }
});
