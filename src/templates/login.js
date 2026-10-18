
"use strict";

// ============================================================
// Sliding login / signup interface
// ============================================================

const signupTab = document.querySelector(".signup");
const loginTab = document.querySelector(".login");
const slider = document.querySelector(".slider");
const formSection = document.querySelector(".form-section");

const authMsg = document.getElementById("auth-msg");

const loginBtn = document.getElementById("login-btn");
const signupBtn = document.getElementById("signup-btn");

function showMsg(message, success = false) {
    authMsg.textContent = message;
    authMsg.style.color = success ? "green" : "crimson";
}

signupTab.addEventListener("click", () => {
    slider.classList.add("moveslider");
    formSection.classList.add("form-section-move");
    showMsg("");
});

loginTab.addEventListener("click", () => {
    slider.classList.remove("moveslider");
    formSection.classList.remove("form-section-move");
    showMsg("");
});

// ============================================================
// Session ID shared with index.html
// ============================================================

let sessionId = sessionStorage.getItem("sid");

if (!sessionId) {
    sessionId = crypto.randomUUID();
    sessionStorage.setItem("sid", sessionId);
}

// ============================================================
// API request
// ============================================================

async function submitAuth(url, payload) {
    loginBtn.disabled = true;
    signupBtn.disabled = true;

    showMsg("Please wait...");

    try {
        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                session_id: sessionId,
                ...payload
            })
        });

        let data;

        try {
            data = await response.json();
        } catch {
            throw new Error(
                `The server returned an invalid response (${response.status}).`
            );
        }

        if (!response.ok || !data.ok) {
            showMsg(
                data.message ||
                `Authentication failed (${response.status}).`
            );
            return;
        }

        showMsg(data.message || "Authentication successful.", true);

        // main.py associates the authenticated customer with this session ID.
        // Preserve the ID when navigating to the chat page.
        window.location.replace("/");

    } catch (error) {
        console.error("Authentication request failed:", error);

        showMsg(
            error.message || "Could not connect to the server."
        );
    } finally {
        loginBtn.disabled = false;
        signupBtn.disabled = false;
    }
}

// ============================================================
// Login
// ============================================================

loginBtn.addEventListener("click", async () => {
    const email = document
        .getElementById("login-email")
        .value
        .trim();

    const password = document
        .getElementById("login-password")
        .value;

    if (!email || !password) {
        showMsg("Enter your email and password.");
        return;
    }

    await submitAuth("/api/signin", {
        email,
        password
    });
});

// ============================================================
// Signup
// ============================================================

signupBtn.addEventListener("click", async () => {
    const name = document
        .getElementById("signup-name")
        .value
        .trim();

    const email = document
        .getElementById("signup-email")
        .value
        .trim();

    const password = document
        .getElementById("signup-password")
        .value;

    const confirmation = document
        .getElementById("signup-confirm")
        .value;

    if (!name || !email || !password || !confirmation) {
        showMsg("Complete all fields.");
        return;
    }

    if (password !== confirmation) {
        showMsg("Passwords do not match.");
        return;
    }

    await submitAuth("/api/signup", {
        name,
        email,
        password
    });
});

// ============================================================
// Submit using Enter
// ============================================================

document.querySelectorAll(".login-box input").forEach(element => {
    element.addEventListener("keydown", event => {
        if (event.key === "Enter") {
            event.preventDefault();
            loginBtn.click();
        }
    });
});

document.querySelectorAll(".signup-box input").forEach(element => {
    element.addEventListener("keydown", event => {
        if (event.key === "Enter") {
            event.preventDefault();
            signupBtn.click();
        }
    });
});
