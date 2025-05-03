function toggleForms() {
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');
    
    if (loginForm.style.display === 'none') {
        loginForm.style.display = 'block';
        signupForm.style.display = 'none';
    } else {
        loginForm.style.display = 'none';
        signupForm.style.display = 'block';
    }
    
    // Clear any error/success messages
    document.querySelectorAll('.error, .success').forEach(el => el.style.display = 'none');
}

async function handleLogin(event) {
    event.preventDefault();
    
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    const errorElement = document.getElementById('loginError');
    
    try {
        const response = await fetch('/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
		'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            },
            body: JSON.stringify({ email, password }),
        });
	const data = await response.json();
	console.log(data);
        
        if (response.ok && data.redirect) {
            window.location.href = data.redirect;
        } else {
            errorElement.textContent = data.detail || 'Login failed';
            errorElement.style.display = 'block';
        }
        
    } catch (error) {
	console.log(error);
        errorElement.textContent = '2 An error occurred. Please try again.';
        errorElement.style.display = 'block';
    }
    
    return false;
}

async function handleSignup(event) {
    event.preventDefault();
    
    const email = document.getElementById('signupEmail').value;
    const password = document.getElementById('signupPassword').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
    const errorElement = document.getElementById('signupError');
    const successElement = document.getElementById('signupSuccess');
    
    // Clear previous messages
    errorElement.style.display = 'none';
    successElement.style.display = 'none';
    
    // Basic validation
    if (password !== confirmPassword) {
        errorElement.textContent = 'Passwords do not match';
        errorElement.style.display = 'block';
        return false;
    }
    
    if (password.length < 8) {
        errorElement.textContent = 'Password must be at least 8 characters long';
        errorElement.style.display = 'block';
        return false;
    }
    
    try {
        const response = await fetch('/auth/signup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password }),
        });
        
        const data = await response.json();
        
        if (response.ok) {
            successElement.textContent = 'Account created successfully! You can now login.';
            successElement.style.display = 'block';
            // Clear the form
            event.target.reset();
            // Switch to login form after a delay
            setTimeout(() => {
                toggleForms();
            }, 2000);
        } else {
            errorElement.textContent = data.error || 'Signup failed';
            errorElement.style.display = 'block';
        }
    } catch (error) {
	console.log(error);
        errorElement.textContent = '1 An error occurred. Please try again.';
        errorElement.style.display = 'block';
    }
    
    return false;
}
