"""Authentication module for user management."""

import os
import json
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

class AuthManager:
    def __init__(self):
        self.users_file = 'users.json'
        self.users = self.load_users()
    
    def load_users(self):
        """Load users from JSON file."""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading users: {e}")
                return {}
        return {}
    
    def save_users(self):
        """Save users to JSON file."""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving users: {e}")
            return False
    
    def register_user(self, username, password, email=None):
        """Register a new user."""
        if not username or not password:
            return False, "Username and password are required"
        
        if username in self.users:
            return False, "Username already exists"
        
        # Hash the password
        password_hash = generate_password_hash(password)
        
        # Create user record
        self.users[username] = {
            'password_hash': password_hash,
            'email': email or '',
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'is_active': True
        }
        
        if self.save_users():
            return True, "User registered successfully"
        else:
            return False, "Error saving user data"
    
    def authenticate_user(self, username, password):
        """Authenticate a user login."""
        if not username or not password:
            return False, "Username and password are required"
        
        if username not in self.users:
            return False, "Invalid username or password"
        
        user = self.users[username]
        
        if not user.get('is_active', True):
            return False, "Account is disabled"
        
        # Check password
        if check_password_hash(user['password_hash'], password):
            # Update last login
            self.users[username]['last_login'] = datetime.now().isoformat()
            self.save_users()
            return True, "Login successful"
        else:
            return False, "Invalid username or password"
    
    def get_user_info(self, username):
        """Get user information (excluding password hash)."""
        if username in self.users:
            user_info = self.users[username].copy()
            user_info.pop('password_hash', None)
            return user_info
        return None
    
    def change_password(self, username, old_password, new_password):
        """Change user password."""
        if username not in self.users:
            return False, "User not found"
        
        user = self.users[username]
        
        # Verify old password
        if not check_password_hash(user['password_hash'], old_password):
            return False, "Current password is incorrect"
        
        # Update password
        self.users[username]['password_hash'] = generate_password_hash(new_password)
        
        if self.save_users():
            return True, "Password changed successfully"
        else:
            return False, "Error saving password change"
    
    def delete_user(self, username):
        """Delete a user account."""
        if username in self.users:
            del self.users[username]
            if self.save_users():
                return True, "User deleted successfully"
            else:
                return False, "Error deleting user"
        return False, "User not found"
    
    def list_users(self):
        """List all users (excluding password hashes)."""
        return {username: {k: v for k, v in user.items() if k != 'password_hash'} 
                for username, user in self.users.items()}
