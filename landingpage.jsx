import React, { useState } from 'react';
import { auth, firestore } from '../firebase';
import { createUserWithEmailAndPassword, signInWithEmailAndPassword } from 'firebase/auth';
import { doc, setDoc, collection } from 'firebase/firestore';
import '../App.css';

const LandingPage = ({ setIsLoggedIn }) => {
  const [showAuth, setShowAuth] = useState(false);
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleAuth = async (e) => {
    e.preventDefault();
    try {
      if (isLogin) {
        await signInWithEmailAndPassword(auth, email, password);
      } else {
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        await setDoc(doc(firestore, 'users', userCredential.user.uid), {
          email: email,
          createdAt: new Date(),
          subjects: []
        });
      }
      setIsLoggedIn(true);
    } catch (error) {
      setError(error.message);
    }
  };

  return (
    <div className="landing-page">
      <header>
        <nav>
          <div className="logo">LearnSmart AI</div>
          <button className="login-btn" onClick={() => {setShowAuth(true); setIsLogin(true)}}>Login</button>
        </nav>

        <div className="hero">
          <h1>Personalized AI Learning Experience</h1>
          <p>Adaptive learning tailored to your unique style and needs</p>
          <button className="get-started" onClick={() => {setShowAuth(true); setIsLogin(false)}}>Get Started</button>
        </div>
      </header>

      <section className="features">
        <div className="feature">
          <h3>Personalized Learning</h3>
          <p>AI adapts to your learning style</p>
        </div>
        <div className="feature">
          <h3>Interactive Lessons</h3>
          <p>Engage with gamified content</p>
        </div>
        <div className="feature">
          <h3>Track Progress</h3>
          <p>Visualize your learning journey</p>
        </div>
      </section>

      {showAuth && (
        <div className="auth-modal">
          <div className="auth-content">
            <button className="close-btn" onClick={() => setShowAuth(false)}>×</button>
            <h2>{isLogin ? 'Login' : 'Register'}</h2>
            {error && <p className="error">{error}</p>}
            <form onSubmit={handleAuth}>
              <input 
                type="email" 
                placeholder="Email" 
                value={email} 
                onChange={(e) => setEmail(e.target.value)} 
                required 
              />
              <input 
                type="password" 
                placeholder="Password" 
                value={password} 
                onChange={(e) => setPassword(e.target.value)} 
                required 
              />
              <button type="submit">{isLogin ? 'Login' : 'Register'}</button>
            </form>
            <p>
              {isLogin ? "Don't have an account? " : "Already have an account? "}
              <button 
                className="switch-auth" 
                onClick={() => setIsLogin(!isLogin)}
              >
                {isLogin ? 'Register' : 'Login'}
              </button>
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default LandingPage;