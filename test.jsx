import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { auth, firestore } from '../firebase';
import { doc, updateDoc, getDoc } from 'firebase/firestore';
import './test.css';

const TopicTest = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { topic, subject, learningStyle } = location.state || {};
  
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showResult, setShowResult] = useState(false);
  const [score, setScore] = useState(0);
  const [isPassed, setIsPassed] = useState(false);
  const [explanations, setExplanations] = useState({});
  const [showExplanations, setShowExplanations] = useState(false);
  
  useEffect(() => {
    const fetchQuestions = async () => {
      if (!topic || !subject) {
        navigate('/subjects');
        return;
      }
      
      setIsLoading(true);
      setError(null);
      
      try {
        const response = await fetch('http://localhost:5000/api/generate-test', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            topic: topic,
            subject: subject,
            learningStyle: learningStyle?.dominant_style || 'visual'
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to fetch test questions');
        }
        
        const data = await response.json();
        setQuestions(data.questions);
        
        // Initialize answers object with empty values
        const initialAnswers = {};
        const initialExplanations = {};
        
        data.questions.forEach(q => {
          initialAnswers[q.id] = '';
          // Store explanations from the RAG generator
          initialExplanations[q.id] = q.explanation || '';
        });
        
        setAnswers(initialAnswers);
        setExplanations(initialExplanations);
      } catch (error) {
        console.error("Error generating test:", error);
        setError(error.message || "Failed to load test. Please try again.");
      } finally {
        setIsLoading(false);
      }
    };

    fetchQuestions();
  }, [topic, subject, learningStyle, navigate]);
  
  // Handle answer selection
  const handleAnswerChange = (questionId, answer) => {
    setAnswers(prevAnswers => ({
      ...prevAnswers,
      [questionId]: answer
    }));
  };
  
  // Handle test submission
  const handleSubmit = async () => {
    // Calculate score
    let correctCount = 0;
    questions.forEach(question => {
      if (answers[question.id] === question.correctAnswer) {
        correctCount++;
      }
    });
    
    const percentage = Math.round((correctCount / questions.length) * 100);
    setScore(percentage);
    
    // Determine if passed or failed
    const passed = percentage >= 70;
    setIsPassed(passed);
    setShowResult(true);
    setShowExplanations(true);
    
    // Only update progress if user passed the test
    if (passed) {
      try {
        const currentUser = auth.currentUser;
        if (currentUser) {
          const userDocRef = doc(firestore, 'users', currentUser.uid);
          const userDoc = await getDoc(userDocRef);
          
          if (userDoc.exists()) {
            const userData = userDoc.data();
            const subjects = userData.subjects || [];
            
            // Find the subject
            const subjectIndex = subjects.findIndex(s => s.name === subject);
            
            if (subjectIndex >= 0) {
              const updatedSubjects = [...subjects];
              
              // If topic progress tracking doesn't exist, create it
              if (!updatedSubjects[subjectIndex].topicProgress) {
                updatedSubjects[subjectIndex].topicProgress = {};
              }
              
              // Update the topic progress with the test score
              updatedSubjects[subjectIndex].topicProgress[topic] = percentage;
              
              // Calculate overall subject progress
              const topicProgresses = Object.values(updatedSubjects[subjectIndex].topicProgress);
              const averageProgress = topicProgresses.reduce((a, b) => a + b, 0) / topicProgresses.length;
              updatedSubjects[subjectIndex].progress = Math.round(averageProgress);
              
              // Update Firestore
              await updateDoc(userDocRef, {
                subjects: updatedSubjects
              });
            }
          }
        }
      } catch (error) {
        console.error("Error updating progress:", error);
      }
    }
  };
  
  // Handle retaking the test
  const handleRetakeTest = () => {
    setShowResult(false);
    setShowExplanations(false);
    setAnswers({});
    // Fetch new questions for a fresh test
    setIsLoading(true);
    fetchQuestions();
  };
  
  // Function to fetch questions (extracted to reuse in retake)
  const fetchQuestions = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/generate-test', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          topic: topic,
          subject: subject,
          learningStyle: learningStyle?.dominant_style || 'visual'
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to fetch test questions');
      }
      
      const data = await response.json();
      setQuestions(data.questions);
      
      // Initialize answers object with empty values
      const initialAnswers = {};
      const initialExplanations = {};
      
      data.questions.forEach(q => {
        initialAnswers[q.id] = '';
        initialExplanations[q.id] = q.explanation || '';
      });
      
      setAnswers(initialAnswers);
      setExplanations(initialExplanations);
    } catch (error) {
      console.error("Error generating test:", error);
      setError(error.message || "Failed to load test. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };
  
  // Handle closing the result popup and returning to subjects page
  const handleCloseResult = () => {
    navigate('/subjects');
  };
  
  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Loading test questions...</p>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="error-container">
        <h2>Error</h2>
        <p>{error}</p>
        <button onClick={() => navigate('/subjects')}>Back to Subjects</button>
      </div>
    );
  }
  
  return (
    <div className="topic-test-container">
      <header className="test-header">
        <h1>Test: {topic}</h1>
        <h2>Subject: {subject}</h2>
      </header>
      
      <div className="test-content">
        {questions.map((question, index) => (
          <div key={question.id} className="question-card">
            <h3>Question {index + 1}</h3>
            <p className="question-text">{question.text}</p>
            
            <div className="options-container">
              {question.options.map((option, optIndex) => (
                <div key={optIndex} className="option">
                  <input
                    type="radio"
                    id={`q${question.id}-opt${optIndex}`}
                    name={`question-${question.id}`}
                    value={option}
                    checked={answers[question.id] === option}
                    onChange={() => handleAnswerChange(question.id, option)}
                    disabled={showResult}
                  />
                  <label 
                    htmlFor={`q${question.id}-opt${optIndex}`}
                    className={showResult ? (
                      option === question.correctAnswer ? "correct-answer" :
                      answers[question.id] === option ? "wrong-answer" : ""
                    ) : ""}
                  >
                    {option}
                  </label>
                </div>
              ))}
            </div>
            
            {showExplanations && (
              <div className="explanation-box">
                <h4>Explanation:</h4>
                <p>{explanations[question.id]}</p>
              </div>
            )}
          </div>
        ))}
      </div>
      
      <div className="test-footer">
        {!showResult && (
          <button className="submit-test-btn" onClick={handleSubmit}>Submit Test</button>
        )}
      </div>
      
      {/* Result Popup */}
      {showResult && (
        <div className="result-popup-overlay">
          <div className="result-popup">
            <h2>Test Results</h2>
            <div className="score-display">
              <div className="score-circle" style={{ 
                backgroundColor: score >= 70 ? '#50c878' : score >= 50 ? '#f5a623' : '#e74c3c' 
              }}>
                {score}%
              </div>
              <p>
                {score >= 80 ? 'Excellent!' :
                 score >= 70 ? 'Great job!' :
                 score >= 50 ? 'Good effort!' :
                 'Keep practicing!'}
              </p>
              <p style={{ fontWeight: 'bold', marginTop: '10px' }}>
                {isPassed ? 'You passed!' : 'You did not pass. Please try again.'}
              </p>
            </div>
            <div className="result-actions">
              {!isPassed ? (
                <button className="retake-btn" onClick={handleRetakeTest}>
                  Retake Test
                </button>
              ) : (
                <button className="close-result-btn" onClick={handleCloseResult}>
                  Continue
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TopicTest;