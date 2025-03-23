import React, { useState } from 'react';
import './vark.css'; // Make sure this CSS file exists

const VArkTest = ({ onClose, onTestComplete }) => {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState({
    visual: 0,
    auditory: 0,
    reading: 0,
    kinesthetic: 0
  });
  const [isComplete, setIsComplete] = useState(false);
  const [result, setResult] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const questions = [
    {
      question: "When trying to remember directions to a new place, you prefer:",
      options: [
        { text: "Looking at a map or visual directions", type: "visual" },
        { text: "Listening to verbal directions", type: "auditory" },
        { text: "Reading written directions", type: "reading" },
        { text: "Walking or driving the route once to remember it", type: "kinesthetic" }
      ]
    },
    {
      question: "When learning a new skill, you prefer:",
      options: [
        { text: "Watching demonstrations or tutorials", type: "visual" },
        { text: "Listening to verbal explanations", type: "auditory" },
        { text: "Reading instructions or manuals", type: "reading" },
        { text: "Trying it out yourself through hands-on practice", type: "kinesthetic" }
      ]
    },
    {
      question: "When studying for a test, you are most likely to:",
      options: [
        { text: "Use diagrams, charts, or mind maps", type: "visual" },
        { text: "Discuss the material with others or recite information aloud", type: "auditory" },
        { text: "Read notes and textbooks, or write summaries", type: "reading" },
        { text: "Create physical models or act out concepts", type: "kinesthetic" }
      ]
    },
    {
      question: "When explaining a concept to someone else, you tend to:",
      options: [
        { text: "Draw a picture or diagram", type: "visual" },
        { text: "Explain verbally with emphasis on how it sounds", type: "auditory" },
        { text: "Write it down or refer to written materials", type: "reading" },
        { text: "Demonstrate through physical actions or analogies", type: "kinesthetic" }
      ]
    },
    {
      question: "When you are bored, you are most likely to:",
      options: [
        { text: "Doodle, watch videos, or look at pictures", type: "visual" },
        { text: "Talk to someone or listen to music", type: "auditory" },
        { text: "Read a book or write something", type: "reading" },
        { text: "Do something active or hands-on", type: "kinesthetic" }
      ]
    }
  ];

  const handleAnswer = (type) => {
    const newAnswers = { ...answers };
    newAnswers[type] += 1;
    setAnswers(newAnswers);

    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
    } else {
      // This is the last question, submit assessment
      submitAssessment(newAnswers);
    }
  };

  const submitAssessment = async (finalAnswers) => {
    setIsSubmitting(true);
    setError(null);
    
    try {
      // For debugging
      console.log("Submitting assessment with data:", finalAnswers);
      
      // Use the correct API URL - ensure this matches your Flask server
      const API_URL = 'http://localhost:5000/api/assessment';
      
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          visual: finalAnswers.visual,
          auditory: finalAnswers.auditory,
          reading: finalAnswers.reading,
          kinesthetic: finalAnswers.kinesthetic
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Assessment response:", data);
      
      if (data.success) {
        setResult(data);
        setIsComplete(true);
        if (onTestComplete) {
          onTestComplete(data);
        }
      } else {
        setError("Failed to process assessment results.");
      }
    } catch (error) {
      console.error('Error submitting assessment:', error);
      setError("Error submitting assessment. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const getLearningStyleDescription = (style) => {
    const descriptions = {
      visual: "You learn best through visual aids like diagrams, charts, and videos. Try using color-coding, mind maps, and watching tutorials.",
      auditory: "You learn best through listening and speaking. Try recording lectures, discussing topics with others, and reading materials aloud.",
      reading: "You learn best through reading and writing. Try taking detailed notes, rewriting information, and creating lists and summaries.",
      kinesthetic: "You learn best through hands-on activities. Try experiments, role-playing, and associating concepts with physical movements."
    };
    return descriptions[style] || "";
  };

  if (error) {
    return (
      <div className="vark-test-modal">
        <div className="vark-test-content">
          <button className="close-btn" onClick={onClose}>×</button>
          <div className="error-message">
            <h3>Error</h3>
            <p>{error}</p>
            <div className="error-actions">
              <button className="retry-btn" onClick={() => submitAssessment(answers)}>
                Try Again
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (isSubmitting) {
    return (
      <div className="vark-test-modal">
        <div className="vark-test-content">
          <h2>Processing Results...</h2>
          <div className="loading-spinner"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="vark-test-modal">
      <div className="vark-test-content">
        <button className="close-btn" onClick={onClose}>×</button>
        
        {!isComplete ? (
          <>
            <h2>Learning Style Assessment</h2>
            <div className="question-progress">
              Question {currentQuestion + 1} of {questions.length}
            </div>
            
            <div className="question">
              <h3>{questions[currentQuestion].question}</h3>
              <div className="options">
                {questions[currentQuestion].options.map((option, index) => (
                  <button 
                    key={index} 
                    className="option-btn"
                    onClick={() => handleAnswer(option.type)}
                  >
                    {option.text}
                  </button>
                ))}
              </div>
            </div>
          </>
        ) : (
          <div className="test-results">
            <h2>Your Learning Style Results</h2>
            <div className="result-summary">
              <p className="dominant-style">
                Your dominant learning style is: <strong>{result.dominant_style.toUpperCase()}</strong>
              </p>
              <p>{getLearningStyleDescription(result.dominant_style)}</p>
            </div>
            
            <div className="score-breakdown">
              <h3>Score Breakdown:</h3>
              <div className="score-bars">
                {Object.entries(result.learning_style).map(([style, score]) => (
                  <div key={style} className="score-item">
                    <div className="style-label">{style.charAt(0).toUpperCase() + style.slice(1)}</div>
                    <div className="score-bar-container">
                      <div 
                        className="score-bar-fill" 
                        style={{width: `${(score / 5) * 100}%`}}
                      ></div>
                    </div>
                    <div className="score-value">{score}</div>
                  </div>
                ))}
              </div>
            </div>
            
            <button className="close-results-btn" onClick={onClose}>
              Return to Dashboard
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default VArkTest;