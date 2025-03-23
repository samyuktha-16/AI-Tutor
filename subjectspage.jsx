import React, { useState, useEffect } from 'react';
import { auth, firestore } from '../firebase';
import { doc, getDoc } from 'firebase/firestore';
import { useNavigate } from 'react-router-dom';
import TopicDetailPopup from './topicsgen';

const SubjectsPage = () => {
  const navigate = useNavigate();
  const [subjects, setSubjects] = useState([]);
  const [selectedSubject, setSelectedSubject] = useState(null);
  const [showTopicPopup, setShowTopicPopup] = useState(false);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [learningStyle, setLearningStyle] = useState(null);

  useEffect(() => {
    const fetchUserData = async () => {
      const currentUser = auth.currentUser;
      if (currentUser) {
        const userDoc = await getDoc(doc(firestore, 'users', currentUser.uid));
        if (userDoc.exists()) {
          const userData = userDoc.data();
          
          // Ensure that topicProgress exists for each subject
          const subjectsWithProgress = (userData.subjects || []).map(subject => {
            if (!subject.topicProgress) {
              subject.topicProgress = {};
            }
            return subject;
          });
          
          setSubjects(subjectsWithProgress);
          setLearningStyle(userData.learningStyle || null);
          
          if (subjectsWithProgress.length > 0) {
            setSelectedSubject(subjectsWithProgress[0]);
          }
        }
      } else {
        // Redirect to login if not authenticated
        navigate('/');
      }
    };
    fetchUserData();
  }, [navigate]);

  const handleSubjectClick = (subject) => {
    setSelectedSubject(subject);
  };

  const handleTopicClick = (topic) => {
    setSelectedTopic(topic);
    setShowTopicPopup(true);
  };

  const handleBackToDashboard = () => {
    navigate('/dashboard');
  };

  const handleClosePopup = () => {
    // Refresh data when popup is closed (in case a test was taken)
    fetchUserData();
    setShowTopicPopup(false);
    setSelectedTopic(null);
  };
  
  // Add this function to fetch user data
  const fetchUserData = async () => {
    const currentUser = auth.currentUser;
    if (currentUser) {
      const userDoc = await getDoc(doc(firestore, 'users', currentUser.uid));
      if (userDoc.exists()) {
        const userData = userDoc.data();
        
        // Ensure that topicProgress exists for each subject
        const subjectsWithProgress = (userData.subjects || []).map(subject => {
          if (!subject.topicProgress) {
            subject.topicProgress = {};
          }
          return subject;
        });
        
        setSubjects(subjectsWithProgress);
        
        // Update the selected subject if it exists
        if (selectedSubject) {
          const updatedSelectedSubject = subjectsWithProgress.find(
            s => s.name === selectedSubject.name
          );
          if (updatedSelectedSubject) {
            setSelectedSubject(updatedSelectedSubject);
          }
        }
      }
    }
  };

  return (
    <div className="subjects-page">
      <header className="subjects-header">
        <h1>Your Subjects</h1>
        <button onClick={handleBackToDashboard} className="back-btn">Back to Dashboard</button>
      </header>

      <div className="subjects-container">
        <div className="subjects-sidebar">
          <h2>All Subjects</h2>
          <ul className="subjects-list">
            {subjects.map((subject, index) => (
              <li 
                key={index} 
                className={selectedSubject && selectedSubject.name === subject.name ? 'active' : ''}
                onClick={() => handleSubjectClick(subject)}
              >
                {subject.name}
              </li>
            ))}
          </ul>
        </div>

        <div className="subject-details">
          {selectedSubject ? (
            <>
              <div className="subject-header">
                <h2>{selectedSubject.name}</h2>
                <div className="circular-progress">
                  <svg viewBox="0 0 36 36" className="circular-chart">
                    <path 
                      className="circle-bg"
                      d="M18 2.0845
                        a 15.9155 15.9155 0 0 1 0 31.831
                        a 15.9155 15.9155 0 0 1 0 -31.831"
                    />
                    <path 
                      className="circle"
                      strokeDasharray={`${selectedSubject.progress}, 100`}
                      d="M18 2.0845
                        a 15.9155 15.9155 0 0 1 0 31.831
                        a 15.9155 15.9155 0 0 1 0 -31.831"
                    />
                    <text x="18" y="20.35" className="percentage">{selectedSubject.progress}%</text>
                  </svg>
                </div>
              </div>

              <div className="topics-section">
                <h3>Topics</h3>
                <div className="topics-grid">
                  {selectedSubject.topics.map((topic, index) => {
                    // Get the topic progress from the topicProgress object
                    const topicProgress = selectedSubject.topicProgress[topic] || 0;
                    
                    return (
                      <div key={index} className="topic-card" onClick={() => handleTopicClick(topic)}>
                        <h4>{topic}</h4>
                        <div className="progress-bar">
                          <div className="progress-fill" style={{ width: `${topicProgress}%` }}></div>
                        </div>
                        <p>{topicProgress}% Complete</p>
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          ) : (
            <div className="empty-state">
              <p>Select a subject to view details</p>
            </div>
          )}
        </div>
      </div>

      {showTopicPopup && selectedTopic && (
        <TopicDetailPopup
          topic={selectedTopic}
          subject={selectedSubject ? selectedSubject.name : ''}
          learningStyle={learningStyle}
          onClose={handleClosePopup}
        />
      )}
    </div>
  );
};

export default SubjectsPage;