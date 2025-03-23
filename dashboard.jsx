import React, { useState, useEffect } from 'react';
import { auth, firestore } from '../firebase';
import { doc, getDoc, updateDoc, arrayUnion, arrayRemove } from 'firebase/firestore';
import VArkTest from './varktest'; 
import { useNavigate } from 'react-router-dom';

const Dashboard = ({ setIsLoggedIn }) => {
    const navigate = useNavigate();
    const [user, setUser] = useState(null);
    const [subjects, setSubjects] = useState([]);
    const [showAddSubject, setShowAddSubject] = useState(false);
    const [newSubject, setNewSubject] = useState({ name: '', topics: '', files: null });
    const [showVarkTest, setShowVarkTest] = useState(false);
    const [learningStyle, setLearningStyle] = useState(null);

    useEffect(() => {
        const fetchUserData = async () => {
            const currentUser = auth.currentUser;
            if (currentUser) {
                const userDoc = await getDoc(doc(firestore, 'users', currentUser.uid));
                if (userDoc.exists()) {
                    setUser(userDoc.data());
                    setSubjects(userDoc.data().subjects || []);
                    setLearningStyle(userDoc.data().learningStyle || null);
                }
            }
        };
        fetchUserData();
    }, []);

    const handleLogout = () => {
        auth.signOut();
        setIsLoggedIn(false);
    };

    const handleAddSubject = async () => {
        if (newSubject.name && newSubject.topics) {
            const topicsList = newSubject.topics.split(',').map(topic => topic.trim());
            const subjectData = {
                name: newSubject.name,
                topics: topicsList,
                progress: 0,
                files: newSubject.files ? [newSubject.files.name] : []
            };

            const currentUser = auth.currentUser;
            const userRef = doc(firestore, 'users', currentUser.uid);
            await updateDoc(userRef, {
                subjects: arrayUnion(subjectData)
            });

            setSubjects([...subjects, subjectData]);
            setNewSubject({ name: '', topics: '', files: null });
            setShowAddSubject(false);
        }
    };

    const handleDeleteSubject = async (index) => {
        // Get the subject to delete
        const subjectToDelete = subjects[index];
        
        // Remove the subject from Firestore
        const currentUser = auth.currentUser;
        const userRef = doc(firestore, 'users', currentUser.uid);
        
        // Clone the subjects array without the deleted subject
        const updatedSubjects = subjects.filter((_, i) => i !== index);
        
        try {
            // Update Firestore
            await updateDoc(userRef, {
                subjects: updatedSubjects
            });
            
            // Update local state
            setSubjects(updatedSubjects);
        } catch (error) {
            console.error("Error deleting subject:", error);
            alert("Failed to delete subject. Please try again.");
        }
    };

    const handleTakeAssessment = () => {
        setShowVarkTest(true);
    };

    const handleTestComplete = async (result) => {
        console.log("Test complete with results:", result);
        setLearningStyle(result);
        
        // Save the result to user profile in Firestore
        const currentUser = auth.currentUser;
        if (currentUser) {
            const userRef = doc(firestore, 'users', currentUser.uid);
            await updateDoc(userRef, {
                learningStyle: result
            });
        }
    };

    const getStyleRecommendations = (style) => {
        const recommendations = {
            visual: [
                "Use diagrams, charts, and mind maps",
                "Highlight notes with different colors",
                "Watch video demonstrations",
                "Visualize concepts in your mind"
            ],
            auditory: [
                "Record lectures and listen to them",
                "Discuss topics with others",
                "Read materials aloud",
                "Use verbal repetition to memorize"
            ],
            reading: [
                "Take detailed notes",
                "Rewrite information in your own words",
                "Create lists and summaries",
                "Use textbooks and written materials"
            ],
            kinesthetic: [
                "Use hands-on activities and experiments",
                "Take breaks and move around while studying",
                "Create physical models or role-play",
                "Associate concepts with physical movements"
            ]
        };
        
        return recommendations[style] || [];
    };

    const goToSubjectsPage = () => {
        navigate('/subjects');
    };

    // Return styling colors based on learning style
    const getStyleColor = (style) => {
        const colors = {
            visual: '#4a90e2',     // Blue
            auditory: '#50c878',   // Green
            reading: '#9370db',    // Purple
            kinesthetic: '#f5a623' // Orange
        };
        return colors[style] || '#777777';
    };

    return (
        <div className="dashboard">
            <header className="dashboard-header">
                <h1>LearnSmart AI Dashboard</h1>
                <button onClick={handleLogout} className="logout-btn">Logout</button>
            </header>

            <div className="assessment-section">
                <div className="assessment-card">
                    <h2>Discover Your Learning Style</h2>
                    {!learningStyle ? (
                        <>
                            <p>Take our VARK assessment to personalize your learning experience</p>
                            <button onClick={handleTakeAssessment} className="assessment-btn">Take Assessment</button>
                        </>
                    ) : (
                        <div className="learning-style-result">
                            <div className="learning-style-header" 
                                style={{backgroundColor: getStyleColor(learningStyle.dominant_style), 
                                        padding: '15px', 
                                        borderRadius: '8px 8px 0 0', 
                                        color: 'white',
                                        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'}}>
                                <h3 style={{margin: '0', fontSize: '24px'}}>
                                    Your Learning Style: {learningStyle.dominant_style.toUpperCase()}
                                </h3>
                            </div>
                            
                            <div className="style-score-container" style={{
                                backgroundColor: '#f9f9f9', 
                                padding: '20px',
                                borderRadius: '0 0 8px 8px',
                                boxShadow: '0 2px 6px rgba(0,0,0,0.08)'
                            }}>
                                <div className="style-score-breakdown" style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    flexWrap: 'wrap',
                                    marginBottom: '20px'
                                }}>
                                    {learningStyle.learning_style && Object.entries(learningStyle.learning_style).map(([style, score]) => (
                                        <div key={style} className="style-score-item" style={{
                                            width: '48%',
                                            marginBottom: '12px',
                                            padding: '10px',
                                            backgroundColor: '#fff',
                                            borderRadius: '6px',
                                            boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
                                        }}>
                                            <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '5px'}}>
                                                <span style={{fontWeight: 'bold'}}>{style.charAt(0).toUpperCase() + style.slice(1)}</span>
                                                <span style={{
                                                    backgroundColor: getStyleColor(style),
                                                    color: 'white',
                                                    padding: '2px 8px',
                                                    borderRadius: '12px',
                                                    fontSize: '14px'
                                                }}>{score}</span>
                                            </div>
                                            <div className="mini-progress-bar" style={{
                                                height: '8px',
                                                backgroundColor: '#e0e0e0',
                                                borderRadius: '4px',
                                                overflow: 'hidden'
                                            }}>
                                                <div className="mini-progress-fill" style={{ 
                                                    width: `${(score / 5) * 100}%`,
                                                    height: '100%',
                                                    backgroundColor: getStyleColor(style)
                                                }}></div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                                
                                <div className="divider" style={{height: '1px', backgroundColor: '#e0e0e0', margin: '20px 0'}}></div>
                                
                                <div className="style-recommendations" style={{
                                    backgroundColor: '#fff',
                                    padding: '15px',
                                    borderRadius: '8px',
                                    boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
                                }}>
                                    <h4 style={{
                                        color: getStyleColor(learningStyle.dominant_style),
                                        marginTop: '0',
                                        marginBottom: '10px'
                                    }}>Recommended Learning Strategies:</h4>
                                    <ul style={{
                                        paddingLeft: '20px',
                                        marginBottom: '0'
                                    }}>
                                        {getStyleRecommendations(learningStyle.dominant_style).map((rec, index) => (
                                            <li key={index} style={{marginBottom: '8px'}}>{rec}</li>
                                        ))}
                                    </ul>
                                </div>
                                
                                <button 
                                    onClick={handleTakeAssessment} 
                                    className="reassess-btn"
                                    style={{
                                        marginTop: '20px',
                                        padding: '10px 18px',
                                        backgroundColor: '#4A90E2',
                                        border: 'none',
                                        borderRadius: '6px',
                                        cursor: 'pointer',
                                        fontWeight: 'bold',
                                        display: 'block',
                                        marginLeft: 'auto',
                                        color: 'white',
                                        fontSize: '16px',
                                        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
                                        transition: 'background-color 0.3s ease',
                                    }}
                                    onMouseOver={(e) => e.target.style.backgroundColor = "#357ABD"}
                                    onMouseOut={(e) => e.target.style.backgroundColor = "#4A90E2"}
                                >
                                    Take Assessment Again
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Combined Subjects and Progress Section */}
            <div className="subjects-section">
                <div className="section-header">
                    <h2>Your Subjects</h2>
                    <div>
                        <button onClick={() => setShowAddSubject(true)} className="add-btn">+ Add Subject</button>
                        <button onClick={goToSubjectsPage} className="goto-btn">Go to Subjects</button>
                    </div>
                </div>

                {subjects.length === 0 ? (
                    <div className="empty-state">
                        <p>No subjects added yet. Add your first subject to begin!</p>
                    </div>
                ) : (
                    <div className="subjects-list">
                        {subjects.map((subject, index) => (
                            <div className="subject-item" key={index}>
                                <div className="subject-header">
                                    <h3>{subject.name}</h3>
                                    <button 
                                        onClick={() => handleDeleteSubject(index)} 
                                        className="delete-btn"
                                        style={{
                                            padding: '5px 10px',
                                            backgroundColor: '#ff4d4f',
                                            color: 'white',
                                            border: 'none',
                                            borderRadius: '4px',
                                            cursor: 'pointer'
                                        }}
                                    >
                                        Delete
                                    </button>
                                </div>
                                <div className="progress-bar" style={{ 
                                    height: '12px', 
                                    backgroundColor: '#e0e0e0', 
                                    borderRadius: '6px',
                                    margin: '10px 0',
                                    overflow: 'hidden'
                                }}>
                                    <div className="progress-fill" style={{ 
                                        width: `${subject.progress}%`,
                                        height: '100%',
                                        backgroundColor: '#4a90e2',
                                        borderRadius: '6px'
                                    }}></div>
                                </div>
                                <p style={{ marginTop: '5px' }}>{subject.progress}% Complete</p>
                                <p><strong>Topics:</strong> {subject.topics.join(', ')}</p>
                                {subject.files && subject.files.length > 0 && (
                                    <p><strong>Files:</strong> {subject.files.join(', ')}</p>
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {showAddSubject && (
                <div className="modal">
                    <div className="modal-content">
                        <h2>Add New Subject</h2>
                        <button className="close-btn" onClick={() => setShowAddSubject(false)}>×</button>
                        <div className="form-group">
                            <label>Subject Name</label>
                            <input 
                                type="text" 
                                value={newSubject.name} 
                                onChange={(e) => setNewSubject({...newSubject, name: e.target.value})} 
                            />
                        </div>
                        <div className="form-group">
                            <label>Topics (comma separated)</label>
                            <input 
                                type="text" 
                                value={newSubject.topics} 
                                onChange={(e) => setNewSubject({...newSubject, topics: e.target.value})} 
                            />
                        </div>
                        <div className="form-group">
                            <label>Upload Files</label>
                            <input 
                                type="file" 
                                onChange={(e) => setNewSubject({...newSubject, files: e.target.files[0]})} 
                            />
                        </div>
                        <button onClick={handleAddSubject} className="add-subject-btn">Add Subject</button>
                    </div>
                </div>
            )}

            {showVarkTest && (
                <VArkTest 
                    onClose={() => setShowVarkTest(false)} 
                    onTestComplete={handleTestComplete}
                />
            )}
        </div>
    );
};

export default Dashboard;