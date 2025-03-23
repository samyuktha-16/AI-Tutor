import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './topicsgencss.css';

const TopicDetailPopup = ({ topic, subject, learningStyle, onClose }) => {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(true);
  const [topicContent, setTopicContent] = useState(null);
  const [error, setError] = useState(null);
  const [effectiveLearningStyle, setEffectiveLearningStyle] = useState(null);

  // Process the learning style prop
  useEffect(() => {
    // Determine the effective learning style to use
    let dominantStyle = 'visual'; // Default fallback
    
    if (learningStyle) {
      // If it's a string, use it directly
      if (typeof learningStyle === 'string') {
        dominantStyle = learningStyle;
      }
      // If it's an object with dominant_style property
      else if (typeof learningStyle === 'object' && learningStyle.dominant_style) {
        dominantStyle = learningStyle.dominant_style;
      }
      // If it has a different structure with VARK scores
      else if (typeof learningStyle === 'object' && (
        learningStyle.visual || learningStyle.auditory || 
        learningStyle.reading || learningStyle.kinesthetic
      )) {
        // Find the highest score
        const scores = {
          visual: learningStyle.visual || 0,
          auditory: learningStyle.auditory || 0,
          reading: learningStyle.reading || 0,
          kinesthetic: learningStyle.kinesthetic || 0
        };
        
        let highestScore = 0;
        for (const [style, score] of Object.entries(scores)) {
          if (score > highestScore) {
            highestScore = score;
            dominantStyle = style;
          }
        }
      }
    }
    
    // Set the effective learning style
    setEffectiveLearningStyle({
      dominant_style: dominantStyle,
      secondary_style: 
        typeof learningStyle === 'object' && learningStyle.secondary_style
          ? learningStyle.secondary_style
          : '',
      visual: typeof learningStyle === 'object' && learningStyle.visual ? learningStyle.visual : 0,
      auditory: typeof learningStyle === 'object' && learningStyle.auditory ? learningStyle.auditory : 0,
      reading: typeof learningStyle === 'object' && learningStyle.reading ? learningStyle.reading : 0,
      kinesthetic: typeof learningStyle === 'object' && learningStyle.kinesthetic ? learningStyle.kinesthetic : 0
    });
    
    console.log("Using learning style:", dominantStyle);
  }, [learningStyle]);

  // Fetch content based on learning style
  useEffect(() => {
    const fetchContent = async () => {
      if (!effectiveLearningStyle) return;
      
      setIsLoading(true);
      setError(null);
      
      try {
        const response = await fetch('http://localhost:5000/api/generate-content', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            topic: topic,
            subject: subject,
            learningStyle: effectiveLearningStyle.dominant_style
          }),
        });

        if (response.ok) {
          const data = await response.json();
          
          // Ensure audio file path is properly formatted
          if (data.content && data.content.audio_file) {
            // Add leading slash if missing
            if (!data.content.audio_file.startsWith('http') && !data.content.audio_file.startsWith('/')) {
              data.content.audio_file = `/${data.content.audio_file}`;
            }
            console.log("Audio file path:", data.content.audio_file);
          }
          
          // Ensure visual asset path is properly formatted
          if (data.content && data.content.visual_asset) {
            // Log the original path
            console.log("Original visual asset path:", data.content.visual_asset);
            
            // Fix the path if needed
            if (!data.content.visual_asset.startsWith('http')) {
              // Extract filename from path
              const filename = data.content.visual_asset.split('/').pop();
              // Create direct URL to Flask server static folder
              data.content.visual_asset = `http://localhost:5000/static/images/${filename}`;
            }
            console.log("Updated visual asset path:", data.content.visual_asset);
          }
          
          // Parse and format the content if it's a string
          if (data.content && typeof data.content.content === 'string') {
            // Format the content to have better structure
            const formattedContent = formatContentWithSections(data.content.content);
            data.content.formattedContent = formattedContent;
          }
          
          setTopicContent(data.content);
        } else {
          throw new Error('Failed to fetch content');
        }
      } catch (error) {
        console.error("Error generating content:", error);
        setError("Failed to load content. Please try again.");
      } finally {
        setIsLoading(false);
      }
    };

    if (effectiveLearningStyle) {
      fetchContent();
    }
  }, [topic, subject, effectiveLearningStyle]);

  // Format the content into sections
  const formatContentWithSections = (content) => {
    // Try to identify sections in the content based on patterns
    let sections = [];
    
    // Common section identifiers
    const sectionMarkers = [
      "introduction", "key concepts", "main explanation", "examples", 
      "summary", "conclusion", "let's", "now let's", "before we"
    ];
    
    // Split content by paragraphs
    let paragraphs = content.split(/\n+/).filter(p => p.trim() !== '');
    
    let currentSection = {
      title: "Introduction",
      content: []
    };
    
    for (let i = 0; i < paragraphs.length; i++) {
      const paragraph = paragraphs[i].trim();
      
      // Check if this paragraph starts a new section
      const lowercasePara = paragraph.toLowerCase();
      const isSectionStart = sectionMarkers.some(marker => 
        lowercasePara.includes(marker) && 
        (paragraph.length < 100 || lowercasePara.startsWith(marker))
      );
      
      if (isSectionStart && currentSection.content.length > 0) {
        // Save current section
        sections.push({...currentSection});
        
        // Start new section
        currentSection = {
          title: paragraph,
          content: []
        };
      } else {
        // Add to current section
        currentSection.content.push(paragraph);
      }
    }
    
    // Add the last section
    if (currentSection.content.length > 0) {
      sections.push(currentSection);
    }
    
    // If no sections were detected, create a default structure
    if (sections.length <= 1) {
      sections = createDefaultSections(paragraphs);
    }
    
    return sections;
  };
  
  // Create default sections if automatic detection failed
  const createDefaultSections = (paragraphs) => {
    // Create a simple introduction, main content, and conclusion structure
    let sections = [];
    
    if (paragraphs.length >= 3) {
      sections = [
        {
          title: "Introduction",
          content: [paragraphs[0]]
        },
        {
          title: "Main Content",
          content: paragraphs.slice(1, paragraphs.length - 1)
        },
        {
          title: "Conclusion",
          content: [paragraphs[paragraphs.length - 1]]
        }
      ];
    } else {
      // If very short content, just use one section
      sections = [
        {
          title: "Content",
          content: paragraphs
        }
      ];
    }
    
    return sections;
  };

  // Handle navigation to test page
  const handleTakeTest = () => {
    navigate('/topic-test', { 
      state: { 
        topic, 
        subject,
        learningStyle: effectiveLearningStyle
      } 
    });
    onClose(); // Close the popup when navigating away
  };

  // Helper function to get color based on learning style
  const getLearningStyleColor = (style) => {
    const colors = {
      visual: '#4a90e2',     // Blue
      auditory: '#50c878',   // Green
      reading: '#9370db',    // Purple
      kinesthetic: '#f5a623' // Orange
    };
    return colors[style] || '#777777';
  };
  
  // Helper function to get a properly formatted URL
  const getMediaUrl = (path) => {
    if (!path) return '';
    
    // If already an absolute URL, return as is
    if (path.startsWith('http')) {
      return path;
    }
    
    // Extract filename from path
    const filename = path.split('/').pop();
    
    // Create direct URL to Flask server static folder
    // Based on the path structure, determine if it's an image or audio
    if (path.includes('images') || path.includes('visual')) {
      return `http://localhost:5000/static/images/${filename}`;
    } else if (path.includes('audio')) {
      return `http://localhost:5000/static/audio/${filename}`;
    }
    
    // Fallback to original path formatting
    const formattedPath = path.startsWith('/') ? path : `/${path}`;
    return `${window.location.origin}${formattedPath}`;
  };

  return (
    <div className="topic-popup-overlay">
      <div className="topic-popup">
        <button className="close-popup-btn" onClick={onClose}>×</button>
        
        <div className="topic-popup-header">
          <h2>{topic}</h2>
          {effectiveLearningStyle && (
            <div 
              className="learning-style-badge" 
              style={{backgroundColor: getLearningStyleColor(effectiveLearningStyle.dominant_style)}}
            >
              {effectiveLearningStyle.dominant_style.toUpperCase()} Learner
            </div>
          )}
        </div>
        
        <div className="topic-popup-content">
          {isLoading ? (
            <div className="loading-spinner">
              <div className="spinner"></div>
              <p>Generating personalized content...</p>
            </div>
          ) : error ? (
            <div className="error-message">{error}</div>
          ) : topicContent ? (
            <>
              <h3 className="topic-title">{topicContent.title}</h3>
              
              {/* Display formatted content if available, otherwise show regular content */}
              {topicContent.formattedContent ? (
                <div className="formatted-content">
                  {topicContent.formattedContent.map((section, idx) => (
                    <div key={idx} className="content-section">
                      <h4 className="section-title">{section.title}</h4>
                      {section.content.map((paragraph, pIdx) => (
                        <p key={pIdx} className="section-paragraph">{paragraph}</p>
                      ))}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="content-section">
                  <p>{topicContent.content}</p>
                </div>
              )}
              
              <div className="activities-section">
                <h4>Recommended Activities:</h4>
                <ul>
                  {topicContent.activities && topicContent.activities.map((activity, index) => (
                    <li key={index}>{activity}</li>
                  ))}
                </ul>
              </div>

              {topicContent.visual_asset && (
                <div className="visual-asset">
                  <h4>Visual Aid:</h4>
                  <img 
                    src={getMediaUrl(topicContent.visual_asset)} 
                    alt="Visual learning aid" 
                    style={{ maxWidth: '100%', height: 'auto', border: '1px solid #ddd' }}
                    onError={(e) => {
                      console.error("Image loading error:", e);
                      console.log("Failed to load image from:", e.target.src);
                      
                      // Try direct hardcoded path as a last resort
                      const originalSrc = e.target.src;
                      const filename = originalSrc.split('/').pop();
                      const fallbackUrl = `http://localhost:5000/static/images/${filename}`;
                      
                      if (originalSrc !== fallbackUrl) {
                        console.log(`Trying fallback URL: ${fallbackUrl}`);
                        e.target.src = fallbackUrl;
                        return;
                      }
                      
                      // If that fails too, show inline placeholder
                      e.target.onerror = null;
                      e.target.outerHTML = `
                        <div style="width: 400px; height: 300px; background-color: #f0f0f0; 
                                    display: flex; align-items: center; justify-content: center; 
                                    border: 1px solid #ddd; color: #666; text-align: center; 
                                    font-family: Arial, sans-serif; border-radius: 4px;">
                          <div>
                            <svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="#666" 
                                 stroke-width="2" style="margin-bottom: 10px;">
                              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                              <circle cx="8.5" cy="8.5" r="1.5"></circle>
                              <polyline points="21 15 16 10 5 21"></polyline>
                            </svg>
                            <div>Visual Aid Not Available</div>
                            <div style="font-size: 12px; margin-top: 8px;">
                              Failed to load: ${originalSrc}
                            </div>
                          </div>
                        </div>
                      `;
                    }}
                  />
                </div>
              )}

              {topicContent.audio_file && (
                <div className="audio-content">
                  <h4>Audio Explanation:</h4>
                  <div className="audio-player-container">
                    <audio 
                      controls 
                      className="audio-player"
                      src={getMediaUrl(topicContent.audio_file)}
                      autoPlay={false}
                      key={topicContent.audio_file} // Force re-render when file changes
                      onError={(e) => {
                        console.error("Audio loading error:", e);
                        console.log("Attempted audio URL:", e.target.src);
                        
                        // Try with a different URL format as a fallback
                        const audioElement = e.target;
                        const currentSrc = audioElement.src;
                        const audioFilename = topicContent.audio_file.split('/').pop();
                        
                        // Try direct URL to Flask server
                        const fallbackUrl = `http://localhost:5000/static/audio/${audioFilename}`;
                        
                        // Only update if it's a different URL to avoid infinite loops
                        if (fallbackUrl !== currentSrc) {
                          console.log(`Trying fallback URL: ${fallbackUrl}`);
                          audioElement.src = fallbackUrl;
                          audioElement.load(); // Reload the audio element
                          return;
                        }
                        
                        // If we reach here, both attempts failed
                        const audioContainer = audioElement.parentNode;
                        audioContainer.innerHTML = `
                          <div class="audio-error">
                            <p>Sorry, the audio could not be loaded. Please try again later.</p>
                            <button onclick="window.location.reload()">Retry</button>
                            <p>Failed URL: ${e.target.src}</p>
                            <p>If the problem persists, try accessing the audio directly at:</p>
                            <a href="http://localhost:5000/static/audio/${audioFilename}" target="_blank">
                              http://localhost:5000/static/audio/${audioFilename}
                            </a>
                          </div>
                        `;
                      }}
                    >
                      Your browser does not support the audio element.
                    </audio>
                    
                    <div className="audio-instructions">
                      <p>Click play to listen to the audio explanation of this topic.</p>
                    </div>
                  </div>
                </div>
              )}
            </>
          ) : (
            <p>No content available for this topic. Please try another topic.</p>
          )}
        </div>
        
        <div className="topic-popup-footer">
          <button className="take-test-btn" onClick={handleTakeTest}>Take Test on this Topic</button>
        </div>
      </div>
    </div>
  );
};

export default TopicDetailPopup;