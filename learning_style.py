def learning_style_quiz():
    print("Welcome to the Learning Style Quiz!")
    print("Please answer the following 5 questions to help determine your preferred learning style.")
    print("For each question, choose the option that best describes you.\n")
    
    # Initialize scores for each learning style
    scores = {
        "Visual": 0,
        "Auditory": 0,
        "Reading/Writing": 0,
        "Kinesthetic": 0
    }
    
    # Question 1
    print("\nQuestion 1: When trying to remember directions to a new place, you prefer:")
    print("a) Looking at a map or visual directions")
    print("b) Listening to verbal directions")
    print("c) Reading written directions")
    print("d) Walking or driving the route once to remember it")
    
    answer = input("Your answer (a/b/c/d): ").lower()
    if answer == 'a':
        scores["Visual"] += 1
    elif answer == 'b':
        scores["Auditory"] += 1
    elif answer == 'c':
        scores["Reading/Writing"] += 1
    elif answer == 'd':
        scores["Kinesthetic"] += 1
    
    # Question 2
    print("\nQuestion 2: When learning a new skill, you prefer:")
    print("a) Watching demonstrations or tutorials")
    print("b) Listening to verbal explanations")
    print("c) Reading instructions or manuals")
    print("d) Trying it out yourself through hands-on practice")
    
    answer = input("Your answer (a/b/c/d): ").lower()
    if answer == 'a':
        scores["Visual"] += 1
    elif answer == 'b':
        scores["Auditory"] += 1
    elif answer == 'c':
        scores["Reading/Writing"] += 1
    elif answer == 'd':
        scores["Kinesthetic"] += 1
    
    # Question 3
    print("\nQuestion 3: When studying for a test, you are most likely to:")
    print("a) Use diagrams, charts, or mind maps")
    print("b) Discuss the material with others or recite information aloud")
    print("c) Read notes and textbooks, or write summaries")
    print("d) Create physical models or act out concepts")
    
    answer = input("Your answer (a/b/c/d): ").lower()
    if answer == 'a':
        scores["Visual"] += 1
    elif answer == 'b':
        scores["Auditory"] += 1
    elif answer == 'c':
        scores["Reading/Writing"] += 1
    elif answer == 'd':
        scores["Kinesthetic"] += 1
    
    # Question 4
    print("\nQuestion 4: When explaining a concept to someone else, you tend to:")
    print("a) Draw a picture or diagram")
    print("b) Explain verbally with emphasis on how it sounds")
    print("c) Write it down or refer to written materials")
    print("d) Demonstrate through physical actions or analogies")
    
    answer = input("Your answer (a/b/c/d): ").lower()
    if answer == 'a':
        scores["Visual"] += 1
    elif answer == 'b':
        scores["Auditory"] += 1
    elif answer == 'c':
        scores["Reading/Writing"] += 1
    elif answer == 'd':
        scores["Kinesthetic"] += 1
    
    # Question 5
    print("\nQuestion 5: When you are bored, you are most likely to:")
    print("a) Doodle, watch videos, or look at pictures")
    print("b) Talk to someone or listen to music")
    print("c) Read a book or write something")
    print("d) Do something active or hands-on")
    
    answer = input("Your answer (a/b/c/d): ").lower()
    if answer == 'a':
        scores["Visual"] += 1
    elif answer == 'b':
        scores["Auditory"] += 1
    elif answer == 'c':
        scores["Reading/Writing"] += 1
    elif answer == 'd':
        scores["Kinesthetic"] += 1
    
    # Determine primary and secondary learning styles
    primary_style = max(scores, key=scores.get)
    
    # Create a copy of scores without the primary style
    secondary_scores = scores.copy()
    secondary_scores.pop(primary_style)
    secondary_style = max(secondary_scores, key=secondary_scores.get)
    
    # Print results
    print("\n--- Your Learning Style Results ---")
    print(f"Primary learning style: {primary_style}")
    print(f"Secondary learning style: {secondary_style}")
    print("\nScores breakdown:")
    for style, score in scores.items():
        print(f"{style}: {score}")
    
    # Provide recommendations based on primary learning style
    print("\nLearning recommendations for your primary style:")
    if primary_style == "Visual":
        print("- Use diagrams, charts, and mind maps")
        print("- Highlight notes with different colors")
        print("- Watch video demonstrations")
        print("- Visualize concepts in your mind")
    elif primary_style == "Auditory":
        print("- Record lectures and listen to them")
        print("- Discuss topics with others")
        print("- Read materials aloud")
        print("- Use verbal repetition to memorize")
    elif primary_style == "Reading/Writing":
        print("- Take detailed notes")
        print("- Rewrite information in your own words")
        print("- Create lists and summaries")
        print("- Use textbooks and written materials")
    elif primary_style == "Kinesthetic":
        print("- Use hands-on activities and experiments")
        print("- Take breaks and move around while studying")
        print("- Create physical models or role-play")
        print("- Associate concepts with physical movements")

def process_vark_answers(answers):
    """
    Process VARK answers submitted via API and return learning style results.
    
    Args:
        answers (dict): Dictionary with scores for each learning style
            {'Visual': score, 'Auditory': score, 'Reading/Writing': score, 'Kinesthetic': score}
        
    Returns:
        dict: Results with primary and secondary styles
    """
    # Find primary style
    primary_style = max(answers, key=answers.get)
    
    # Create a copy of scores without the primary style
    secondary_scores = answers.copy()
    secondary_scores.pop(primary_style)
    secondary_style = max(secondary_scores, key=secondary_scores.get)
    
    return {
        'primary_style': primary_style,
        'secondary_style': secondary_style,
        'scores': answers
    }


if __name__ == "__main__":
    learning_style_quiz()