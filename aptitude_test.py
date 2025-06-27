import streamlit as st
import matplotlib.pyplot as plt

# Define questions for the Holland Personality Test (RIASEC model)
holland_questions = {
    "R": [
        "I enjoy working with machines and tools.",
        "I like to work with numbers and solve mathematical problems.",
        "I prefer practical tasks over abstract ones."
    ],
    "I": [
        "I enjoy solving puzzles and brain teasers.",
        "I like conducting experiments and exploring new ideas.",
        "I enjoy analyzing data to find patterns and trends."
    ],
    "A": [
        "I enjoy drawing, painting, or creating visual art.",
        "I like expressing myself through music or dance.",
        "I like writing poetry or stories."
    ],
    "S": [
        "I enjoy helping people solve their problems.",
        "I like volunteering and contributing to my community.",
        "I enjoy teaching and educating others."
    ],
    "E": [
        "I enjoy taking on leadership roles and responsibilities.",
        "I like persuading and convincing others.",
        "I like organizing events and gatherings."
    ],
    "C": [
        "I prefer working with numbers and data.",
        "I like creating and following organized systems.",
        "I enjoy record-keeping and data analysis."
    ]
}

# Personality types and career recommendations
personality_info = {
    "R": {
        "name": "Realistic",
        "description": "Practical, hands-on, and enjoy working with tools and machines.",
        "careers": ["Carpenter", "Electrician", "Mechanic", "Plumber", "Welder"]
    },
    "I": {
        "name": "Investigative",
        "description": "Analytical and enjoy solving complex problems.",
        "careers": ["Scientist", "Engineer", "Researcher", "Computer Programmer", "Mathematician"]
    },
    "A": {
        "name": "Artistic",
        "description": "Creative and enjoy expressing themselves through art and design.",
        "careers": ["Artist", "Graphic Designer", "Writer", "Interior Designer", "Photographer"]
    },
    "S": {
        "name": "Social",
        "description": "Compassionate and enjoy helping and caring for others.",
        "careers": ["Teacher", "Social Worker", "Nurse", "Counselor", "Psychologist"]
    },
    "E": {
        "name": "Enterprising",
        "description": "Ambitious and enjoy leadership roles and entrepreneurship.",
        "careers": ["Entrepreneur", "Sales Manager", "Marketing Manager", "Business Consultant", "Politician"]
    },
    "C": {
        "name": "Conventional",
        "description": "Detail-oriented and enjoy organizing and managing tasks and data.",
        "careers": ["Accountant", "Financial Analyst", "Data Analyst", "Office Manager", "Banker"]
    }
}


# Function to run the aptitude test
def run_aptitude_test(username=None):
    st.title(f"Holland Personality Test - {username if username else 'Guest'}")

    st.write("Answer the following questions to determine your personality type.")

    if "responses" not in st.session_state:
        st.session_state.responses = {}

    for personality_type, questions in holland_questions.items():
        st.subheader(f"Section: {personality_info[personality_type]['name']}")

        for question in questions:
            selected_option = st.radio(question,
                                       ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
                                       key=f"{personality_type}_{question}")
            st.session_state.responses[f"{personality_type}_{question}"] = selected_option

    if st.button("Submit Test"):
        calculate_results()


# Function to calculate personality type
def calculate_results():
    score_map = {"Strongly Disagree": 1, "Disagree": 2, "Neutral": 3, "Agree": 4, "Strongly Agree": 5}

    scores = {ptype: 0 for ptype in holland_questions.keys()}

    for key, response in st.session_state.responses.items():
        personality_type = key.split("_")[0]
        scores[personality_type] += score_map[response]

    # Find the dominant personality type
    dominant_personality = max(scores, key=scores.get)

    st.session_state.dominant_personality = dominant_personality
    st.session_state.scores = scores

    show_results()


# Function to display results
def show_results():
    dominant_personality = st.session_state.dominant_personality
    scores = st.session_state.scores

    st.success(f"Your Holland Personality Type is: {personality_info[dominant_personality]['name']}")
    st.write(personality_info[dominant_personality]['description'])


    show_donut_chart(scores)


# Function to show donut chart
def show_donut_chart(scores):
    labels = list(scores.keys())
    values = list(scores.values())

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(center_circle)

    ax.axis('equal')  # Ensure the pie chart is circular
    plt.title("Aptitude Based Personality Type Distribution")

    # Show chart in Streamlit
    st.pyplot(fig)
