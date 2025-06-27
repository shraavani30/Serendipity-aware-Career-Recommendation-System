import sqlite3

# Connect to SQLite database (creates if not exists)
conn = sqlite3.connect("career_recommendation.db")
cursor = conn.cursor()

# Create users table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE
    )
''')

# Create career test results table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS career_test_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        career TEXT,
        confidence_score REAL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
''')

# Create aptitude test results table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS aptitude_test_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        aptitude_score INTEGER,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
''')

# Create recommendations table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        recommended_career TEXT,
        serendipity_factor REAL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
''')

def connect_db():
    return sqlite3.connect("career_recommendation.db")

# Insert user
def insert_user(name, email):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name, email))
    conn.commit()
    conn.close()

# Insert career test result
def insert_career_test_result(user_id, career, confidence_score):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO career_test_results (user_id, career, confidence_score) VALUES (?, ?, ?)",
                   (user_id, career, confidence_score))
    conn.commit()
    conn.close()

# Insert aptitude test result
def insert_aptitude_test_result(user_id, aptitude_score):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO aptitude_test_results (user_id, aptitude_score) VALUES (?, ?)",
                   (user_id, aptitude_score))
    conn.commit()
    conn.close()

# Fetch user ID by email
def get_user_id(email):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email=?", (email,))
    user = cursor.fetchone()
    conn.close()
    return user[0] if user else None

# Fetch recommendations for a user
def get_recommendations(user_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT recommended_career, serendipity_factor FROM recommendations WHERE user_id=?", (user_id,))
    recommendations = cursor.fetchall()
    conn.close()
    return recommendations

conn.commit()
conn.close()
