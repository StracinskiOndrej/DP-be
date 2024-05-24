import mysql.connector


def connect_to_database(host, user, password, database):
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        print("Connected to the database")
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None


# def connect_to_local():
#     return connect_to_database('localhost', 'root', 'password', 'dp')


def user_exists(connection, username=None, email=None):
    try:
        cursor = connection.cursor()
        print(username+ ", " + email)

        if email:
            # Check if email exists
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            exists = cursor.fetchone()
            print("email exists")
            if exists:
                return True, "Email already exists"
        if username:
            # Check if username exists
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            exists = cursor.fetchone()
            print("username exists")
            if exists:
                return True, "Username already exists"
            else:
                print("good ending")
                return False, "User does not exist"
        else:
            cursor.close()
            return False, "No input provided"

        cursor.close()

    except mysql.connector.Error as error:
        print("Error:", error)
        return False, "Error occurred"



def create_user(conn, username, password, email):
    try:
        cursor = conn.cursor()

        sql = "INSERT INTO USERS (username, password, email) VALUES (%s, %s, %s)"
        data = (username, password, email)

        cursor.execute(sql, data)
        conn.commit()

        print("User created successfully")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        cursor.close()



def get_user_by_email(conn, email):
    try:
        cursor = conn.cursor(dictionary=True)

        # SQL query to select user data by email
        sql = "SELECT * FROM USERS WHERE email = %s"
        data = (email,)

        cursor.execute(sql, data)
        user = cursor.fetchone()

        if user:
            print("User found:")
        else:
            print("User not found")

        return user
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None
    finally:
        cursor.close()



def close_connection(conn):
    try:
        conn.close()
        print("Connection closed")
    except mysql.connector.Error as err:
        print(f"Error: {err}")


# def ensure_connection(conn):
#     try:
#         conn.ping(True)  # This will check the connection, and try to reconnect if it's down.
#     except mysql.connector.Error as err:
#         print("Lost connection to the database. Reconnecting...")
#         conn = connect_to_local()  # Re-establish the connection
#     return conn


def get_username_by_id(conn, user_id):
    try:
        cursor = conn.cursor()

        # SQL query to select username by user ID
        sql = "SELECT username FROM USERS WHERE id = %s"
        data = (user_id,)

        cursor.execute(sql, data)
        username = cursor.fetchone()

        if username:
            return username[0]
        else:
            return None
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None
    finally:
        cursor.close()
