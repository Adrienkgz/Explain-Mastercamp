from flask_login import UserMixin

class User(UserMixin):
    def __init__(self, id):
        self.id = id

    @staticmethod
    def get(user_id):
        if user_id in users:
            return User(user_id)
        return None

# Simulated database
users = {
    'killian': {'password': 'maudet'},
    'adrien': {'password': 'jsp'}
}
