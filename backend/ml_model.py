# ml_model.py
import pickle

# Dummy predict function for demo
def predict_genre(plot: str):
    # You can replace this with your trained model later
    if "love" in plot.lower():
        return "Romance"
    elif "fight" in plot.lower() or "war" in plot.lower():
        return "Action"
    elif "magic" in plot.lower() or "sorcerer" in plot.lower():
        return "Fantasy"
    else:
        return "Drama"
