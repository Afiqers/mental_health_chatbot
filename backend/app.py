from model import classify_text
from responses import generate_response

def chat():
    print("Mental Health Chatbot (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Take care. You're not alone.")
            break

        emotion, confidence = classify_text(user_input)
        reply = generate_response(emotion)

        print(f"Chatbot ({emotion}, {confidence:.2f}): {reply}\n")

if __name__ == "__main__":
    chat()
