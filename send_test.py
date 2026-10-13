import requests

# YOUR CREDENTIALS
TOKEN = "8617968739:AAFndhntXBj9uJ4NMNvp5SAruibK7mimZ1k"
CHAT_ID = "1725822681"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Message sent successfully! Check your phone.")
        else:
            print(f"Failed to send message. Error: {response.text}")
    except Exception as e:
        print(f"Network error: {e}")

# Test the connection
print("Sending test message to Telegram...")
send_telegram_message("Hello! This is my AI Trading Bot speaking. I am alive! 🚀")