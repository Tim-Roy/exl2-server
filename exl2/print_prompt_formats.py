from exl2.chat.chat_prompts import get_available_chat_formats

if __name__ == "__main__":
    for format in get_available_chat_formats():
        print(format)
